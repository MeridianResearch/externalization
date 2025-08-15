import torch
import gzip
import pickle
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader

#from shared_utils.data import CSVPromptDataset
from shared_utils.load import get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text

from early_exit.util import get_model, save_model

from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode

import wandb
from datetime import datetime


# LOAD IN EXPERIMENT ARGS
num_epoch = 1                     # args.num_epoch
num_exit_samples = 4                  # args.num_exit_samples
device = "cuda"                    # args.device
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"                    # args.model_name
model_config_path = "config_deepseek.yaml"                     # args.model_config_path
#dataset_path = "results_and_data/early_exit_sft_dataset/test/data.csv"                  # args.dataset_path
#prompt_config_path = "results_and_data/early_exit_sft_dataset/test/prompt_config.json"                    # args.prompt_config_path
# teacher_data_path = "/workspace/data/teacher_generated_data_gzip/merged_teacher_data_sparse.pkl.gz" #update location - on runpod moving to workspace allowed for more disc space
teacher_data_path = "results_and_data/early_exit_sft_dataset/test/merged_teacher_data_sparse.pkl.gz" # maybe we can move to config?
batch_size = 1                    # args.batch_size -- might want to sort out batching, but increasing num_exit_samples might be better + less effort

save_freq = 250

args = {
    'num_epoch': num_epoch,
    'num_exit_samples': num_exit_samples,
    'device': device,
    'model_name': model_name,
    'model_config_path': model_config_path,
    'batch_size': batch_size,
}


# LOAD IN THE MODEL AND TOKENIZER
tokenizer = get_tokenizer(model_name)
config = configs_from_yaml(model_config_path, tokenizer.eos_token_id)
model = get_model(model_name, config['model'], device)

# LOAD IN DATASET - now with streaming and metadata header
def iter_merged_teacher_data(merged_path: str):
    """Lazily iterate samples from the merged stream."""
    with gzip.open(merged_path, "rb") as f:
        header = pickle.load(f)  # {'metadata': ...}
        while True:
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            if isinstance(obj, dict) and obj.get('_end'):
                break
            yield obj


# ENABLE EARLY EXITING
model = replace_attention_layers(model, config['lora'], device)
model.train()

save_dir = f"models/early_exit_{datetime.now().strftime('%Y%m%d')}_layers_{model.total_exitable_layers}_big"

optimiser = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)


run = wandb.init(
    # entity="cot-mrc",
    project="early-exit",
    config=dict(
        **config,
        args=args,
        model_exitable_layers=model.exitable_layer_idxs.tolist()
    )
)


batch_ticker = 0

for epoch in range(num_epoch):

    #for prompt_batch in dataloader:
    for teacher_batch in iter_merged_teacher_data(teacher_data_path):

        batch_ticker += 1
        with torch.no_grad():
            set_transformer_early_exit_mode(model, 'sft_teacher')

            prompt_idx = teacher_batch['prompt_idx']
            full_user_prompt = teacher_batch['full_user_prompt']
            sft_teacher_response = teacher_batch['sft_teacher_response']
            sft_teacher_generated_tokens = teacher_batch['sft_teacher_generated_tokens'].to(device)
            kl_div_per_layer = teacher_batch['kl_div1_per_layer'].to(device)
            current_exitable_indices = model.exitable_layer_idxs[:-1].long()
            kl_div_per_layer = kl_div_per_layer[:, current_exitable_indices, :]
        
            sft_teacher_final_layer_logprobs = teacher_batch['sft_teacher_final_layer_logprobs'].to(device)
            if sft_teacher_final_layer_logprobs.is_sparse:
                dense = sft_teacher_final_layer_logprobs.to_dense()          
                dense[dense == 0] = -14.0 # restore safe floor
                sft_teacher_final_layer_logprobs = dense
            
        #exitable_layer_idxs = teacher_batch['exitable_layer_idxs'].to(device)

        #direct from model mixins
        
            KL_FACTOR = 1.0  #currently unchanged
        
            sigmoid_kls = torch.sigmoid(KL_FACTOR * kl_div_per_layer)  # [batch, num_layers, seq_len]
            sigmoid_kls = 2.0 * sigmoid_kls - 1.0
            sigmoid_kls = 1.0 - sigmoid_kls
        
            batch_size, num_layers, seq_len = sigmoid_kls.shape
            stickbreaking_probs = torch.zeros(batch_size, num_layers + 1, seq_len, device = sigmoid_kls.device)
        
            for l in range(num_layers):
                if l == 0:
                    prod_term = torch.ones((batch_size, seq_len), device = sigmoid_kls.device)
                else:
                    prod_term = torch.prod(1 - sigmoid_kls[:, :l, :], dim=1)
                stickbreaking_probs[:, l, :] = sigmoid_kls[:, l, :] * prod_term
            
            stickbreaking_probs[:, -1, :] = torch.prod(1 - sigmoid_kls, dim=1)
            
            early_exit_probs = stickbreaking_probs.permute(0, 2, 1) #transpose for [batch, seq_len, num_layers+1]

        ##rest is as is

            repeated_sft_teacher_final_layer_logprobs = sft_teacher_final_layer_logprobs.repeat(num_exit_samples, 1, 1)     # XXX repeat_interleave? [batch * samples, full length, vocabulary]
            sft_teacher_generated_tokens = sft_teacher_generated_tokens[:, :-1] # Removing the last token to match the log probs and hidden states shape


        # Sample early exits
        batch, gen_len, elayers = early_exit_probs.shape                                                                                                # [batch, generation length, exitable layers]
        full_len = sft_teacher_generated_tokens.shape[1]
        repeated_sft_teacher_generated_tokens = sft_teacher_generated_tokens.expand(num_exit_samples * batch, full_len)                                 # [batch * samples, full length]

        sampled_early_exit_layer_idxs_early_with_sample_dim = torch.distributions.Categorical(probs = early_exit_probs).sample((num_exit_samples,))     # [samples, batch, generation length] 
        sampled_early_exit_layer_idxs_early = sampled_early_exit_layer_idxs_early_with_sample_dim.reshape(batch * num_exit_samples, gen_len)            # [batch * samples, generation length]
        sampled_early_exit_layer_idxs = model.exitable_layer_idxs[sampled_early_exit_layer_idxs_early.cpu()]                                            # [batch * samples, generation length]     

        # Generate with prescription
        set_transformer_early_exit_mode(model, 'sft_student')
        sft_student_output_scores, collected_exit_logits = model(repeated_sft_teacher_generated_tokens, prescribed_exit_layer_idxs = sampled_early_exit_layer_idxs) # [batch * samples, full length, vocabulary]

        # Get KL divergences of the outputs
        print('CRUDE KL AND MAKE SURE PROBS ARE ALIGNED')
        eps = 1e-16
        sft_teacher_probs = sft_teacher_final_layer_logprobs.softmax(-1)                        # [batch * samples, gen len, vocabulary]
        #sft_student_probs = sft_student_output_scores.logits[:,-gen_len-1:-1].softmax(-1)           # [batch * samples, gen len, vocabulary]
        sft_student_probs = sft_student_output_scores.logits[:,-gen_len:].softmax(-1)
        token_logits_kl_div = (sft_student_probs * ((sft_student_probs + eps) / (sft_teacher_probs + eps)).log()).sum(-1)   # [batch * samples, gen len]
        mean_logit_kl = token_logits_kl_div.mean()
        
        # For plotting: KL divergence using (r-1) - log(r) approximation
        ratio_for_plot = (sft_student_probs + eps) / (sft_teacher_probs + eps)
        token_logits_kl_div_plot = (sft_student_probs * ((ratio_for_plot - 1) - ratio_for_plot.log())).sum(-1)
        mean_logit_kl_plot = token_logits_kl_div_plot.mean()

        # Get KL divergences of early exit preds
        print('CRUDE KL AND MAKE SURE PROBS ARE ALIGNED AGAIN')
        eps = 1e-16
        sft_student_early_exit_probs = model.early_exit_student_probs(collected_exit_logits)            # [batch, gen len, layers + 1]
        mean_exit_logprob = - (sft_student_early_exit_probs + 1e-16).gather(index = sampled_early_exit_layer_idxs_early.unsqueeze(-1), dim = 2).log().mean()  # [batch, gen len, 1] -> scalar

        optimiser.zero_grad()
        total_loss = mean_logit_kl + mean_exit_logprob
        total_loss.backward()
        optimiser.step()
        
        early_exit_probs_repeated = early_exit_probs.repeat(num_exit_samples, 1, 1)
        sampled_early_exit_layer_idxs_early_expanded = sampled_early_exit_layer_idxs_early.unsqueeze(-1).to(early_exit_probs_repeated.device)  
        sampled_early_exit_probs = early_exit_probs_repeated.gather(dim=2, index=sampled_early_exit_layer_idxs_early_expanded)  
        prob_diff = torch.abs(sampled_early_exit_probs - (sft_student_early_exit_probs).gather(index = sampled_early_exit_layer_idxs_early.unsqueeze(-1), dim = 2))  # [samples, batch, gen len, layers + 1]
  
        torch.cuda.empty_cache()

        # Package and log
        with torch.no_grad():

            log_dict = {
                # 'epoch': epoch,
                'batch_in_epoch': batch_ticker,
                'prompt_idx': prompt_idx,
                'student_teacher_token_kl_divergence': mean_logit_kl_plot.item(),
                'early_exit_decision_loss': mean_exit_logprob.item(),
                'total_loss': total_loss.item(),
                'sampled_exit_prob_diff': prob_diff.mean().item(),
            }

            # Probability of exiting, according to the teacher
            layer_mean_exit_probabilities = early_exit_probs.mean(0).mean(0)  # shape: [layers]
            log_dict.update({
                f'layer_mean_exit_probabilities_teacher/layer_{model.exitable_layer_idxs[i]}': v.item() 
                for i, v in enumerate(layer_mean_exit_probabilities)
            })

            # Probability of exiting, according to the student
            layer_mean_exit_probabilities_student = sft_student_early_exit_probs.mean(0).mean(0)  # shape: [layers]
            log_dict.update({
                f'layer_mean_exit_probabilities_student/layer_{model.exitable_layer_idxs[i]}': v.item() 
                for i, v in enumerate(layer_mean_exit_probabilities_student)
            })

            # Empirical smapling rate of each layer exit option
            log_dict.update({
                f'layer_empirical_exit_props/layer_{idx.item()}': (sampled_early_exit_layer_idxs == idx).sum() / sampled_early_exit_layer_idxs.numel()
                for idx in model.exitable_layer_idxs
            })
            
            # KL divergence of output when student forced to exit at each layer
            log_dict.update({
                f"mean_token_logits_kl_div_per_prescribed_early_exit/layer_{round(model.exitable_layer_idxs[ei].item(), 0)}": 
                    token_logits_kl_div[sampled_early_exit_layer_idxs_early == ei].mean().item() 
                for ei in range(len(model.exitable_layer_idxs))
            })

            assert batch == 1, "Again, batch greater than 1 not allowed yet"
            wandb.log(log_dict)

        if batch_ticker % save_freq == 0:
            checkpoint_path = f"{save_dir}/step_{batch_ticker}"
            save_model(model, checkpoint_path, upload_to_wandb=False)
            print(f"Checkpoint saved to {checkpoint_path}")

    print(f"\nEpoch {epoch+1} completed!")

print(f"Saving final model to {save_dir}...")
save_model(model, save_dir)

wandb.finish()