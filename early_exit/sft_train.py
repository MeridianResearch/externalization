import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader

from shared_utils.data import CSVPromptDataset
from shared_utils.load import get_model, get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text

from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode

import wandb


# LOAD IN EXPERIMENT ARGS
num_epoch = 1                     # args.num_epoch
num_exit_samples = 4                  # args.num_exit_samples
device = "cuda"                    # args.device
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"                    # args.model_name
model_config_path = "config_deepseek.yaml"                     # args.model_config_path
dataset_path = "results_and_data/early_exit_sft_dataset/test/data.csv"                  # args.dataset_path
prompt_config_path = "results_and_data/early_exit_sft_dataset/test/prompt_config.json"                    # args.prompt_config_path
batch_size = 1                    # args.batch_size -- might want to sort out batching, but increasing num_exit_samples might be better + less effort

args = {
    'num_epoch': num_epoch,
    'num_exit_samples': num_exit_samples,
    'device': device,
    'model_name': model_name,
    'model_config_path': model_config_path,
    'dataset_path': dataset_path,
    'prompt_config_path': prompt_config_path,
    'batch_size': batch_size,
}


# LOAD IN THE MODEL AND TOKENIZER
tokenizer = get_tokenizer(model_name)
config = configs_from_yaml(model_config_path, tokenizer.eos_token_id)
model = get_model(model_name, config['model'], device)


# LOAD IN DATASET
dataset = CSVPromptDataset(dataset_path, prompt_config_path)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=True)


# ENABLE EARLY EXITING
model = replace_attention_layers(model, config['lora'], device)
model.train()

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

    for prompt_batch in dataloader:

        batch_ticker += 1

        with torch.no_grad():
            # Generate SFT targets
            set_transformer_early_exit_mode(model, 'sft_teacher')
            sft_teacher_response, (sft_teacher_generated_tokens, sft_teacher_final_layer_logprobs, gathered_early_exit_hidden_states) =\
                generate_text(
                    model=model, 
                    prompt=prompt_batch.full_user_prompt, 
                    system_prompt=dataset.system_prompt, 
                    prefiller=dataset.prefiller, 
                    tokenizer=tokenizer, 
                    generation_config=config['generation'], 
                    device=device
                )
            print(sft_teacher_response)

            early_output_log_probs = model.early_exit_hidden_state_readout(gathered_early_exit_hidden_states)               # [batch, num exitable layers, gen len, vocabulary]
            early_exit_probs = model.early_exit_target_probs(early_output_log_probs = early_output_log_probs, teacher_final_layer_log_probs = sft_teacher_final_layer_logprobs)
            repeated_sft_teacher_final_layer_logprobs = sft_teacher_final_layer_logprobs.repeat(num_exit_samples, 1, 1)     # XXX repeat_interleave? [batch * samples, full length, vocabulary]


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
        sft_student_probs = sft_student_output_scores.logits[:,-gen_len:].softmax(-1)           # [batch * samples, gen len, vocabulary]
        token_logits_kl_div = (sft_student_probs * ((sft_student_probs + eps) / (sft_teacher_probs + eps)).log()).sum(-1)   # [batch * samples, gen len]
        mean_logit_kl = token_logits_kl_div.mean()

        # Get KL divergences of early exit preds
        print('CRUDE KL AND MAKE SURE PROBS ARE ALIGNED AGAIN')
        eps = 1e-16
        sft_student_early_exit_probs = model.early_exit_student_probs(collected_exit_logits)            # [batch, gen len, layers + 1]
        mean_exit_logprob = - (sft_student_early_exit_probs + 1e-16).gather(index = sampled_early_exit_layer_idxs_early.unsqueeze(-1), dim = 2).log().mean()  # [batch, gen len, 1] -> scalar

        optimiser.zero_grad()
        total_loss = mean_logit_kl + mean_exit_logprob
        total_loss.backward()
        optimiser.step()

        torch.cuda.empty_cache()

        # Package and log
        with torch.no_grad():

            log_dict = {
                'epoch': epoch,
                'batch_in_epoch': batch_ticker,
                'prompt_idx': prompt_batch.idx[0],
                'mean_logit_kl': mean_logit_kl.item(),
                'mean_exit_logprob': mean_exit_logprob.item(),
                'total_loss': total_loss.item(),
            }

            # Probability of exiting, according to the teacher
            layer_mean_exit_probabilities = early_exit_probs.mean(0).mean(0)  # shape: [layers]
            log_dict.update({
                f'layer_mean_exit_probabilities/layer_{model.exitable_layer_idxs[i]}': v.item() 
                for i, v in enumerate(layer_mean_exit_probabilities)
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

            assert len(prompt_batch.idx) == 1, "Again, batch greater than 1 not allowed yet"
            wandb.log(log_dict)