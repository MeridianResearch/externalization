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
model_config_path = "data/config_deepseek.yaml"                     # args.model_config_path
dataset_path = "data/data_short.csv"                  # args.dataset_path
prompt_config_path = "data/prompt_config.json"                    # args.prompt_config_path
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
            print(f"Teacher response: {sft_teacher_response}")
            print(f"Generated tokens shape: {sft_teacher_generated_tokens.shape}, dtype: {sft_teacher_generated_tokens.dtype}")
            print(f"Final layer logprobs shape: {sft_teacher_final_layer_logprobs.shape}, dtype: {sft_teacher_final_layer_logprobs.dtype}")  # [batch, gen len, vocabulary]
            print(f"Gathered early exit hidden states shape: {gathered_early_exit_hidden_states.shape}, dtype: {gathered_early_exit_hidden_states.dtype}")  # [batch, num exitable layers, gen len, hidden size]

            early_output_log_probs = model.early_exit_hidden_state_readout(gathered_early_exit_hidden_states)               # [batch, num exitable layers, gen len, vocabulary]
            print(f"Early output log probs shape: {early_output_log_probs.shape}, dtype: {early_output_log_probs.dtype}")
            
            early_exit_probs = model.early_exit_target_probs(early_output_log_probs = early_output_log_probs, teacher_final_layer_log_probs = sft_teacher_final_layer_logprobs)
            print(f"Early exit probs shape: {early_exit_probs.shape}, dtype: {early_exit_probs.dtype}")
            print(f"Early exit probs range: [{early_exit_probs.min():.6f}, {early_exit_probs.max():.6f}]")
            
            repeated_sft_teacher_final_layer_logprobs = sft_teacher_final_layer_logprobs.repeat(num_exit_samples, 1, 1)     # XXX repeat_interleave? [batch * samples, full length, vocabulary]
            print(f"Repeated teacher logprobs shape: {repeated_sft_teacher_final_layer_logprobs.shape}")



        # Sample early exits
        batch, gen_len, elayers = early_exit_probs.shape                                                                                                # [batch, generation length, exitable layers]
        full_len = sft_teacher_generated_tokens.shape[1]
        print(f"Full sequence length: {full_len}")
        repeated_sft_teacher_generated_tokens = sft_teacher_generated_tokens.expand(num_exit_samples * batch, full_len) # [batch * samples, full length]
        print(f"Repeated teacher tokens shape: {repeated_sft_teacher_generated_tokens.shape}")
        sampled_early_exit_layer_idxs_early_with_sample_dim = torch.distributions.Categorical(probs = early_exit_probs).sample((num_exit_samples,))     # [samples, batch, generation length] 
        print(f"Sampled early exit indices (with sample dim) shape: {sampled_early_exit_layer_idxs_early_with_sample_dim.shape}")
        print(f"Sample of sampled indices: {sampled_early_exit_layer_idxs_early_with_sample_dim[0, 0, :10]}")
        
        sampled_early_exit_layer_idxs_early = sampled_early_exit_layer_idxs_early_with_sample_dim.reshape(batch * num_exit_samples, gen_len)            # [batch * samples, generation length]
        print(f"Reshaped sampled indices shape: {sampled_early_exit_layer_idxs_early.shape}")

        sampled_early_exit_layer_idxs = model.exitable_layer_idxs[sampled_early_exit_layer_idxs_early.cpu()]                                            # [batch * samples, generation length]
        print(f"Final sampled layer indices shape: {sampled_early_exit_layer_idxs.shape}")
        print(f"Unique sampled layer indices: {torch.unique(sampled_early_exit_layer_idxs)}")
        

        # Generate with prescription
        set_transformer_early_exit_mode(model, 'sft_student')
        sft_student_output_scores, collected_exit_logits = model(repeated_sft_teacher_generated_tokens, prescribed_exit_layer_idxs = sampled_early_exit_layer_idxs) # [batch * samples, full length, vocabulary]
        print(f"Student output scores shape: {sft_student_output_scores.logits.shape}")
        print(f"Collected exit logits shape: {collected_exit_logits.shape}")

        # Get KL divergences of the outputs
        print('CRUDE KL AND MAKE SURE PROBS ARE ALIGNED')
        eps = 1e-16
        sft_teacher_probs = sft_teacher_final_layer_logprobs.softmax(-1)                        # [batch * samples, gen len, vocabulary]
        print(f"Teacher probs shape: {sft_teacher_probs.shape}")
        print(f"Teacher probs range: [{sft_teacher_probs.min():.6f}, {sft_teacher_probs.max():.6f}]")
        print(f"Teacher probs sum along vocab: {sft_teacher_probs.sum(-1)[0, :5]}")  # Should be ~1.0
        print(f"Teacher probs entropy (first 5 tokens): {-(sft_teacher_probs * (sft_teacher_probs + eps).log()).sum(-1)[0, :5]}")
        
        sft_student_probs = sft_student_output_scores.logits[:,-gen_len:].softmax(-1)           # [batch * samples, gen len, vocabulary]
        print(f"Student probs shape: {sft_student_probs.shape}")
        print(f"Student probs range: [{sft_student_probs.min():.6f}, {sft_student_probs.max():.6f}]")
        print(f"Student probs sum along vocab: {sft_student_probs.sum(-1)[0, :5]}")  # Should be ~1.0
        print(f"Student probs entropy (first 5 tokens): {-(sft_student_probs * (sft_student_probs + eps).log()).sum(-1)[0, :5]}")
        # Check alignment by comparing expected tokens
        teacher_top_tokens = sft_teacher_probs.argmax(-1)
        student_top_tokens = sft_student_probs.argmax(-1)
        token_alignment = (teacher_top_tokens == student_top_tokens).float().mean()
        print(f"Token alignment (top-1 match): {token_alignment:.4f}")
        print(f"Teacher top tokens (first 10): {teacher_top_tokens[0, :10]}")
        print(f"Student top tokens (first 10): {student_top_tokens[0, :10]}")
        
        # Detailed KL computation
        ratio = (sft_student_probs + eps) / (sft_teacher_probs + eps)
        log_ratio = ratio.log()
        kl_per_token_per_vocab = sft_student_probs * log_ratio
        token_logits_kl_div = (sft_student_probs * ((sft_student_probs + eps) / (sft_teacher_probs + eps)).log()).sum(-1)   # [batch * samples, gen len]
        print(f"Ratio stats - min: {ratio.min():.6f}, max: {ratio.max():.6f}, mean: {ratio.mean():.6f}")
        print(f"Log ratio stats - min: {log_ratio.min():.6f}, max: {log_ratio.max():.6f}, mean: {log_ratio.mean():.6f}")
        print(f"KL per token per vocab stats - min: {kl_per_token_per_vocab.min():.6f}, max: {kl_per_token_per_vocab.max():.6f}")
        
        print(f"Token logits KL div shape: {token_logits_kl_div.shape}, dtype: {token_logits_kl_div.dtype}")  # [batch * samples, gen len]
        print(f"Token logits KL div range: [{token_logits_kl_div.min():.6f}, {token_logits_kl_div.max():.6f}]")
        print(f"Token logits KL div mean per position: {token_logits_kl_div.mean(0)[:10]}")
        print(f"Token logits KL div sample (first sample, first 10 tokens): {token_logits_kl_div[0, :10]}")
        
        mean_logit_kl = token_logits_kl_div.mean()
        print(f"Mean logit KL: {mean_logit_kl:.6f}")

        # Get KL divergences of early exit preds
        print('CRUDE KL AND MAKE SURE PROBS ARE ALIGNED AGAIN')
        eps = 1e-16
        sft_student_early_exit_probs = model.early_exit_student_probs(collected_exit_logits)            # [batch, gen len, layers + 1]
        print(f"Student early exit probs shape: {sft_student_early_exit_probs.shape}")
        print(f"Student early exit probs range: [{sft_student_early_exit_probs.min():.6f}, {sft_student_early_exit_probs.max():.6f}]")
        print(f"Student early exit probs sum along layers: {sft_student_early_exit_probs.sum(-1)[0, :5]}")  # Should be ~1.0
        
        # Check which layers are being sampled
        print(f"Sampled early exit layer indices (early) shape: {sampled_early_exit_layer_idxs_early.shape}")
        print(f"Sampled indices range: [{sampled_early_exit_layer_idxs_early.min()}, {sampled_early_exit_layer_idxs_early.max()}]")
        print(f"Sampled indices histogram: {torch.bincount(sampled_early_exit_layer_idxs_early.flatten())}")
        
        gathered_probs = (sft_student_early_exit_probs + 1e-16).gather(index = sampled_early_exit_layer_idxs_early.unsqueeze(-1), dim = 2)
        print(f"Gathered probs shape: {gathered_probs.shape}")
        print(f"Gathered probs range: [{gathered_probs.min():.6f}, {gathered_probs.max():.6f}]")
        print(f"Gathered probs mean: {gathered_probs.mean():.6f}")
        print(f"Gathered probs sample (first 10): {gathered_probs.squeeze()[0, :10]}")
        
        # Check log computation
        log_gathered_probs = gathered_probs.log()
        print(f"Log gathered probs range: [{log_gathered_probs.min():.6f}, {log_gathered_probs.max():.6f}]")
        print(f"Log gathered probs mean: {log_gathered_probs.mean():.6f}")
        
        mean_exit_logprob = - (sft_student_early_exit_probs + 1e-16).gather(index = sampled_early_exit_layer_idxs_early.unsqueeze(-1), dim = 2).log().mean()  # [batch, gen len, 1] -> scalar
        print(f"Mean exit log prob: {mean_exit_logprob:.6f}")

        # Additional exit probability analysis per layer
        print("\n--- PER-LAYER EXIT PROBABILITY ANALYSIS ---")
        for layer_idx in range(len(model.exitable_layer_idxs)):
            layer_mask = sampled_early_exit_layer_idxs_early == layer_idx
            if layer_mask.sum() > 0:
                layer_probs = gathered_probs[layer_mask]
                layer_log_probs = log_gathered_probs[layer_mask]
                print(f"Layer {model.exitable_layer_idxs[layer_idx].item()}: {layer_mask.sum()} samples")
                print(f"  Prob range: [{layer_probs.min():.6f}, {layer_probs.max():.6f}], mean: {layer_probs.mean():.6f}")
                print(f"  Log prob range: [{layer_log_probs.min():.6f}, {layer_log_probs.max():.6f}], mean: {layer_log_probs.mean():.6f}")
            else:
                print(f"Layer {model.exitable_layer_idxs[layer_idx].item()}: 0 samples")

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

