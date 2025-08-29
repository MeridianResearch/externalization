"""


- Flow: K rollouts per prompt → compute rewards (verify - beta*KL - lambda*avg_exit_layer) → center per-prompt → weighted SFT.
"""

import torch
from torch.optim import Adam
import wandb
from datasets import load_dataset
from typing import Optional

from early_exit.util import get_model, load_model_from_wandb, load_model
from early_exit.rl_utils import generate_k_completions, center_rewards_per_prompt, map_layers_to_indices, weighted_sft_step, get_input_prompt_length
from early_exit.rl_types import RLHyperparams, RolloutBatch
from early_exit.rewards import compute_verification_rewards, compute_token_kl_from_logprobs, compute_token_logprobs_reference, compute_token_logprobs_student, compute_avg_exit_layer, extract_solution
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode
from shared_utils.load import get_tokenizer, configs_from_yaml
from torch.nn.utils.rnn import pad_sequence

device = "cuda"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
config_path = "config_deepseek.yaml"
sft_model_path = "models/gsm_8k_model"  # TODO: set path to SFT checkpoint

RL_HPARAMS = RLHyperparams()


# --- Models (schema) ---
tokenizer = get_tokenizer(model_name)
config = configs_from_yaml(config_path, tokenizer.eos_token_id)

student = get_model(model_name, config['model'], device)
student = replace_attention_layers(student, config['lora'], device)
# TODO: Change artifact path to sft trained gsm-8k model
#student = load_model_from_wandb(student, model_path = "models/trained_model_v0", artifact_path = 'vkarthik095-university-of-amsterdam/early-exit/early-exit-model-fs5ofmzp:v0')
student = load_model(student, sft_model_path)

# Reference policy: base unmodified model without early exit
reference = get_model(model_name, config['model'], device)
# TODO: ensure no early-exit logic is active for reference model

# Dataset
dataset = load_dataset("gsm8k", "main")  # TODO: verify/parse answer format


def main_rl_training():
    """
    Schema: Generate → Reward → Center → Weighted SFT
    """
    # TODO: optimizer (e.g., Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=1e-5))
    # Check if there are better optimizers for this problem
    optimizer = Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=1e-5)
    # we use https://huggingface.co/docs/trl/rloo_trainer  as an inspiration for logging. 

    run = wandb.init(
        project="early-exit-RL-test",
        entity="vkarthik095-university-of-amsterdam",
        config=dict(
            **config,
            rl_hparams=vars(RL_HPARAMS),
            model_exitable_layers=getattr(student, 'exitable_layer_idxs', []).tolist() if hasattr(student, 'exitable_layer_idxs') else None,
            metric_descriptions={
                # Objective & rewards
                'objective/rlhf_reward': 'Mean total reward per step',
                'objective/kl': 'Mean token KL vs reference policy',
                'objective/non_score_reward': 'Mean of penalty terms (KL + exit)',
                'rewards/verify_mean': 'Mean task verification reward',
                'rewards/kl_penalty_component_mean': 'Mean KL penalty contribution',
                'rewards/exit_layer_penalty_component_mean': 'Mean exit-layer penalty contribution',
                # Exit
                'exit/avg_layer': 'Mean prescribed exit layer index',
                'exit/min_layer': 'Min prescribed exit layer index (normalized 0..1)',
                'exit/max_layer': 'Max prescribed exit layer index (normalized 0..1)',
                'exit/std_layer': 'Std of prescribed exit layer index (normalized 0..1)',
                # Loss & training progress
                'loss/policy_avg': 'Policy loss (RLOO-weighted SFT)',
                'training/lr': 'Optimizer learning rate',
                'training/episode': 'Training step index',
                # Completions
                'completions/mean_length': 'Mean completion length (tokens)',
                'completions/min_length': 'Min completion length (tokens)',
                'completions/max_length': 'Max completion length (tokens)',
                'completions/clipped_ratio': 'Frac. completions without EOS',
                'completions/num_eos_tokens': 'Total EOS tokens in batch',
                # Samples table - deterministic “first-N” sampling  (Every sample_log_interval episodes, we log the first sample_max_rows completions)
                'samples/generations': 'W&B table with periodic sample generations and per-sample metrics',
                'samples/prompt_text': 'Original input prompt text for each sample',
                'samples/completion_text': 'Raw generated completion text for each sample',
                'samples/correct_answer': 'Correct answer for the prompt (extracted from dataset)',
                'samples/verify_reward': 'Verification reward for the sample (1.0 correct, <=0 penalized)',
                'samples/kl_estimate': 'Average per-token log-prob difference vs reference (student - ref)',
                'samples/avg_exit_layer': 'Normalized average exit layer used for the sample (0..1)',
                'samples/gen_len': 'Number of generated tokens (excluding prompt tokens)',
                'samples/contains_eos': 'Whether the sample generation contained an EOS token',
                'samples/selection_index': 'Row index within the K completions for the prompt',
                'samples/selection_policy': 'Policy used for choosing logged completions (first N)',
                'samples/selection_count': 'Number of completions logged in the samples table',
            }
        )
    )

    # Define metric step and categories for clean grouping in W&B UI
    wandb.define_metric('training/episode')
    for pattern in ['objective/*', 'rewards/*', 'exit/*', 'loss/*', 'completions/*', 'training/*', 'samples/*']:
        wandb.define_metric(pattern, step_metric='training/episode')

    # TODO: batching. For simplicity, treat batch_size = 1 here.
    train_dataset = dataset["train"]
    for i, example in enumerate(train_dataset):


        prompt = example["question"]
        correct_answer = example["answer"]

        # 1) Rollouts (student free-generate K)
        completions, exit_info = generate_k_completions(student, [prompt], k=RL_HPARAMS.k, 
                                                        tokenizer=tokenizer, config=config, device=device, 
                                                        system_prompt = RL_HPARAMS.system_prompt)  # TODO
        input_prompt_length = get_input_prompt_length(tokenizer, prompt, system_prompt = RL_HPARAMS.system_prompt)  # TODO: very hacky, do it in a cleaner way
        print(f"Input prompt length (in tokens): {input_prompt_length}")
        set_transformer_early_exit_mode(student, 'sft_student')

        # 2) Log-probs for KL and rewards (reference vs student)  # TODO: confirm scoring design
        
        ref_logprobs = compute_token_logprobs_reference(reference, 
                                                        completions['tokens'],
                                                        input_prompt_length)  # TODO
        
        prescribed_exit_layers = pad_sequence(exit_info['prescribed_exit_layers'], batch_first=True, padding_value=torch.inf)
        stu_logprobs, student_early_exit_probs = compute_token_logprobs_student(student, 
                                                      completions['tokens'], 
                                                      prescribed_exit_layers=prescribed_exit_layers,
                                                      input_prompt_length=input_prompt_length)  # TODO
        
        # import ipdb; ipdb.set_trace()
        # Runtime validation of rollout tensors (dtype/shape checks)
        _ = RolloutBatch(
            tokens=completions['tokens'],
            texts=completions['texts'],
            ref_logprobs=ref_logprobs,
            stu_logprobs=stu_logprobs,
            # prescribed_exit_layers=exit_info.get('prescribed_exit_layers', None),
            prescribed_exit_layers=prescribed_exit_layers,
            input_prompt_length=input_prompt_length,
            student_early_exit_probs=student_early_exit_probs
            #avg_exit_layer=exit_info.get('avg_exit_layer', None), #calced in rewards later
        )

        # 3) Reward components
        verify = compute_verification_rewards(completions['texts'], [correct_answer] * RL_HPARAMS.k)
        kl_tokens = compute_token_kl_from_logprobs(stu_logprobs, ref_logprobs)  # TODO
        avg_exit_layer = compute_avg_exit_layer(exit_info['prescribed_exit_layers'], student) #need to pass model to get total layers
        # import ipdb; ipdb.set_trace()
        # 4) Total reward per sequence (simple linear combination)
        reward = verify.to(device) - RL_HPARAMS.beta_kl * kl_tokens - RL_HPARAMS.lambda_exit * avg_exit_layer.to(device)  # TODO: tune weights, consider normalization

        # 5) Centering per prompt
        advantages = center_rewards_per_prompt(reward, batch_size=1, k=RL_HPARAMS.k)
        # Normalize advantages by their standard deviation to stabilize learning
        adv_std = advantages.std(unbiased=False) + 1e-8
        #adv_std = 1.
        normalized_advantages = advantages / adv_std
        
        # 6) Weighted SFT update
        sampled_early_exit_layer_idxs_early = map_layers_to_indices(prescribed_exit_layers, student.exitable_layer_idxs).to(device)
        student_sampled_exit_logprobs = (student_early_exit_probs + 1e-16).gather(
            index = sampled_early_exit_layer_idxs_early.unsqueeze(-1), dim = 2).log().squeeze(-1)
        
        # import ipdb; ipdb.set_trace()
        loss = weighted_sft_step(stu_logprobs, student_sampled_exit_logprobs, normalized_advantages, optimizer, RL_HPARAMS)  # TODO
        # 7) Logging (schema)
        with torch.no_grad():
            tokens_tensor = completions['tokens']  # [batch*K, seq_len]
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
            eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1

            seq_lens = (tokens_tensor != pad_id).sum(dim=1).float()
            contains_eos = (tokens_tensor == eos_id).any(dim=1) if eos_id != -1 else torch.zeros_like(seq_lens, dtype=torch.bool)
            clipped_ratio = 1.0 - contains_eos.float().mean().item()
            num_eos_tokens = int((tokens_tensor == eos_id).sum().item()) if eos_id != -1 else 0

            log_dict = {
                # Objective metrics
                'objective/rlhf_reward': reward.mean().item(),
                'objective/kl': kl_tokens.mean().item(),
                'objective/non_score_reward': (- RL_HPARAMS.beta_kl * kl_tokens - RL_HPARAMS.lambda_exit * avg_exit_layer.to(device)).mean().item(),
                
                # Exit metrics
                'exit/min_layer': avg_exit_layer.min().item(),
                'exit/max_layer': avg_exit_layer.max().item(),
                'exit/avg_layer': avg_exit_layer.mean().item(),
                'exit/std_layer': avg_exit_layer.std(unbiased=False).item(),
                
                # Reward components
                'rewards/verify_mean': verify.mean().item(),
                'rewards/kl_penalty_component_mean': (RL_HPARAMS.beta_kl * kl_tokens).mean().item(),
                'rewards/exit_layer_penalty_component_mean': (RL_HPARAMS.lambda_exit * avg_exit_layer).mean().item(),

                # Training advantage
                'training/lr': optimizer.param_groups[0]['lr'],
                'training/episode': i,
                'training/loss': float(loss.item() if hasattr(loss, 'item') else loss),
                # 'training/advantage_mean': advantages.mean().item(),
                'training/advantage_std': advantages.std(unbiased=False).item(),
                'training/total_neg_logprobs_mean': -(stu_logprobs.mean().item() + student_sampled_exit_logprobs.mean().item()),
                
                # Logprobs Mean
                'neg_logprobs/total': -(stu_logprobs.mean().item() + student_sampled_exit_logprobs.mean().item()),
                'neg_logprobs/prediction': -stu_logprobs.mean().item(),
                'neg_logprobs/exit': -student_sampled_exit_logprobs.mean().item(),

                # Completions
                'completions/mean_length': seq_lens.mean().item(),
                'completions/min_length': seq_lens.min().item(),
                'completions/max_length': seq_lens.max().item(),
                'completions/clipped_ratio': clipped_ratio,
                'completions/num_eos_tokens': num_eos_tokens,
            }

            # Periodic sample generations table
            if (i % RL_HPARAMS.sample_log_interval) == 0:
                num_rows = min(len(completions['texts']), RL_HPARAMS.sample_max_rows)
                columns = [
                    'episode',
                    'samples/prompt_text',
                    'samples/completion_text',
                    'samples/correct_answer',
                    'samples/verify_reward',
                    'samples/kl_estimate',
                    'samples/avg_exit_layer',
                    'samples/gen_len',
                    'samples/contains_eos',
                    'samples/selection_index',
                ]
                table = wandb.Table(columns=columns)
                for row_idx in range(num_rows):
                    full_len = int(seq_lens[row_idx].item())
                    gen_len = max(0, full_len - int(input_prompt_length))
                    table.add_data(
                        i,
                        prompt,
                        completions['texts'][row_idx],
                        extract_solution(correct_answer),
                        float(verify[row_idx].item()),
                        float(kl_tokens[row_idx].item()),
                        float(avg_exit_layer[row_idx].item()),
                        int(gen_len),
                        bool(contains_eos[row_idx].item()),
                        int(row_idx),
                    )
                log_dict['samples/generations'] = table
                # log_dict['samples/selection_policy'] = 'first'
                # log_dict['samples/selection_count'] = num_rows

            wandb.log(log_dict)

    wandb.finish()


main_rl_training()