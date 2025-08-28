"""


- Flow: K rollouts per prompt → compute rewards (verify - beta*KL - lambda*avg_exit_layer) → center per-prompt → weighted SFT.
"""

import torch
from torch.optim import Adam
import wandb
from datasets import load_dataset
from typing import Optional

from early_exit.util import get_model, load_model_from_wandb
from early_exit.util import generate_k_completions, center_rewards_per_prompt, weighted_sft_step, get_input_prompt_length
from early_exit.rl_types import RLHyperparams, RolloutBatch
from early_exit.rewards import compute_verification_rewards, compute_token_kl_from_logprobs, compute_token_logprobs_reference, compute_token_logprobs_student, compute_avg_exit_layer
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode
from shared_utils.load import get_tokenizer, configs_from_yaml
from torch.nn.utils.rnn import pad_sequence

device = "cuda"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
config_path = "config_deepseek.yaml"
sft_model_path = "models/early_exit_sft_trained"  # TODO: set path to SFT checkpoint

RL_HPARAMS = RLHyperparams()


# --- Models (schema) ---
tokenizer = get_tokenizer(model_name)
config = configs_from_yaml(config_path, tokenizer.eos_token_id)

student = get_model(model_name, config['model'], device)
student = replace_attention_layers(student, config['lora'], device)
# TODO: Change artifact path to sft trained gsm-8k model
student = load_model_from_wandb(student, model_path = "models/trained_model_v0", 
                              artifact_path = 'vkarthik095-university-of-amsterdam/early-exit/early-exit-model-fs5ofmzp:v0')

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
    # Minimal wandb init (extend later)
    run = wandb.init(
        project="early-exit-RL",
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
            }
        )
    )

    # Define metric step and categories for clean grouping in W&B UI
    wandb.define_metric('training/episode')
    for pattern in ['objective/*', 'rewards/*', 'exit/*', 'loss/*', 'completions/*', 'training/*']:
        wandb.define_metric(pattern, step_metric='training/episode')

    # TODO: batching. For simplicity, treat batch_size = 1 here.
    train_dataset = dataset["train"]
    for i, example in enumerate(train_dataset):


        prompt = example["question"]
        correct_answer = example["answer"]

        # 1) Rollouts (student free-generate K)
        completions, exit_info = generate_k_completions(student, [prompt], k=RL_HPARAMS.k, tokenizer=tokenizer, config=config, device=device)  # TODO
        input_prompt_length = get_input_prompt_length(tokenizer, prompt)  # TODO: very hacky, do it in a cleaner way
        print(f"Input prompt length (in tokens): {input_prompt_length}")
        set_transformer_early_exit_mode(student, 'sft_student')

        # 2) Log-probs for KL and rewards (reference vs student)  # TODO: confirm scoring design
        
        ref_logprobs = compute_token_logprobs_reference(reference, 
                                                        completions['tokens'],
                                                        input_prompt_length)  # TODO
        
        prescribed_exit_layers = pad_sequence(exit_info['prescribed_exit_layers'], batch_first=True, padding_value=torch.inf)
        stu_logprobs = compute_token_logprobs_student(student, 
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
            input_prompt_length=input_prompt_length
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
        advantages = advantages / adv_std

        # 6) Weighted SFT update
        
        loss = weighted_sft_step(stu_logprobs, advantages, optimizer, RL_HPARAMS)  # TODO
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
                'rewards/verify_mean': verify.mean().item(),
                'objective/kl': kl_tokens.mean().item(),
                'exit/avg_layer': avg_exit_layer.mean().item(),
                'objective/non_score_reward': (- RL_HPARAMS.beta_kl * kl_tokens - RL_HPARAMS.lambda_exit * avg_exit_layer.to(device)).mean().item(),
                # Reward components
                'rewards/verify_reward_component_mean': verify.mean().item(),
                'rewards/kl_penalty_component_mean': (RL_HPARAMS.beta_kl * kl_tokens).mean().item(),
                'rewards/exit_layer_penalty_component_mean': (RL_HPARAMS.lambda_exit * avg_exit_layer).mean().item(),

                # Loss / training progress
                'loss/policy_avg': float(loss.item() if hasattr(loss, 'item') else loss),
                'training/lr': optimizer.param_groups[0]['lr'],
                'training/episode': i,

                # Completions
                'completions/mean_length': seq_lens.mean().item(),
                'completions/min_length': seq_lens.min().item(),
                'completions/max_length': seq_lens.max().item(),
                'completions/clipped_ratio': clipped_ratio,
                'completions/num_eos_tokens': num_eos_tokens,
            }

            wandb.log(log_dict)

    wandb.finish()


main_rl_training()