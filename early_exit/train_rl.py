"""


- Flow: K rollouts per prompt → compute rewards (verify - beta*KL - lambda*avg_exit_layer) → center per-prompt → weighted SFT.
"""

import torch
from torch.optim import Adam
import wandb
from datasets import load_dataset
from typing import Optional
import asyncio
import pandas as pd

import sys
sys.path.append("../")

from early_exit.util import get_model, load_model_from_wandb, load_model, configs_from_json
from early_exit.rl_utils import apply_masking, create_attention_mask_from_tokens, generate_k_completions, center_rewards_per_prompt, map_layers_to_indices, weighted_sft_step, get_input_prompt_length, evaluate_coherence, compute_sample_labels, load_gsm8k_with_difficulty, compute_accuracy_by_difficulty
from early_exit.rl_types import RLHyperparams, RolloutBatch
from early_exit.rewards import compute_verification_rewards, compute_token_kl_from_logprobs, compute_token_logprobs_reference, compute_token_logprobs_student, compute_avg_exit_layer, extract_solution
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode
from shared_utils.load import get_tokenizer, configs_from_yaml
from torch.nn.utils.rnn import pad_sequence

import torch

device = "cuda"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
config_path = "../config_deepseek.yaml"
sft_model_path = "models/early_exit_20250906_layers_5_big"  # TODO: set path to SFT checkpoint

RL_HPARAMS = RLHyperparams()


# --- Models (schema) ---
tokenizer = get_tokenizer(model_name)
config = configs_from_yaml(config_path, tokenizer.eos_token_id)

student = get_model(model_name, config['model'], device)
student = replace_attention_layers(student, config['lora'], device)
# TODO: Change artifact path to sft trained gsm-8k model
student = load_model_from_wandb(student, model_path = "models/trained_model_v0", artifact_path = 'vkarthik095-university-of-amsterdam/early-exit/early-exit-model-fs5ofmzp:v0')
#student = load_model(student, sft_model_path)

# Reference policy: base unmodified model without early exit
reference = get_model(model_name, config['model'], device)
reference.eval()
# TODO: ensure no early-exit logic is active for reference model

# Dataset
#dataset = load_dataset("gsm8k", "main")  # TODO: verify/parse answer format
dataset = load_gsm8k_with_difficulty()


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
                'objective/rlhf_reward': 'Mean total reward per step: verification reward minus beta_kl×token-level KL estimate and lambda_exit×normalized average exit layer.',
                'objective/kl': 'Mean over generated tokens of (log p_student − log p_reference). This is a token-level log-probability gap, not the full softmax KL.',
                'objective/non_score_reward': 'Mean of the penalty-only terms (− beta_kl×KL − lambda_exit×avg_exit). Higher magnitude indicates stronger regularization pressure.',
                'rewards/verify_mean': "Mean verification reward from the final '#### <answer>' extraction. Exact match = 1.0, flexible numeric match = 0.5, wrong = 0.0, no answer/format errors = −1.0, minus small format penalties.",
                'rewards/kl_penalty_component_mean': 'Mean of the KL penalty contribution beta_kl×(token-level KL estimate).',
                'rewards/exit_layer_penalty_component_mean': 'Mean of the exit-layer penalty contribution lambda_exit×normalized average exit layer.',
                # Exit
                'exit/avg_layer': 'Mean normalized exit layer index used during generation (0 = first layer, 1 = final layer).',
                'exit/min_layer': 'Minimum normalized exit layer index within the batch (0..1).',
                'exit/max_layer': 'Maximum normalized exit layer index within the batch (0..1).',
                'exit/std_layer': 'Standard deviation of normalized exit layer indices within the batch (0..1).',
                # Loss & training progress
                'loss/policy_avg': 'RLOO-weighted SFT loss: − mean(adv.detach() × (mean token log-likelihood + mean early-exit log-prob)).',
                'training/lr': 'Current optimizer learning rate.',
                'training/episode': 'Training step index used as the x-axis step for metrics.',
                'training/loss': 'Scalar training loss returned by the weighted SFT step for this episode.',
                'training/advantage_std': 'Standard deviation of per-sample advantages before normalization (unbiased=False).',
                'training/total_neg_logprobs_mean': 'Mean negative log-probability across token prediction and exit selection; lower is better.',
                # Log-prob breakdown
                'neg_logprobs/total': 'Mean negative log-prob across token prediction and early-exit selection: −(mean token log-likelihood + mean exit log-prob).',
                'neg_logprobs/prediction': 'Mean negative token log-likelihood over generated tokens (− mean log p_student).',
                'neg_logprobs/exit': 'Mean negative early-exit log-prob for the sampled exit layer positions.',
                # Completions
                'completions/mean_length': 'Mean sequence length in tokens (including prompt; excludes padding).',
                'completions/min_length': 'Minimum sequence length in tokens (including prompt; excludes padding).',
                'completions/max_length': 'Maximum sequence length in tokens (including prompt; excludes padding).',
                'completions/clipped_ratio': 'Fraction of sequences without an EOS token (likely due to max-length clipping).',
                'completions/num_eos_tokens': 'Total number of EOS tokens observed across the batch.',
                # Accuracy metrics (overall and by difficulty)
                #'accuracy/correct_rate': 'Fraction of samples with perfect correctness (reward = 1.0).',
                #'accuracy/partial_rate': 'Fraction of samples with partial correctness (reward = 0.5).',
                #'accuracy/incorrect_rate': 'Fraction of samples with incorrect answers (reward = 0.0).',
                #'accuracy/format_error_rate': 'Fraction of samples with format errors (reward = -1.0).',
                #'accuracy/good_format_rate': 'Fraction of samples with proper "####" format.',
                'accuracy/answer_accuracy': 'Overall fraction of samples with correct and properly formatted answers.',
                'accuracy/format_accuracy': 'Overall fraction of samples with proper format.',
                'accuracy/easy_answer_accuracy': 'Answer accuracy for Easy difficulty problems.',
                'accuracy/easy_format_accuracy': 'Format accuracy for Easy difficulty problems.',
                'accuracy/medium_answer_accuracy': 'Answer accuracy for Medium difficulty problems.',
                'accuracy/medium_format_accuracy': 'Format accuracy for Medium difficulty problems.',
                'accuracy/hard_answer_accuracy': 'Answer accuracy for Hard difficulty problems.',
                'accuracy/hard_format_accuracy': 'Format accuracy for Hard difficulty problems.',
                # Samples table - deterministic “first-N” sampling  (Every sample_log_interval episodes, we log the first sample_max_rows completions)
                'samples/generations': 'W&B table with periodic sample generations and per-sample metrics for quick qualitative inspection.',
                'samples/prompt_text': 'Original input prompt text for each sample.',
                'samples/completion_text': 'Raw generated completion text for each sample.',
                'samples/correct_answer': 'Correct answer for the prompt (parsed from the dataset).',
                'samples/difficulty_category': 'Difficulty category: Easy, Medium, Hard',
                'samples/verify_reward': 'Verification reward for the sample using the same rules as rewards/verify_mean.',
                'samples/kl_estimate': 'Token-level log-probability gap mean (student − reference) for the sample.',
                'samples/avg_exit_layer': 'Normalized average exit layer used for the sample (0..1).',
                'samples/gen_len': 'Number of generated tokens excluding the prompt tokens.',
                'samples/contains_eos': 'Whether the sample generation contained an EOS token.',
                'samples/selection_index': 'Row index within the K completions for the prompt.',
                # Sample table labels
                'samples/correctness_label': 'Correctness category: correct (1.0), partial (>=0.5), incorrect (0.0), format_error (>=-1.0).',
                'samples/format_label': 'Format quality: good_format (contains 1 #### with integer after and nothing else)',
            }
        )
    )

    # Log a one-time table of metric descriptions for convenient reference in W&B
    metric_descs = run.config.get('metric_descriptions', {}) if run is not None else {}
    if isinstance(metric_descs, dict) and len(metric_descs) > 0:
        desc_table = wandb.Table(columns=['metric', 'description'])
        for key, desc in metric_descs.items():
            desc_table.add_data(key, desc)
        wandb.log({'meta/metric_descriptions': desc_table})

    # Define metric step and categories for clean grouping in W&B UI
    wandb.define_metric('training/episode')
    for pattern in ['objective/*', 'rewards/*', 'exit/*', 'loss/*', 'completions/*', 'training/*', 'samples/*', 'accuracy/*']:
        wandb.define_metric(pattern, step_metric='training/episode')

    # TODO: batching. For simplicity, treat batch_size = 1 here.
    train_dataset = dataset["train"]
    table_history = []

    for i, example in enumerate(train_dataset):


        prompt = example["question"]
        correct_answer = example["answer"]

        # difficulty_info = difficulty_lookup.get(prompt, {
        #     'solved_percentage': None,
        #     'difficulty_category': 'Unknown'
        # })
        # difficulty_category = difficulty_info['difficulty_category']
        difficulty_category = example['difficulty_category']

        # 1) Rollouts (student free-generate K)
        completions, exit_info = generate_k_completions(student, [prompt], k=RL_HPARAMS.k, 
                                                        tokenizer=tokenizer, config=config, device=device, 
                                                        system_prompt = RL_HPARAMS.system_prompt)  # TODO
        input_prompt_length = get_input_prompt_length(tokenizer, prompt, system_prompt = RL_HPARAMS.system_prompt)  # TODO: very hacky, do it in a cleaner way
        generated_attention_mask = create_attention_mask_from_tokens(completions['tokens'], tokenizer.pad_token_id)[:, input_prompt_length:]
        assert generated_attention_mask.sum(-1).tolist() == [len(item) for item in exit_info['prescribed_exit_layers']]
        
        print(f"Input prompt length (in tokens): {input_prompt_length}")
        set_transformer_early_exit_mode(student, 'sft_student')

        # 2) Log-probs for KL and rewards (reference vs student)  # TODO: confirm scoring design
        ref_logprobs = compute_token_logprobs_reference(reference, 
                                                        completions['tokens'],
                                                        input_prompt_length)  # TODO
        prescribed_exit_layers = pad_sequence(exit_info['prescribed_exit_layers'], batch_first=True, padding_value=torch.inf)
        stu_logprobs, student_early_exit_logprobs = compute_token_logprobs_student(student, 
                                                      completions['tokens'], 
                                                      prescribed_exit_layers=prescribed_exit_layers,
                                                      input_prompt_length=input_prompt_length)  # TODO
        
        # import ipdb; ipdb.set_trace()
        stu_logprobs = apply_masking(stu_logprobs, completions['tokens'], input_prompt_length, tokenizer.pad_token_id)
        student_early_exit_logprobs = apply_masking(student_early_exit_logprobs, completions['tokens'], input_prompt_length, 
                                                 tokenizer.pad_token_id, mode = 'early_exit_probs')
        ref_logprobs = apply_masking(ref_logprobs, completions['tokens'], input_prompt_length, tokenizer.pad_token_id)
        # Runtime validation of rollout tensors (dtype/shape checks)
        _ = RolloutBatch(
            tokens=completions['tokens'],
            texts=completions['texts'],
            ref_logprobs=ref_logprobs,
            stu_logprobs=stu_logprobs,
            # prescribed_exit_layers=exit_info.get('prescribed_exit_layers', None),
            prescribed_exit_layers=prescribed_exit_layers,
            input_prompt_length=input_prompt_length,
            student_early_exit_logprobs=student_early_exit_logprobs
            #avg_exit_layer=exit_info.get('avg_exit_layer', None), #calced in rewards later
        )

        # 3) Reward components
        verify = compute_verification_rewards(completions['tokens'], completions['texts'], [correct_answer] * RL_HPARAMS.k, input_prompt_length, tokenizer)
        kl_tokens = compute_token_kl_from_logprobs(stu_logprobs, ref_logprobs, generated_attention_mask)
        avg_exit_layer = compute_avg_exit_layer(exit_info['prescribed_exit_layers'], student) #need to pass model to get total layers

        # 3.1) Compute sample labels (similar to format_accuracy/answer_accuracy in reference)
        sample_labels = compute_sample_labels(verify)

        difficulty_categories = [difficulty_category] * RL_HPARAMS.k
        #compute accuracy metrics by difficulty (all samples have same difficulty as same prompt)
        difficulty_accuracies = compute_accuracy_by_difficulty(verify, difficulty_categories)

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
        with torch.no_grad():
            sampled_early_exit_layer_idxs_early = map_layers_to_indices(prescribed_exit_layers, student.exitable_layer_idxs).to(device)
        student_sampled_exit_logprobs = student_early_exit_logprobs.gather(
            index = sampled_early_exit_layer_idxs_early.unsqueeze(-1), dim = 2).squeeze(-1)
        
        # import ipdb; ipdb.set_trace()
        loss = weighted_sft_step(stu_logprobs, student_sampled_exit_logprobs, normalized_advantages, generated_attention_mask, optimizer, RL_HPARAMS)  # TODO
        # 7) Logging (schema)
        torch.cuda.empty_cache()
        with torch.no_grad():
            tokens_tensor = completions['tokens']  # [batch*K, seq_len]
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
            eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1

            seq_lens = (tokens_tensor != pad_id).sum(dim=1).float()
            generated_lens = seq_lens - input_prompt_length
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
                'completions/mean_length': generated_lens.mean().item(),
                'completions/min_length': generated_lens.min().item(),
                'completions/max_length': generated_lens.max().item(),
                'completions/clipped_ratio': clipped_ratio,
                'completions/num_eos_tokens': num_eos_tokens,

                # Accuracy metrics (similar to format_accuracy/answer_accuracy in reference)
                #'accuracy/correct_rate': sample_labels['stats']['correct_rate'],
                #'accuracy/partial_rate': sample_labels['stats']['partial_rate'],
                #'accuracy/incorrect_rate': sample_labels['stats']['incorrect_rate'],
                #'accuracy/format_error_rate': sample_labels['stats']['format_error_rate'],
                #'accuracy/good_format_rate': sample_labels['stats']['good_format_rate'],
                'accuracy/answer_accuracy': sample_labels['stats']['answer_accuracy'],
                'accuracy/format_accuracy': sample_labels['stats']['format_accuracy'],
            }

            for key, value in difficulty_accuracies.items():
                log_dict[f'accuracy/{key}'] = value #add difficulty accuracies to log dict

            # Periodic sample generations table
            if (i % RL_HPARAMS.sample_log_interval) == 0:
                num_rows = min(len(completions['texts']), RL_HPARAMS.sample_max_rows)

                async def evaluate_batch_coherence():
                    tasks = []
                    for row_idx in range(num_rows):
                        task = evaluate_coherence(prompt, completions['texts'][row_idx])
                        tasks.append(task)
                    
                    results = await asyncio.gather(*tasks)
                    return results
                
                coherence_batch_results = asyncio.run(evaluate_batch_coherence())

                avg_coherence = sum(r['coherence'] for r in coherence_batch_results) / len(coherence_batch_results)
                avg_completeness = sum(r['completeness'] for r in coherence_batch_results) / len(coherence_batch_results)
                avg_clarity = sum(r['clarity'] for r in coherence_batch_results) / len(coherence_batch_results)
                avg_no_repetition = sum(r['no_repetition'] for r in coherence_batch_results) / len(coherence_batch_results)
                avg_overall = sum(r['average'] for r in coherence_batch_results) / len(coherence_batch_results)
    
                log_dict.update({
                    'coherence/batch_coherence': avg_coherence,
                    'coherence/batch_completeness': avg_completeness,
                    'coherence/batch_clarity': avg_clarity,
                    'coherence/batch_no_repetition': avg_no_repetition,
                    'coherence/batch_average': avg_overall,
                })

                columns = [
                    'episode',
                    'samples/prompt_text',
                    'samples/completion_text',
                    'samples/correct_answer',
                    'samples/difficulty_category',
                    'samples/verify_reward',
                    'samples/kl_estimate',
                    'samples/avg_exit_layer',
                    'samples/gen_len',
                    'samples/contains_eos',
                    'samples/selection_index',
                    'samples/correctness_label',
                    'samples/format_label',
                    'samples/coherence_coherence',
                    'samples/coherence_completeness',
                    'samples/coherence_clarity',
                    'samples/coherence_no_repetition',
                    'samples/coherence_average',
                    'samples/coherence_explanation',
                ]
                for row_idx in range(num_rows):
                    full_len = int(seq_lens[row_idx].item())
                    gen_len = max(0, full_len - int(input_prompt_length))

                    coherence_result = coherence_batch_results[row_idx]

                    table_history.append([
                        i,
                        prompt,
                        completions['texts'][row_idx],
                        extract_solution(correct_answer),
                        difficulty_category,
                        float(verify[row_idx].item()),
                        float(kl_tokens[row_idx].item()),
                        float(avg_exit_layer[row_idx].item()),
                        int(gen_len),
                        bool(contains_eos[row_idx].item()),
                        int(row_idx),
                        sample_labels['labels'][row_idx]['correctness'],
                        sample_labels['labels'][row_idx]['format_quality'],
                        int(coherence_result['coherence']),
                        int(coherence_result['completeness']),
                        int(coherence_result['clarity']),
                        int(coherence_result['no_repetition']),
                        float(coherence_result['average']),
                        coherence_result['explanation']
                    ])

                table = wandb.Table(columns=columns)
                for row in table_history:
                    table.add_data(*row)
                log_dict['samples/generations'] = table
                # log_dict['samples/selection_policy'] = 'first'
                # log_dict['samples/selection_count'] = num_rows

            wandb.log(log_dict)

    wandb.finish()


main_rl_training()