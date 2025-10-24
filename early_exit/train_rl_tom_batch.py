"""


- Flow: K rollouts per prompt → compute rewards (verify - beta*KL - lambda*avg_exit_layer) → center per-prompt → weighted SFT.
"""

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import wandb
from datasets import load_dataset
from typing import Optional
import asyncio
import pandas as pd
from datetime import datetime

from early_exit.util import get_model, load_model_from_wandb, load_model, configs_from_json, save_model, CSVPromptDataset
from early_exit.rl_utils import apply_masking, create_attention_mask_from_tokens, generate_k_completions, center_rewards_per_prompt, map_layers_to_indices, weighted_sft_step, get_input_prompt_length, evaluate_coherence, compute_sample_labels, load_tom, compute_accuracy_by_difficulty, weighted_sft_loss
from early_exit.rl_types import RLHyperparams, RolloutBatch
from early_exit.rewards import compute_verification_rewards_text, compute_token_kl_from_logprobs, compute_token_logprobs_reference, compute_token_logprobs_student, compute_avg_exit_layer, extract_solution
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode
from shared_utils.load import get_tokenizer, configs_from_yaml
#from shared_utils.data import CSVPromptDataset
from torch.nn.utils.rnn import pad_sequence


device = "cuda"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
config_path = "config_deepseek.yaml"
sft_model_path = "models/early_exit_20251002_layers_5_big"  # TODO: set path to SFT checkpoint

RL_HPARAMS = RLHyperparams()
training_steps_per_rollout = 1

BATCH_SIZE = 4

save_freq = 250
save_dir = f"models/rl_{datetime.now().strftime('%Y%m%d')}"

# --- Models (schema) ---
tokenizer = get_tokenizer(model_name)
config = configs_from_yaml(config_path, tokenizer.eos_token_id)

student = get_model(model_name, config['model'], device)
student = replace_attention_layers(student, config['lora'], device)

student = load_model_from_wandb(student, model_path = "models/sft_model", artifact_path = 'vkarthik095-university-of-amsterdam/early-exit/my-model:v0')
#student = load_model(student, sft_model_path)

# Reference policy: base unmodified model without early exit
reference = get_model(model_name, config['model'], device)
reference.eval()

# Dataset
dataset_path = "results_and_data/early_exit_sft_dataset/test/tom_rl.csv"
prompt_config_path = "results_and_data/early_exit_sft_dataset/test/prompt_config_tom.json"

dataset = CSVPromptDataset(dataset_path, prompt_config_path)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn, shuffle=False)


def main_rl_training():
    """
    Schema: Generate → Reward → Center → Weighted SFT
    """
    # TODO: optimizer (e.g., Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=1e-5))
    # Check if there are better optimizers for this problem
    lora_params = []
    exit_decision_params = []
    for name, param in student.named_parameters():
        if param.requires_grad:
            if 'lora' in name:
                lora_params.append(param)
            elif 'early_exit_decision_weights' in name:
                exit_decision_params.append(param) 
            else: 
                raise ValueError(f"Unknown trainable parameter: {name}")
                
    optimizer = Adam([ {'params': lora_params, 'lr': 1e-4}, {'params': exit_decision_params, 'lr': 1e-5} ])
    #optimizer = Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=1e-5)
    # we use https://huggingface.co/docs/trl/rloo_trainer  as an inspiration for logging. 

    global_training_step = 0
    table_history = []

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
                #'samples/difficulty_category': 'Difficulty category: Easy, Medium, Hard',
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

    for batch_idx, prompt_batch in enumerate(dataloader):
        prompts = prompt_batch.full_user_prompt
        correct_answers = prompt_batch.correct_answers
        #difficulty_category = example['difficulty_category']
        B = len(prompts)
        optimizer.zero_grad(set_to_none=True)
        total_loss_val = 0.0
        accum_den = float(B * training_steps_per_rollout)

        #lists to accumulate batched values
        batch_rewards = []
        batch_kls = []
        batch_verify_scores = []
        batch_exit_layers = []
        batch_neglog_pred = []
        batch_neglog_exit = []
        batch_neglog_total = []
        batch_generated_lens = []
        batch_contains_eos_flags = []
        batch_num_eos = 0
        acc_answer_sum = 0.0
        acc_format_sum = 0.0
        acc_difficulty_sums = {}
        acc_count = 0
        sample_rows = []
        
        for i in range(B):
            prompt = prompts[i]
            correct_answer = correct_answers[i]

            # 1) Rollouts (student free-generate K)
            completions, exit_info = generate_k_completions(student, [prompt], k=RL_HPARAMS.k, 
                                                            tokenizer=tokenizer, config=config, device=device, 
                                                            system_prompt = dataset.system_prompt)  # TODO
            input_prompt_length = get_input_prompt_length(tokenizer, prompt, system_prompt = dataset.system_prompt)  # TODO: very hacky, do it in a cleaner way
            generated_attention_mask = create_attention_mask_from_tokens(completions['tokens'], tokenizer.pad_token_id)[:, input_prompt_length:]
            assert generated_attention_mask.sum(-1).tolist() == [len(item) for item in exit_info['prescribed_exit_layers']]
        
            print(f"Input prompt length (in tokens): {input_prompt_length}")
            set_transformer_early_exit_mode(student, 'sft_student')

            # 2) Log-probs for KL and rewards (reference vs student)  # TODO: confirm scoring design
            ref_logprobs = compute_token_logprobs_reference(reference, 
                                                            completions['tokens'],
                                                            input_prompt_length)
            prescribed_exit_layers = pad_sequence(exit_info['prescribed_exit_layers'], batch_first=True, padding_value=torch.inf)

            # 3) Reward components (computed once per rollout)
            verify, aux = compute_verification_rewards_text(completions['tokens'], completions['texts'], [correct_answer] * RL_HPARAMS.k, input_prompt_length, tokenizer)
            avg_exit_layer = compute_avg_exit_layer(exit_info['prescribed_exit_layers'], student) #need to pass model to get total layers

            # 3.1) Compute sample labels
            sample_labels = compute_sample_labels(verify, aux=aux)

            difficulty_categories = ['easy'] * RL_HPARAMS.k
            #compute accuracy metrics by difficulty (all samples have same difficulty as same prompt)
            difficulty_accuracies = compute_accuracy_by_difficulty(verify, difficulty_categories)

            # Pre-compute data that doesn't change between training steps
            with torch.no_grad():
                sampled_early_exit_layer_idxs_early = map_layers_to_indices(prescribed_exit_layers, student.exitable_layer_idxs).to(device)
            ref_logprobs = apply_masking(ref_logprobs, completions['tokens'], input_prompt_length, tokenizer.pad_token_id)

            for training_step in range(training_steps_per_rollout):
                # Compute current student logprobs (these change after each training step)
                stu_logprobs, student_early_exit_logprobs = compute_token_logprobs_student(student, 
                                                              completions['tokens'], 
                                                              prescribed_exit_layers=prescribed_exit_layers,
                                                              input_prompt_length=input_prompt_length)  # TODO
                
                stu_logprobs = apply_masking(stu_logprobs, completions['tokens'], input_prompt_length, tokenizer.pad_token_id)
                student_early_exit_logprobs = apply_masking(student_early_exit_logprobs, completions['tokens'], input_prompt_length, 
                                                         tokenizer.pad_token_id, mode = 'early_exit_probs')
    
                if training_step == 0:
                    first_step_lp_tokens = (stu_logprobs * generated_attention_mask).sum(dim=1).detach()

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
                
                student_sampled_exit_logprobs = student_early_exit_logprobs.gather(
                    index = sampled_early_exit_layer_idxs_early.unsqueeze(-1), dim = 2).squeeze(-1)
    
                # Compute current KL (this changes as student changes)
                kl_tokens = compute_token_kl_from_logprobs(stu_logprobs, ref_logprobs, generated_attention_mask)
    
                # 4) Total reward per sequence (simple linear combination)
                reward = verify.to(device) - RL_HPARAMS.beta_kl * kl_tokens - RL_HPARAMS.lambda_exit * avg_exit_layer.to(device)  # TODO: tune weights, consider normalization
    
                # 5) Centering per prompt
                advantages = center_rewards_per_prompt(reward, batch_size=1, k=RL_HPARAMS.k)
                # Normalize advantages by their standard deviation to stabilize learning
                adv_std = advantages.std(unbiased=False) + 1e-8
                normalized_advantages = advantages / adv_std
                
                #Probability ratio: new / old (1.0 for step 0)
                new_lp_tokens = (stu_logprobs * generated_attention_mask).sum(dim=1)
                ratio = (new_lp_tokens - first_step_lp_tokens).exp().detach()
                #print(f"ratio: {ratio}, new_lp_tokens: {new_lp_tokens}, first_step_lp_tokens {first_step_lp_tokens}")
                
                final_advantages = normalized_advantages * ratio
            
                # 6) Weighted SFT update
                #loss = weighted_sft_step(stu_logprobs, student_sampled_exit_logprobs, normalized_advantages, generated_attention_mask, optimizer, RL_HPARAMS)
                loss = weighted_sft_loss(stu_logprobs, student_sampled_exit_logprobs, final_advantages, generated_attention_mask, RL_HPARAMS)
                (loss / accum_den).backward()
                total_loss_val += float(loss.detach())

                #batched accumulations
                with torch.no_grad():
                    batch_rewards.append(reward.detach().cpu())                     # [K]
                    batch_kls.append(kl_tokens.detach().cpu())                      # [K]
                    batch_verify_scores.append(verify.detach().cpu())
                    batch_exit_layers.append(avg_exit_layer.detach().cpu())         # [K]
                    nl_pred = -stu_logprobs.mean().detach().cpu()
                    nl_exit = -student_sampled_exit_logprobs.mean().detach().cpu()
                    batch_neglog_pred.append(nl_pred)
                    batch_neglog_exit.append(nl_exit)
                    batch_neglog_total.append((nl_pred + nl_exit))
                    tokens_tensor = completions['tokens']  # shape [K, T]
                    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
                    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1
                    seq_lens = (tokens_tensor != pad_id).sum(dim=1).float()         # [K]
                    gen_lens = (seq_lens - input_prompt_length).clamp_min(0)
                    batch_generated_lens.append(gen_lens.detach().cpu())
                    if eos_id != -1:
                        contains_eos = (tokens_tensor == eos_id).any(dim=1)          # [K] bool
                        batch_contains_eos_flags.append(contains_eos.cpu())
                        batch_num_eos += int((tokens_tensor == eos_id).sum().item())
                    acc_answer_sum += sample_labels['stats']['answer_accuracy']
                    acc_format_sum += sample_labels['stats']['format_accuracy']
                    for k, v in compute_accuracy_by_difficulty(verify, ['easy'] * RL_HPARAMS.k).items():
                        acc_difficulty_sums[k] = acc_difficulty_sums.get(k, 0.0) + float(v)
                    acc_count += 1

                    for k_idx in range(RL_HPARAMS.k):
                        sample_rows.append({
                            "episode": global_training_step + 1,  # next step
                            "prompt_text": prompt,
                            "completion_text": completions['texts'][k_idx],
                            "correct_answer": correct_answer,
                            "verify_reward": float(verify[k_idx].item()),
                            "kl_estimate": float(kl_tokens[k_idx].item()),
                            "avg_exit_layer": float(avg_exit_layer[k_idx].item()),
                            "gen_len": int(gen_lens[k_idx].item()),
                            "contains_eos": bool(contains_eos[k_idx].item()),
                            "selection_index": int(k_idx),
                            # if you still want labels, pass the right slice here:
                            "correctness_label": sample_labels['labels'][k_idx]['correctness'],
                            "format_label": sample_labels['labels'][k_idx]['format_quality'],
                        })

        optimizer.step()


        # 7) Logging
        global_training_step += 1
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

            reward = torch.cat(batch_rewards)                    # rewards
            verify = torch.cat(batch_verify_scores)
            kl_tokens = torch.cat(batch_kls)
            avg_exit_layer = torch.cat(batch_exit_layers)
            generated_lens = torch.cat(batch_generated_lens)  # [B*K]
    
            if batch_contains_eos_flags:
                contains_all = torch.cat(batch_contains_eos_flags).float()  # CPU
                clipped_ratio = 1.0 - contains_all.mean().item()
            else:
                clipped_ratio = 0.0

            log_dict = {
                # Objective metrics
                'objective/rlhf_reward': reward.mean().item(),
                'objective/kl': kl_tokens.mean().item(),
                'objective/non_score_reward': (- RL_HPARAMS.beta_kl * kl_tokens - RL_HPARAMS.lambda_exit * avg_exit_layer).mean().item(),
                
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
                'training/episode': global_training_step,
                'training/loss': total_loss_val / float(B),  #float(loss.item() if hasattr(loss, 'item') else loss),
                # 'training/advantage_mean': advantages.mean().item(),
                'training/advantage_std': advantages.std(unbiased=False).item(),
                'training/total_neg_logprobs_mean': torch.stack(batch_neglog_total).mean().item(),

                # Logprobs Mean
                'neg_logprobs/total': torch.stack(batch_neglog_total).mean().item(),
                'neg_logprobs/prediction': torch.stack(batch_neglog_pred).mean().item(),
                'neg_logprobs/exit': torch.stack(batch_neglog_exit).mean().item(),
                
                # Completions
                'completions/mean_length': generated_lens.mean().item(),
                'completions/min_length': generated_lens.min().item(),
                'completions/max_length': generated_lens.max().item(),
                'completions/clipped_ratio': clipped_ratio,
                'completions/num_eos_tokens': batch_num_eos,

                # Accuracy (averaged over prompts in the batch)
                'accuracy/answer_accuracy': acc_answer_sum / max(1, acc_count),
                'accuracy/format_accuracy': acc_format_sum / max(1, acc_count),
            }

            for key, value in acc_difficulty_sums.items():
                log_dict[f'accuracy/{key}'] = value / max(1, acc_count) #add difficulty accuracies to log dict

            # Periodic sample generations table
            if (global_training_step % RL_HPARAMS.sample_log_interval) == 0 and len(sample_rows) > 0:
                rows_to_log = sample_rows[:RL_HPARAMS.sample_max_rows]
                #num_rows = min(len(completions['texts']), RL_HPARAMS.sample_max_rows)

                async def eval_coherence_batch():
                    tasks = [evaluate_coherence(r["prompt_text"], r["completion_text"]) for r in rows_to_log]
                    return await asyncio.gather(*tasks)
                
                coherence_batch_results = asyncio.run(eval_coherence_batch())
                
                n = max(1, len(coherence_batch_results))
                avg_coherence = sum(r['coherence'] for r in coherence_batch_results) / n
                avg_completeness = sum(r['completeness'] for r in coherence_batch_results) / n
                avg_clarity = sum(r['clarity'] for r in coherence_batch_results) / n
                avg_no_repetition = sum(r['no_repetition'] for r in coherence_batch_results) / n
                avg_overall = sum(r['average'] for r in coherence_batch_results) / n
                
                log_dict.update({
                    'coherence/batch_coherence': avg_coherence,
                    'coherence/batch_completeness': avg_completeness,
                    'coherence/batch_clarity': avg_clarity,
                    'coherence/batch_no_repetition': avg_no_repetition,
                    'coherence/batch_average': avg_overall,
                })
                
                for r, c in zip(rows_to_log, coherence_batch_results):
                    r["coherence_coherence"] = int(c["coherence"])
                    r["coherence_completeness"] = int(c["completeness"])
                    r["coherence_clarity"] = int(c["clarity"])
                    r["coherence_no_repetition"] = int(c["no_repetition"])
                    r["coherence_average"] = float(c["average"])
                    r["coherence_explanation"] = c["explanation"]
            
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
                    'samples/correctness_label',
                    'samples/format_label',
                    'samples/coherence_coherence',
                    'samples/coherence_completeness',
                    'samples/coherence_clarity',
                    'samples/coherence_no_repetition',
                    'samples/coherence_average',
                    'samples/coherence_explanation',
                ]
            
                table = wandb.Table(columns=columns)
                for r in rows_to_log:
                    table.add_data(
                        r["episode"],
                        r["prompt_text"],
                        r["completion_text"],
                        r["correct_answer"],
                        r["verify_reward"],
                        r["kl_estimate"],
                        r["avg_exit_layer"],
                        r["gen_len"],
                        r["contains_eos"],
                        r["selection_index"],
                        r["correctness_label"],
                        r["format_label"],
                        r.get("coherence_coherence"),
                        r.get("coherence_completeness"),
                        r.get("coherence_clarity"),
                        r.get("coherence_no_repetition"),
                        r.get("coherence_average"),
                        r.get("coherence_explanation"),
                    )
                log_dict['samples/generations'] = table

            wandb.log(log_dict)

        if global_training_step % save_freq == 0:
            checkpoint_path = f"{save_dir}/step_{global_training_step}"
            save_model(student, checkpoint_path, upload_to_wandb=False)
            print(f"Checkpoint saved to {checkpoint_path}")

    wandb.finish()


main_rl_training()