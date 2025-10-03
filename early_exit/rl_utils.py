import torch
from torch.optim import Adam

from datasets import load_dataset, Dataset
import pandas as pd

from early_exit.patching import set_transformer_early_exit_mode
from shared_utils.generate import generate_text
from typing import List

from early_exit.rl_types import *

from shared_utils.generate import generate_text

import asyncio
from inspect_ai.model import get_model as get_inspect_model
from huggingface_hub import hf_hub_download, upload_file

# ---------------- RL helper functions moved from train_rl.py ----------------
def generate_k_completions(model, prompt, k: int, tokenizer, config, device, system_prompt):
    """
    Free-generate K completions per prompt with early exits enabled.

    Expected outputs (used later in the pipeline):
    - completions:
        - tokens: LongTensor of shape [batch*K, seq_len]; dtype=torch.long
        - texts: list[str] of length batch*K
    - exit_info:
        - prescribed_exit_layers: Optional[LongTensor] of shape [batch*K, seq_len] for re-scoring

    Typical ranges:
    - seq_len: 16–512 depending on generation configuration
    """
    set_transformer_early_exit_mode(model, 'free_generate')

    all_tokens = []
    all_texts = []
    all_prescribed_exit_layers = []
    
    for p in prompt:
        for _ in range(k):
            with torch.no_grad():
                decoded_response, model_outputs = generate_text(
                    model=model,
                    prompt=p,
                    system_prompt=system_prompt,
                    prefiller='',
                    tokenizer=tokenizer,
                    generation_config=config['generation'],
                    device=device
                )

            sequences, exit_layer_idxs = model_outputs
            tokens = sequences[0]
            prescribed_exit_layers = exit_layer_idxs[0]
            
            all_tokens.append(tokens[:-1])
            all_texts.append(decoded_response)
            all_prescribed_exit_layers.append(prescribed_exit_layers[1:])
    
    max_seq_len = max(len(tokens) for tokens in all_tokens)
    padded_tokens = []
    final_prescribed_layers = []  # no padding since will mess up avg exit layer

    for i, (tokens, exit_layers) in enumerate(zip(all_tokens, all_prescribed_exit_layers)):
        pad_length = max_seq_len - len(tokens)
        if pad_length > 0:
            padded_token = torch.cat([tokens, torch.full((pad_length,), tokenizer.pad_token_id, dtype=tokens.dtype, device=tokens.device)])
        else:
            padded_token = tokens

        padded_tokens.append(padded_token)
        final_prescribed_layers.append(exit_layers)

    completions_tokens = torch.stack(padded_tokens, dim=0)

    completions = {
        'tokens': completions_tokens,
        'texts': all_texts
    }

    exit_info = {
        'prescribed_exit_layers': final_prescribed_layers
    }
    
    return completions, exit_info


def center_rewards_per_prompt(rewards, batch_size: int, k: int):
    """
    Center rewards across the K completions for each prompt (simple baseline).
    Returns: advantages FloatTensor [batch*K].
    """
    rewards = rewards.view(batch_size, k)
    adv = rewards - rewards.mean(dim=1, keepdim=True)
    return adv.reshape(-1)


def compute_sequence_mean_loglik_student(student_log_likelihoods, student_early_exit_logprobs, attention_mask):
    """
    TODO: Add the exit log probs!
    Returns: FloatTensor [batch*K].
    """
    assert student_log_likelihoods.dim() == 2, "student_log_likelihoods must be [batch*K, seq_len]"
    assert student_early_exit_logprobs.dim() == 2, f"student_early_exit_logprobs must be [batch*K, seq_len], it was instead {student_early_exit_logprobs.shape}"
    assert student_early_exit_logprobs.shape == student_log_likelihoods.shape, \
    f"student_early_exit_logprobs (shape = {student_early_exit_logprobs.shape})  must have the same shape as \
    student_log_likelihoods = {student_log_likelihoods.shape}"
    assert student_log_likelihoods.shape == student_early_exit_logprobs.shape, "Student log likelihoods and logprobs should have same shapes"
    assert student_log_likelihoods.shape == attention_mask.shape, "Student log likelihoods and attention mask should have same shapes"
    return (student_log_likelihoods.sum(-1) + student_early_exit_logprobs.sum(-1))/attention_mask.sum(-1)


def weighted_rloo_loss(advantages, log_likelihoods, RL_HPARAMS):
    """Computes loss on one weighted RLOO step, based on unlabelled 
    equation after equation 8 in https://arxiv.org/pdf/2402.14740v2.pdf. 
    It is important to detch advantages so that no gradients flow through them.
    loss = -mean(adv.detach() * seq_loglik_student)
    Args:
        advantages (FloatTensor): Shape (batch_size, k) containing the advantage values.
        log_likelihoods (FloatTensor): Shape (batch_size, k) containing the sum of log_likelihood of next tokens
    """
    assert advantages.shape == log_likelihoods.shape, "adv and log_likelihoods must have the same shape"
    num_exit_samples = RL_HPARAMS.k
    batch_size = RL_HPARAMS.batch_size
    advantages = advantages.view(batch_size, num_exit_samples)
    log_likelihoods = log_likelihoods.view(batch_size, num_exit_samples)
    return -(advantages.detach() * log_likelihoods).mean(-1)


def weighted_sft_step(student_log_likelihoods, student_early_exit_logprobs, advantages, attention_mask, optimizer, RL_HPARAMS):
    """
    Returns: scalar loss (FloatTensor).
    """
    optimizer.zero_grad()
    sequence_mean_log_likelihoods = compute_sequence_mean_loglik_student(student_log_likelihoods, 
                                                                         student_early_exit_logprobs,
                                                                         attention_mask) # ignore first token's exit prob
    loss = weighted_rloo_loss(advantages, sequence_mean_log_likelihoods, RL_HPARAMS)
    loss.backward()
    optimizer.step()
    return loss.detach()


def get_input_prompt_length(tokenizer, prompt, system_prompt):
    from shared_utils.generate import format_conversation, transform_conversations, full_tokenize
    
    pre_transformed_conversation = format_conversation(user_prompts = [prompt], system_prompt=system_prompt)
    tokens = tokenizer.apply_chat_template(pre_transformed_conversation, 
                                           tokenize=True,
                                           add_generation_prompt=True, # adds the <｜Assistant｜> token at the end,
                                           return_tensors="pt"
                                        )
    
    return tokens.shape[1]  # seq_len dimension

def map_layers_to_indices(layer_tensor, exitable_layer_idxs):
    """Map layer indices to their positions in exitable_layer_idxs array."""
    # Handle inf separately
    inf_mask = torch.isinf(layer_tensor)
    result = torch.zeros_like(layer_tensor, dtype=torch.long)
    
    # Map finite values
    if (~inf_mask).any():
        finite_layers = layer_tensor[~inf_mask]
        for i, target_layer in enumerate(exitable_layer_idxs[:-1]):  # exclude inf
            result[~inf_mask & (layer_tensor == target_layer)] = i
    
    # Map inf values to last index
    result[inf_mask] = len(exitable_layer_idxs) - 1
    
    return result

def create_attention_mask_from_tokens(tokens, pad_token_id: int):
    """
    Create an attention mask from token IDs, preserving the first EOS token
    that marks the actual end of sequence.
    
    Args:
        tokens: Token tensor of shape [batch_size, seq_len]
        pad_token_id: ID of the padding token (often same as EOS)
    
    Returns:
        Attention mask of shape [batch_size, seq_len] where 1 = real token, 0 = padding
    """
    assert tokens.dim() == 2, "tokens must be of shape [batch_size, seq_len]"
    batch_size, seq_len = tokens.shape
    mask = torch.ones_like(tokens, dtype=torch.float32)
    
    for i in range(batch_size):
        # Find the first occurrence of pad_token_id (real EOS)
        pad_positions = (tokens[i] == pad_token_id).nonzero(as_tuple=False)
        
        if len(pad_positions) > 0:
            # Keep the first EOS, mask everything after it
            first_eos_pos = pad_positions[0].item()
            mask[i, first_eos_pos:] = 0  # Mask everything AFTER the first EOS
            # The first EOS at position first_eos_pos remains with mask=1
    
    return mask

def apply_masking(input_probs, tokens, input_prompt_length, pad_token_id, mode = 'logprobs'):
    generated_tokens = tokens[:, input_prompt_length:]
    mask = create_attention_mask_from_tokens(generated_tokens, pad_token_id=pad_token_id)  # Assuming pad_token_id is 0
    
    if mode == 'logprobs':
        assert generated_tokens.shape == input_probs.shape, \
        f"Mismatch in generated tokens ({generated_tokens.shape}) and logprobs ({input_probs.shape}) shapes"
        masked_logprobs = input_probs * mask
        
    elif mode == 'early_exit_probs':
        assert generated_tokens.shape == input_probs.shape[:-1], \
        f"Mismatch in generated tokens ({generated_tokens.shape}) and logprobs ({input_probs.shape}) shapes" # The last dim is num_exit_layers
        masked_logprobs = input_probs * mask.unsqueeze(-1)
    
    else:
        raise ValueError(f"Unknown mode {mode} in apply_masking")
    
    return masked_logprobs

@torch.no_grad()
def compute_sample_labels(verification_rewards):
    """
    Compute correctness and format labels for samples, similar to format_accuracy/answer_accuracy in reference.

    Args:
        verification_rewards: Tensor of reward values [-1.0, 1.0]
        completions_texts: List of completion text strings

    Returns:
        dict with label counts and ratios
    """
    labels = []
    for i, reward in enumerate(verification_rewards):
        reward_val = reward.item()

        # Determine correctness label based on reward value
        if reward_val == 1.0:
            correctness_label = 'correct_format'
        elif reward_val >= 0.5:
            correctness_label = 'correct_noformat'
        elif reward_val == 0.0:
            correctness_label = 'incorrect_format'
        elif reward_val >= -1.0:
            correctness_label = 'incorrect_noformat'
        else:
            correctness_label = 'unknown'

        # Determine format quality based on presence of ####
        #completion_text = completions_texts[i]
        #has_format_marker = '####' in completion_text
        format_label = 'good_format' if correctness_label in ['correct_format', 'incorrect_format'] else 'poor_format'

        labels.append({
            'correctness': correctness_label,
            'format_quality': format_label
        })

    # Compute statistics
    total_samples = len(labels)
    correct_count = sum(1 for l in labels if l['correctness'] == 'correct_format')
    #partial_count = sum(1 for l in labels if l['correctness'] == 'partial')
    #incorrect_count = sum(1 for l in labels if l['correctness'] == 'incorrect')
    #format_error_count = sum(1 for l in labels if l['correctness'] == 'format_error')
    good_format_count = sum(1 for l in labels if l['format_quality'] == 'good_format')

    return {
        'labels': labels,
        'stats': {
            'answer_accuracy': correct_count / total_samples,
            #'partial_rate': partial_count / total_samples,
            #'incorrect_rate': incorrect_count / total_samples,
            #'format_error_rate': format_error_count / total_samples,
            #'good_format_rate': good_format_count / total_samples,
            #'answer_accuracy': (correct_count + partial_count) / total_samples,  # Similar to reference
            'format_accuracy': good_format_count / total_samples  # Similar to reference
        }
    }

async def evaluate_coherence(prompt: str, response: str) -> dict:
    """
    Evaluate coherence of a response using GPT-4 judge model.
    Returns a dictionary with all coherence metrics.
    """
    eval_prompt = f"""
You are evaluating the coherence and quality of an AI assistant's chain of thought and response.
Allow exploratory/meandering reasoning as long as it stays internally consistent.

Original Question: {prompt}

AI Response: {response}

Please evaluate this response on the following criteria:
1. Coherence and logical flow (1-10 scale) Does the reasoning follow a sensible progression, even if unfinished?
2. Completeness of the reasoning (1-10 scale) Does it reach a correct and explicit chain of thought? If partial but on-track and relevant to answer, award mid-range (4–7).
3. Clarity and readability (1-10 scale) Is it easy to follow? Minor meandering is okay if readable.
4. Absence of repetition or errors (1-10 scale) Penalize contradictions, factual mistakes about the prompt, or heavy repetition.

Rate each criterion and provide an overall score from 1-10:
- 1: major breakdown (nonsensical, off-topic)
- 4: noticeable issues but some useful reasoning
- 7: generally solid, with minor flaws or cut-offs
- 10: excellent, complete, and polished

Meta / Wrapper Policy:
- The evaluation input may include wrapper/markup such as: angle-bracket role tags (e.g., <｜User｜>, <｜Assistant｜>) and <think>.
- These wrappers are expected and should not reduce scores for Clarity, Coherence, or No Repetition.

Format your response as:
Coherence: X/10
Completeness: X/10
Clarity: X/10
No Repetition: X/10
Overall: X/40

Brief explanation: [your reasoning]
"""
    
    judge_model = get_inspect_model("openai/gpt-4")
    eval_result = await judge_model.generate(eval_prompt)
    
    eval_text = eval_result.completion
    
    coherence_score = 0
    completeness_score = 0
    clarity_score = 0
    no_repetition_score = 0
    overall_score = 0
    
    for line in eval_text.split('\n'):
        line = line.strip()
        if line.startswith('Coherence:'):
            coherence_score = int(line.split(':')[1].strip().split('/')[0])
        elif line.startswith('Completeness:'):
            completeness_score = int(line.split(':')[1].strip().split('/')[0])
        elif line.startswith('Clarity:'):
            clarity_score = int(line.split(':')[1].strip().split('/')[0])
        elif line.startswith('No Repetition:'):
            no_repetition_score = int(line.split(':')[1].strip().split('/')[0])
        elif line.startswith('Overall:'):
            overall_score = int(line.split(':')[1].strip().split('/')[0])
    
    #extract explanation (everything after "Brief explanation:")
    explanation = ""
    explanation_start = eval_text.find("Brief explanation:")
    if explanation_start != -1:
        explanation = eval_text[explanation_start + len("Brief explanation:"):].strip()
    else:
        explanation = eval_text
    
    return {
        'coherence': coherence_score,
        'completeness': completeness_score,
        'clarity': clarity_score,
        'no_repetition': no_repetition_score,
        'average': overall_score / 4.0 if overall_score > 0 else (coherence_score + completeness_score + clarity_score + no_repetition_score) / 4.0,
        'explanation': explanation
    }


def load_gsm8k_with_difficulty():

    enriched_file = hf_hub_download(
        repo_id="lizardp1/gsm8k_early_exit",
        filename="rl_with_difficulty.parquet",
        repo_type="dataset"
    )
    
    dataset = Dataset.from_parquet(enriched_file)
    gsm8k_dataset = {'train': dataset}
    
    return gsm8k_dataset


def load_tom(
    repo_id: str = "lizardp1/tom_externalization",
    filename: str = "tom_rl.csv"
) -> dict[str, Dataset]:

    csv_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset"
    )

    ds_dict = load_dataset("csv", data_files={"train": csv_path})
    ds = ds_dict["train"]

    return {"train": ds}


@torch.no_grad()
def compute_accuracy_by_difficulty(verify_rewards, difficulty_categories):
    sample_labels = compute_sample_labels(verify_rewards)
    labels = sample_labels['labels']
    
    difficulty_stats = {}
    
    for label, difficulty in zip(labels, difficulty_categories):
        if difficulty not in difficulty_stats:
            difficulty_stats[difficulty] = {'correct': 0, 'good_format': 0, 'total': 0}
        
        difficulty_stats[difficulty]['total'] += 1
        
        if label['correctness'] == 'correct_format':
            difficulty_stats[difficulty]['correct'] += 1
        
        if label['format_quality'] == 'good_format':
            difficulty_stats[difficulty]['good_format'] += 1
    
    results = {}

    for difficulty in difficulty_stats:
        total = difficulty_stats[difficulty]['total']
        format_acc = difficulty_stats[difficulty]['good_format'] / total
        answer_acc = difficulty_stats[difficulty]['correct'] / total
        results[f'{difficulty.lower()}_format_accuracy'] = format_acc
        results[f'{difficulty.lower()}_answer_accuracy'] = answer_acc
    
    return results