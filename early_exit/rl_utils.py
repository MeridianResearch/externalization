import torch
from torch.optim import Adam

from early_exit.patching import set_transformer_early_exit_mode
from shared_utils.generate import generate_text
from typing import List

from early_exit.rl_types import *

from shared_utils.generate import generate_text

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
            
            all_tokens.append(tokens)
            all_texts.append(decoded_response)
            all_prescribed_exit_layers.append(prescribed_exit_layers)
    
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


def compute_sequence_mean_loglik_student(student_log_likelihoods):
    """
    TODO: Add the exit log probs!
    Returns: FloatTensor [batch*K].
    """
    return student_log_likelihoods.mean(-1)


def weighted_rloo_loss(advantages, log_likelihoods, RL_HPARAMS):
    """Computes loss on one weighted RLOO step, based on unlabelled 
    equation after equation 8 in https://arxiv.org/pdf/2402.14740v2.pdf. 
    It is important to detch advantages so that no gradients flow through them.
    loss = -mean(adv.detach() * seq_loglik_student)
    Args:
        advantages (FloatTensor): Shape (batch_size, k) containing the advantage values.
        log_likelihoods (FloatTensor): Shape (batch_size, k) containing the sum of log_likelihood of next tokens
    """
    assert advantages.shape == log_likelihoods.shape, "adv and log_likelihood_gradients must have the same shape"
    num_exit_samples = RL_HPARAMS.k
    batch_size = RL_HPARAMS.batch_size
    advantages = advantages.view(batch_size, num_exit_samples)
    log_likelihoods = log_likelihoods.view(batch_size, num_exit_samples)
    return -(advantages.detach() * log_likelihoods).mean(-1)


def weighted_sft_step(student_log_likelihoods, advantages, optimizer, RL_HPARAMS):
    """
    Returns: scalar loss (FloatTensor).
    """
    optimizer.zero_grad()
    sequence_mean_log_likelihoods = compute_sequence_mean_loglik_student(student_log_likelihoods)  # [batch*K]
    loss = weighted_rloo_loss(advantages, sequence_mean_log_likelihoods, RL_HPARAMS)
    loss.backward()
    optimizer.step()
    return loss.detach()


def get_input_prompt_length(tokenizer, prompt, system_prompt):
    from shared_utils.generate import format_conversation, transform_conversations
    pre_transformed_conversation = format_conversation(user_prompts=[prompt], system_prompt=system_prompt)
    formatted_prompt = transform_conversations(pre_transformed_conversation, prefiller='')[0]
    input_prompt_length = len(tokenizer(formatted_prompt)['input_ids'])
    return input_prompt_length