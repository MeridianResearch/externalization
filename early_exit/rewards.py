import re
import torch
from torch import Tensor as _T

def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]
    if method == "strict":
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str) #looks for specific format #### <answer>
        if len(solutions) == 0:
            final_answer = None
        else:
            #take last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str) #looks for any number, with/wo commas
        final_answer = None
        if len(answer) == 0:
            #no reward if no answer
            pass
        else:
            invalid_str = ["", "."]
            #find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer

def check_format_violations(completion_text):
    penalty = 0.0
    
    hash_count = completion_text.count("####") #check for multiple #### 
    if hash_count > 1:
        penalty += 0.1
    
    final_hash_match = re.search(r"#### (\\-?[0-9\\.\\,]+)(.+)", completion_text, re.DOTALL) #check for text after final #### answer
    if final_hash_match:
        text_after = final_hash_match.group(2).strip()
        if text_after:
            penalty += 0.1
    
    malformed_hash = re.findall(r"#### (?![0-9\\-])", completion_text) #check for #### with no number after
    if malformed_hash:
        penalty += 0.1
    
    return penalty

def compute_verification_rewards(completions_text, correct_answers):

    rewards = torch.zeros(len(completions_text), dtype=torch.float32)
    
    for i, completion_text in enumerate(completions_text):
        ground_truth_idx = i // len(correct_answers) if len(correct_answers) > 1 else 0
        ground_truth = str(correct_answers[ground_truth_idx])
        
        if "#### " in ground_truth:
            answer_match = re.search(r"#### (.+)", ground_truth)
            if answer_match:
                ground_truth = answer_match.group(1).strip().replace(",", "").replace("$", "")
        
        extracted_answer = extract_solution(completion_text, method="strict")
        extracted_answer_flexible = extract_solution(completion_text, method="flexible")

        format_penalty = check_format_violations(completion_text)
        
        if extracted_answer is not None and extracted_answer == ground_truth:
            rewards[i] = 1.0 - format_penalty #full reward minus any format penalties
        elif extracted_answer_flexible is not None and extracted_answer_flexible == ground_truth:
            rewards[i] = 0.5  #penalize misformatted outputs
        elif extracted_answer is None:
            rewards[i] = -1.0  #extra penalty for wrong format and no answer
        else:
            rewards[i] = 0.0 #wrong answer but correct format
    
    return rewards

def compute_next_token_logprobs_from_logits(logits, tokens):
    """
    Given model logits and next tokens, compute log-probabilities of all the generated tokens.

    Args:
        logits (FloatTensor): Model output logits of shape [batch*K, full_seq_len, vocab_size].
        tokens (LongTensor): Input token IDs of shape [batch*K, full_seq_len].

    Returns:
        next_token_logprobs (FloatTensor): Tensor of shape [batch*K, gen_len] containing
            log p(y_t | y_<t, x) for all the tokens including the prompt rollout.
    """
    # Here B = batch*K, T = full_seq_len, V = vocab_size
    assert logits.dim() == 3 and tokens.dim() == 2, "Shapes must be [B,T,V] and [B,T]"
    B, T, V = logits.shape
    assert tokens.shape == (B, T), "tokens must be [B, T]"

    # log-softmax over vocab
    output_logprobs = logits.log_softmax(-1)  # [B, T, V]

    # for each time step t (0..T-2), pick log p of token at t+1 from step t
    # align shapes: index must be [B, T-1, 1]
    idx = tokens[:, 1:].unsqueeze(-1)                # [B, T-1, 1]

    next_token_logprobs = torch.gather(
        output_logprobs[:, :-1, :], dim=-1, index=idx
    ).squeeze(-1)                                    # [B, T-1]

    return next_token_logprobs

def compute_token_logprobs_student(model, tokens, prescribed_exit_layers, input_prompt_length):
    """
    Compute per-token log-probabilities under the student model for a sampled sequence.

    Args:
        model: The student model, expected to support `prescribed_exit_layer_idxs`.
        tokens (LongTensor): Input token IDs of shape [batch*K, seq_len].
        prescribed_exit_layers (LongTensor): Exit layer indices used during generation
            (student SFT mode).

    Returns:
        student_generated_token_logprobs (FloatTensor): Tensor of shape [batch*K, gen_len]
            containing log p_student(y_t | y_<t, x) for the generated tokens.
    """
    # TODO: Should the model be in a free generation mode or the student mode? Add assert statement accordingly.
    # Current implementation assumes student mode.
    student_output_scores, collected_exit_logits = model(tokens, prescribed_exit_layer_idxs = prescribed_exit_layers) # [batch * samples, full length, vocabulary]
    student_next_token_logprobs = compute_next_token_logprobs_from_logits(student_output_scores.logits, tokens)
    # gen_len = tokens.shape[-1] - prescribed_exit_layers.shape[-1] # Check this, is there a -1 needed?
    student_generated_token_logprobs = student_next_token_logprobs[:, input_prompt_length:]
    return student_generated_token_logprobs

def compute_token_logprobs_reference(model, tokens, input_prompt_length):
    """
    Compute per-token log-probabilities under the reference model for a sampled sequence.
    Unlike the student, the reference model does not use early exiting.

    Args:
        model: The reference (teacher) model.
        tokens (LongTensor): Input token IDs of shape [batch*K, seq_len].
        gen_len (int): Number of generated tokens to score (excludes prompt length).

    Returns:
        reference_generated_ntp_logprobs (FloatTensor): Tensor of shape [batch*K, gen_len]
            containing log p_ref(y_t | y_<t, x) for the generated tokens.
    """
    # raise NotImplementedError("TODO: implement compute_token_logprobs_reference")
    outputs = model(tokens) # [batch * samples, full length, vocabulary]
    next_token_logprobs = compute_next_token_logprobs_from_logits(outputs['logits'], tokens)
    reference_generated_ntp_logprobs = next_token_logprobs[:, input_prompt_length:]
    return reference_generated_ntp_logprobs


def compute_token_kl_from_logprobs(student_generated_ntp_logprobs, reference_generated__ntp_logprobs):
    """
    Compute the average per-token KL-like divergence term between student and reference
    log-probabilities.

    Args:
        student_generated_ntp_logprobs (FloatTensor): Tensor [batch*K, gen_len] of
            log p_student(y_t | y_<t, x).
        reference_generated_ntp_logprobs (FloatTensor): Tensor [batch*K, gen_len] of
            log p_ref(y_t | y_<t, x).

    Returns:
        kl_estimate (FloatTensor):  Tensor [batch*K]
            (log p_student - log p_ref) across tokens.
            Note: this is not the full KL divergence over the vocabulary.
    """

    logprobs_diff = student_generated_ntp_logprobs - reference_generated__ntp_logprobs
    kl_estimate = logprobs_diff.sum(-1)
    return kl_estimate

    
def compute_avg_exit_layer(prescribed_exit_layers, model):
    """
    Extract/compute average exit layer per sequence from prescribed_exit_layers.

    Returns:
        FloatTensor [batch*K]: Average exit layer per sequence.
    """
    
    total_layers = model.config.num_hidden_layers if hasattr(model, 'config') else 28 #get total layers from config
    final_layer_idx = float(total_layers - 1)  #0-indexed

    avg_exit_layers = []
    for exit_layers in prescribed_exit_layers:
        finite_layers = torch.where( #replace inf with final layer idx
            torch.isinf(exit_layers), 
            torch.full_like(exit_layers, final_layer_idx),
            exit_layers.float()
        )
        #avg over the actual generation length (no padding as messes up mean)
        avg_exit_layers.append(finite_layers.mean())

    avg_exit_layers = torch.stack(avg_exit_layers)

    avg_exit_layers = avg_exit_layers / final_layer_idx #if normalizing
    
    return avg_exit_layers