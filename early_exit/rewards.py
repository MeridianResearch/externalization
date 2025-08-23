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
        
        if extracted_answer is not None and extracted_answer == ground_truth:
            rewards[i] = 1.0
        elif extracted_answer_flexible is not None and extracted_answer_flexible == ground_truth:
            rewards[i] = -1.0  #penalize misformatted outputs
        else:
            rewards[i] = 0.0
    
    return rewards

def compute_token_kl_from_logprobs(student_logprobs: _T, reference_logprobs: _T, eps: float = 1e-16) -> _T:
    """
    Compute average per-token KL divergence between student and reference distributions over each sequence.
    
    This function calculates KL(student || reference) for each token position, then averages across
    the sequence length to get a single KL value.
    
    TODO: Confirm shapes of input tensors.
    Args:
        student_logprobs: [batch*K, seq_len, vocab_size] - Log probabilities from student model
        reference_logprobs: [batch*K, seq_len, vocab_size] - Log probabilities from reference model  
        eps: float - Small epsilon value to prevent log(0) errors
        
    Returns:
        Single float tensor - Average per-token KL divergence for each sequence in the batch
    """    
    student_probs = student_logprobs.exp()
    reference_probs = reference_logprobs.exp()
    token_logits_kl_div = (student_probs * ((student_probs + eps) / (reference_probs + eps)).log()).sum(-1)   # [batch * samples, gen len]
    mean_logit_kl = token_logits_kl_div.mean()
    return mean_logit_kl