import re
import torch
from torch import Tensor as _T
from rapidfuzz.distance import Levenshtein as L

'''def extract_solution(solution_str, method="strict"):
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
    '''
#above is old extract solution

def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]
    if method == "strict":
        #try \boxed{} format first (Qwen's preferred format)
        solutions = re.findall(r"\\boxed\{([+-]?[0-9\\.\\,]+)\}", solution_str)
            
        #fallback for **Final Answer:** format
        if len(solutions) == 0:
            solutions = re.findall(r"\*\*Final Answer:\*\*[\\s\\n]*([+-]?[0-9\\.\\,]+)", solution_str)
            
        if len(solutions) == 0:
            final_answer = None
        else:
            final_answer = solutions[-1].replace(",", "").replace("$", "")
            
    elif method == "flexible":
        answer = re.findall(r"([+-]?[0-9\\.\\,]+)", solution_str) #looks for any number, with/wo commas
        final_answer = None
        if len(answer) == 0:
            #no reward if no answer
            pass
        else:
            invalid_str = ["", "."]
            #find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    final_answer = final_answer.replace(",", "").replace("$", "")
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

def compute_verification_rewards(completions_tokens, completions_text, correct_answers, input_prompt_length, tokenizer):
    rewards = torch.zeros(len(completions_text), dtype=torch.float32)
    
    for i, full_completion_text in enumerate(completions_text):
        completion_tokens = completions_tokens[i][input_prompt_length:] #remove prompt
        completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True) #tokens to text
        
        ground_truth_idx = i // len(correct_answers) if len(correct_answers) > 1 else 0
        ground_truth = str(correct_answers[ground_truth_idx])
        
        if "#### " in ground_truth:
            answer_match = re.search(r"#### (.+)", ground_truth)
            if answer_match:
                ground_truth = answer_match.group(1).strip().replace(",", "").replace("$", "")
        
        extracted_answer = extract_solution(completion_text, method="strict")
        extracted_answer_flexible = extract_solution(completion_text, method="flexible")
        
        if extracted_answer is not None and extracted_answer == ground_truth:
            rewards[i] = 1.0  #full reward for correct answer in proper format
        elif extracted_answer_flexible is not None and extracted_answer_flexible == ground_truth:
            rewards[i] = 0.5  #partial reward for correct answer but wrong format
        elif extracted_answer is None:
            rewards[i] = -1.0  #penalty for no extractable answer
        else:
            rewards[i] = 0.0  #wrong answer
    
    return rewards

def extract_solution_text(solution_str: str, method: str = "strict"):
    """
    strict   : extract the text after the LAST 'Answer:' line.
    flexible : return the LAST non-empty line of the text.
    """
    assert method in ["strict", "flexible"]

    if not isinstance(solution_str, str) or not solution_str.strip():
        return None

    s = solution_str

    s = s.replace("<｜begin▁of▁sentence｜>", "").replace("<｜end▁of▁sentence｜>", "")
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S | re.I)
    s = re.sub(r"<\|.*?\|>", "", s, flags=re.S)  # remove <|...|> tokens

    if method == "strict":
        parts = re.split(r"<\|\s*Assistant\s*\|>", s, flags=re.I)
        if len(parts) >= 2:
            s = parts[-1]

        matches = list(re.finditer(
            r"^\s*(?:\*\*\s*)?answer(?:\s*\*\*)?\s*[:：]\s*(.+?)\s*$",
            s, flags=re.I | re.M
        ))
        if not matches:
            return None

        ans = matches[-1].group(1).strip()
        ans = re.sub(r"^\*\*|\*\*$", "", ans).strip()
        ans = ans.replace("<｜end▁of▁sentence｜>", "").strip()
        return ans or None

    else:  # method == "flexible"
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        return (lines[-1] if lines else None)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def ok_english_char(ch: str) -> bool:
    o = ord(ch)

    #common whitespace controls
    if ch in ("\n", "\t", "\r"):
        return True

    #ASCII printable
    if 0x20 <= o <= 0x7E:
        return True

    #common English typography / spacing
    if o in (0x00A0, 0x2026) or (0x2018 <= o <= 0x201F) or (0x2013 <= o <= 0x2014):
        return True

    return False

def compute_verification_rewards_text(completions_tokens, completions_text, correct_answers, input_prompt_length, tokenizer):
    rewards = torch.zeros(len(completions_text), dtype=torch.float32)

    format_ok = []
    sims = []
    exacts = []

    for i, _ in enumerate(completions_text):
        completion_tokens = completions_tokens[i][input_prompt_length:] #remove prompt
        completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True) #tokens to text

        ground_truth_idx = 0 if len(correct_answers) == 1 else (i % len(correct_answers))
        ground_truth = _norm(str(correct_answers[ground_truth_idx]))

        extracted_answer = extract_solution_text(completion_text, method="strict")
        extracted_answer_flexible = extract_solution_text(completion_text, method="flexible")

        if extracted_answer is not None:
            p = _norm(extracted_answer)
            if p == ground_truth:
                rewards[i] = 1.0
                fmt_ok, sim, ex = True, 1.0, True
            else:
                sim = float(L.normalized_similarity(p, ground_truth)) / 100.0
                rewards[i] = sim
                fmt_ok, ex = True, False
        else:
            if extracted_answer_flexible is None:
                rewards[i] = -1.0
                fmt_ok, sim, ex = False, 0.0, False
            else:
                p = _norm(extracted_answer_flexible)
                if p == ground_truth:
                    rewards[i] = 0.5
                    fmt_ok, sim, ex = False, 1.0, True
                else:
                    sim = float(L.normalized_similarity(p, ground_truth)) / 100.0
                    rewards[i] = -1.0 + sim
                    fmt_ok, ex = False, False

        #extra penalty for non English characters
        #bad = sum(1 for ch in completion_text if not ok_english_char(ch))
        #ratio = bad / max(1, len(completion_text))
        #penalty = 0.5 * ratio
        #rewards[i] = rewards[i] - penalty

        #extra penalty for multiple "Answer:"
        answer_count = len(re.findall(r"^\s*(?:\*\*\s*)?answer(?:\s*\*\*)?\s*[:：]", completion_text, flags=re.I | re.M))
        if answer_count > 1:
            multi_answer_penalty = 0.2 * (answer_count - 1)  # 0.2 penalty per extra
            rewards[i] = rewards[i] - multi_answer_penalty

        format_ok.append(fmt_ok)
        sims.append(sim)
        exacts.append(ex)

    aux = {'format_ok': torch.tensor(format_ok, dtype=torch.bool), 'similarity': torch.tensor(sims, dtype=torch.float32), 'exact': torch.tensor(exacts, dtype=torch.bool)}
    return rewards, aux


'''
def compute_verification_rewards(completions_tokens, completions_text, correct_answers, input_prompt_length, tokenizer):

    rewards = torch.zeros(len(completions_text), dtype=torch.float32)
    
    for i, full_completion_text in enumerate(completions_text):
        completion_tokens = completions_tokens[i][input_prompt_length:] #remove prompt
        completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True) #tokens to text

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
            rewards[i] = 0.0 - format_penalty #wrong answer but correct format
    
    return rewards
'''

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
    assert tokens.shape[-1] - prescribed_exit_layers.shape[-1] == input_prompt_length, "Mismatch in prescripted exit layers and prompt length"
    student_output_scores, collected_exit_logits = model(tokens, prescribed_exit_layer_idxs = prescribed_exit_layers) # [batch * samples, full length, vocabulary]
    student_next_token_logprobs = compute_next_token_logprobs_from_logits(student_output_scores.logits, tokens)
    student_generated_token_logprobs = student_next_token_logprobs[:, input_prompt_length-1:] # the input_promt_length -1 is to capture all the generated tokens, not a trivial thing
    student_early_exit_logprobs = (model.early_exit_student_probs(collected_exit_logits) + 1e-16).log() 
    generated_logits = student_output_scores.logits[:, input_prompt_length-1:-1] #added for entropy calc
    return student_generated_token_logprobs, student_early_exit_logprobs, generated_logits

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
    with torch.no_grad():
        outputs = model(tokens) # [batch * samples, full length, vocabulary]
        next_token_logprobs = compute_next_token_logprobs_from_logits(outputs['logits'], tokens)
        reference_generated_ntp_logprobs = next_token_logprobs[:, input_prompt_length-1:] # the input_promt_length -1 is to capture all the generated tokens, not a trivial thing
    return reference_generated_ntp_logprobs


def compute_token_kl_from_logprobs(student_generated_ntp_logprobs, reference_generated__ntp_logprobs, attention_mask):
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
    assert logprobs_diff.shape == attention_mask.shape, "Log probs and attention mask should be of same shapes"
    kl_estimate = logprobs_diff.sum(-1)/attention_mask.sum(-1)
    return kl_estimate

    
def compute_avg_exit_layer(prescribed_exit_layers, model):
    """
    Extract/compute average exit layer per sequence from prescribed_exit_layers.

    Returns:
        FloatTensor [batch*K]: Average exit layer per sequence.
    """
    
    total_layers = model.config.num_hidden_layers if hasattr(model, 'config') else 36 #get total layers from config
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