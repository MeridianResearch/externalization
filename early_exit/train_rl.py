"""


- Flow: K rollouts per prompt → compute rewards (verify - beta*KL - lambda*avg_exit_layer) → center per-prompt → weighted SFT.
"""

import torch
from torch.optim import Adam
import wandb
from datasets import load_dataset

from early_exit.util import get_model, load_model
from early_exit.rewards import compute_verification_rewards
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode
from shared_utils.load import get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text

device = "cuda"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
config_path = "config_deepseek.yaml"
sft_model_path = "models/early_exit_sft_trained"  # TODO: set path to SFT checkpoint

K = 4  # TODO: number of rollouts per prompt (resource-constrained)
beta_kl = 0.1  # TODO: KL penalty weight (to sweep)
lambda_exit = 0.5  # TODO: early-exit average-layer penalty weight (to sweep)


# --- Models (schema) ---
tokenizer = get_tokenizer(model_name)
config = configs_from_yaml(config_path, tokenizer.eos_token_id)

student = get_model(model_name, config['model'], device)
student = replace_attention_layers(student, config['lora'], device)
# TODO: Load SFT checkpoint weights for student
# student = load_model(student, sft_model_path)  # TODO

# Reference policy: base unmodified model without early exit
reference = get_model(model_name, config['model'], device)
# TODO: ensure no early-exit logic is active for reference model

# Dataset
dataset = load_dataset("gsm8k", "main")  # TODO: verify/parse answer format


# --- Core schema functions ---
def generate_k_completions(model, prompts, k: int):
    """
    TODO: Free-generate K completions per prompt with early exits enabled.
    Returns:
        completions:
            - tokens: LongTensor [batch*K, seq_len]  (student-sampled sequences)
            - texts: list[str] length batch*K
        exit_info:
            - avg_exit_layer: FloatTensor [batch*K] (average exit layer per sequence)
            - prescribed_exit_layers: LongTensor [batch*K, seq_len] (optional; for re-scoring)
    """
    # TODO: set_transformer_early_exit_mode(model, 'free_generate') and call generate_text(...)
    raise NotImplementedError("TODO: implement generate_k_completions")


def compute_verification_rewards(completions_text, correct_answers):
    """
    TODO: Return FloatTensor [batch*K] with +1 for correct formatted answers, 0 otherwise.
    TODO: enforce format like '#### <answer>'; penalize misformatted outputs.
    """
    raise NotImplementedError("TODO: implement compute_verification_rewards")


def compute_token_logprobs_student(model, tokens, prescribed_exit_layers=None):
    """
    TODO: Re-score student-sampled sequences with gradient-attached log-probs per token.
    - Optionally prescribe the exit layers taken during generation (student SFT mode).
    Returns: FloatTensor [batch*K, seq_len] of log p_student(y_t | y_<t, x).
    """
    raise NotImplementedError("TODO: implement compute_token_logprobs_student")


def compute_token_logprobs_reference(model, tokens):
    """
    TODO: Score the same sequences under the reference policy (no early exit).
    Returns: FloatTensor [batch*K, seq_len] of log p_ref(y_t | y_<t, x).
    """
    raise NotImplementedError("TODO: implement compute_token_logprobs_reference")


def compute_token_kl_from_logprobs(student_logprobs, reference_logprobs):
    """
    TODO: Compute average per-token KL between student and reference over the sequence.
    Returns: FloatTensor [batch*K] (e.g., mean over seq_len of KL_t).
    """
    raise NotImplementedError("TODO: implement compute_token_kl_from_logprobs")


def compute_avg_exit_layer(exit_info):
    """
    TODO: Extract/compute average exit layer per sequence from exit_info.
    Returns: FloatTensor [batch*K]. Consider normalizing by max exitable layer.
    """
    raise NotImplementedError("TODO: implement compute_avg_exit_layer")


def center_rewards_per_prompt(rewards, batch_size: int, k: int):
    """
    Center rewards across the K completions for each prompt (simple baseline).
    Returns: advantages FloatTensor [batch*K].
    """
    rewards = rewards.view(batch_size, k)
    adv = rewards - rewards.mean(dim=1, keepdim=True)
    return adv.reshape(-1)


def compute_sequence_loglik_student(model, tokens, prescribed_exit_layers=None):
    """
    TODO: Sum token log-probs over generated tokens per sequence for weighted SFT.
    Returns: FloatTensor [batch*K].
    """
    raise NotImplementedError("TODO: implement compute_sequence_loglik_student")


def weighted_sft_step(model, tokens, advantages, optimizer, prescribed_exit_layers=None):
    """
    TODO: One weighted SFT step using sequence log-likelihoods.
    loss = -mean(adv.detach() * seq_loglik_student)
    Returns: scalar loss (FloatTensor).
    """
    raise NotImplementedError("TODO: implement weighted_sft_step")


def main_rl_training():
    """
    Schema: Generate → Reward → Center → Weighted SFT
    """
    # TODO: optimizer (e.g., Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=1e-5))

    # TODO: wandb.init(project=..., config=...)

    # TODO: batching. For simplicity, treat batch_size = 1 here.
    train_dataset = dataset["train"]
    for i, example in enumerate(train_dataset):


        prompt = example["question"]
        correct_answer = example["answer"]

        # 1) Rollouts (student free-generate K)
        completions, exit_info = generate_k_completions(student, [prompt], k=K)  # TODO

        # 2) Log-probs for KL and rewards (reference vs student)  # TODO: confirm scoring design

        ref_logprobs = compute_token_logprobs_reference(reference, completions['tokens'])  # TODO
        stu_logprobs = compute_token_logprobs_student(student, completions['tokens'], prescribed_exit_layers=exit_info.get('prescribed_exit_layers', None))  # TODO

        # 3) Reward components
        verify = compute_verification_rewards(completions['texts'], [correct_answer] * K)  # TODO
        kl_tokens = compute_token_kl_from_logprobs(stu_logprobs, ref_logprobs)  # TODO
        avg_exit_layer = compute_avg_exit_layer(exit_info)  # TODO

        # 4) Total reward per sequence (simple linear combination)
        reward = verify - beta_kl * kl_tokens - lambda_exit * avg_exit_layer  # TODO: tune weights, consider normalization

        # 5) Centering per prompt
        advantages = center_rewards_per_prompt(reward, batch_size=1, k=K)

        # 6) Weighted SFT update
        loss = weighted_sft_step(student, completions['tokens'], advantages, optimizer, prescribed_exit_layers=exit_info.get('prescribed_exit_layers', None))  # TODO

        # 7) Logging (schema)
        # TODO: wandb.log({ 'step': i, 'loss': ..., 'reward/mean': ..., 'verify/acc': ..., 'kl/tokens_mean': ..., 'exit/avg_layer': ... })

