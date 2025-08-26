"""


- Flow: K rollouts per prompt → compute rewards (verify - beta*KL - lambda*avg_exit_layer) → center per-prompt → weighted SFT.
"""

import torch
from torch.optim import Adam
import wandb
from datasets import load_dataset
from dataclasses import dataclass

from early_exit.util import get_model, load_model, load_model_from_wandb
from early_exit.rewards import compute_verification_rewards, compute_token_kl_from_logprobs, compute_token_logprobs_reference, compute_token_logprobs_student
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode
from shared_utils.load import get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text

device = "cuda"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
config_path = "config_deepseek.yaml"
sft_model_path = "models/early_exit_sft_trained"  # TODO: set path to SFT checkpoint

@dataclass
class RLHyperparams:
    batch_size: int = 1  # for simplicity, keep batch_size=1 
    k: int = 4  # number of rollouts per prompt (resource-constrained)
    beta_kl: float = 0.1  # KL penalty weight (to sweep)
    lambda_exit: float = 0.5  # early-exit average-layer penalty weight (to sweep)


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


def compute_sequence_loglik_student(student_log_likelihoods):
    """
    TODO: Sum token log-probs over generated tokens per sequence for weighted SFT.
    Returns: FloatTensor [batch*K].
    """
    return student_log_likelihoods.sum(-1)

def weighted_rloo_loss(advantages, log_likelihoods, RL_HPARAMS):
    
    """Implements one weighted RLOO step, using unlabelled 
    equation after equation 8 in https://arxiv.org/pdf/2402.14740v2.pdf. 
    It is impoertant to detch adv

    Args:
        advantages (FloatTensor): Shape (batch_size, k) containing the advantage values.
        log_likelihoods (FloatTensor): Shape (batch_size, k) containing the sum of log_likelihood of next tokens
    """
    assert advantages.shape == log_likelihoods.shape, "adv and log_likelihood_gradients must have the same shape"
    num_exit_samples = RL_HPARAMS.k
    batch_size = RL_HPARAMS.batch_size
    advantages = advantages.view(batch_size, num_exit_samples)
    log_likelihoods = log_likelihoods.view(batch_size, num_exit_samples)
    return (advantages.detach() * log_likelihoods).mean(-1)


def weighted_sft_step(student_log_likelihoods, advantages, optimizer, num_exit_samples):
    """
    TODO: One weighted SFT step using sequence log-likelihoods.
    loss = -mean(adv.detach() * seq_loglik_student)
    Returns: scalar loss (FloatTensor).
    """
    optimizer.zero_grad()
    sequence_log_likelihoods = compute_sequence_loglik_student(student_log_likelihoods)  # [batch*K]
    loss = weighted_rloo_loss(advantages, sequence_log_likelihoods, num_exit_samples) # should this be average instead of sum?
    loss.backward()
    optimizer.step()


def main_rl_training():
    """
    Schema: Generate → Reward → Center → Weighted SFT
    """
    # TODO: optimizer (e.g., Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=1e-5))
    # Check if there are better optimizers for this problem
    optimizer = Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=1e-5)
    # TODO: wandb.init(project=..., config=...)

    # TODO: batching. For simplicity, treat batch_size = 1 here.
    train_dataset = dataset["train"]
    for i, example in enumerate(train_dataset):


        prompt = example["question"]
        correct_answer = example["answer"]

        # 1) Rollouts (student free-generate K)
        completions, exit_info = generate_k_completions(student, [prompt], k=RL_HPARAMS.k)  # TODO

        # 2) Log-probs for KL and rewards (reference vs student)  # TODO: confirm scoring design

        ref_logprobs = compute_token_logprobs_reference(reference, completions['tokens'])  # TODO
        stu_logprobs = compute_token_logprobs_student(student, completions['tokens'], prescribed_exit_layers=exit_info.get('prescribed_exit_layers', None))  # TODO

        # 3) Reward components
        verify = compute_verification_rewards(completions['texts'], [correct_answer] * RL_HPARAMS.k)  # TODO
        kl_tokens = compute_token_kl_from_logprobs(stu_logprobs, ref_logprobs)  # TODO
        avg_exit_layer = compute_avg_exit_layer(exit_info)  # TODO

        # 4) Total reward per sequence (simple linear combination)
        reward = verify - RL_HPARAMS.beta_kl * kl_tokens - RL_HPARAMS.lambda_exit * avg_exit_layer  # TODO: tune weights, consider normalization

        # 5) Centering per prompt
        advantages = center_rewards_per_prompt(reward, batch_size=1, k=RL_HPARAMS.k)
        # Normalize advantages by their standard deviation to stabilize learning
        adv_std = advantages.std(unbiased=False) + 1e-8
        advantages = advantages / adv_std

        # 6) Weighted SFT update
        
        weighted_sft_step(stu_logprobs, advantages, optimizer, RL_HPARAMS)  # TODO
        # 7) Logging (schema)
        # TODO: wandb.log({ 'step': i, 'loss': ..., 'reward/mean': ..., 'verify/acc': ..., 'kl/tokens_mean': ..., 'exit/avg_layer': ... })

