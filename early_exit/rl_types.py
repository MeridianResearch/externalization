from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class RLHyperparams:
    """
    Hyperparameters for RL training

    - batch_size: number of p rompts per batch
    - k: number of rollouts per prompt
    - beta_kl: weight for KL penalty
    - lambda_exit: weight for average exit layer penalty
    - sample_log_interval: episodes between logging sample generations
    - sample_max_rows: maximum number of sample generations to log per event
    - sample_log_selection: how to choose rows for samples table ('random'|'first')
    - sample_log_seed: optional seed for reproducible sample selection
    """
    batch_size: int = 1
    k: int = 4  
    beta_kl: float = 0.05
    lambda_exit: float = 0.5
    sample_log_interval: int = 25
    sample_max_rows: int = 2
    sample_log_selection: str = "first"
    sample_log_seed: Optional[int] = None
    system_prompt: Optional[str] = "I am going to give you a math word problem. Solve it step by step, showing your reasoning. After your work, provide your final numerical answer"
    SAVE_EVERY_HOURS: int = 5
    REWARD_CORRECT: float = 1.0
    start_epoch: int = 0
    lr_lora: float = 1e-4
    lr_exit: float = 1e-5
    SAVE_MODEL: bool = False
    max_new_tokens: int = 900

@dataclass
class RolloutBatch:
    """
    Container with runtime checks for rollout tensors and metadata.

    - tokens: LongTensor [batch*K, seq_len]
    - texts: list[str] length batch*K
    - ref_logprobs: FloatTensor [batch*K, seq_len]
    - stu_logprobs: FloatTensor [batch*K, seq_len]
    - prescribed_exit_layers (optional): LongTensor [batch*K, seq_len]
    - avg_exit_layer (optional): FloatTensor [batch*K]
    - input_prompt_length (optional): int, number of tokens in the input prompt to exclude from logprobs
    """
    tokens: torch.Tensor
    texts: List[str]
    ref_logprobs: torch.Tensor
    stu_logprobs: torch.Tensor
    prescribed_exit_layers: Optional[torch.Tensor] = None
    avg_exit_layer: Optional[torch.Tensor] = None
    input_prompt_length: Optional[int] = None
    student_early_exit_logprobs: Optional[torch.Tensor] = None
    generation_length = property(lambda self: self.tokens.shape[1] - self.input_prompt_length if self.input_prompt_length is not None else self.tokens.shape[1])

    def __post_init__(self) -> None:
        # tokens
        assert isinstance(self.tokens, torch.Tensor), "tokens must be a torch.Tensor"
        assert self.tokens.dtype == torch.long, "tokens must be dtype torch.long"
        assert self.tokens.dim() == 2, "tokens must be 2D [batch*K, seq_len]"
        batchK, seq_len = self.tokens.shape

        # texts
        assert isinstance(self.texts, list), "texts must be a list[str]"
        assert len(self.texts) == batchK, "len(texts) must equal batch*K"
        assert all(isinstance(t, str) for t in self.texts), "texts list must contain only strings"

        # logprobs
        for name, lp in (("ref_logprobs", self.ref_logprobs), ("stu_logprobs", self.stu_logprobs)):
            assert isinstance(lp, torch.Tensor), f"{name} must be a torch.Tensor"
            assert torch.is_floating_point(lp), f"{name} must be a floating tensor"
            # TODO: Check the -1 for the shape. Why generation_length - 1?
            assert lp.dim() == 2 and tuple(lp.shape) == (batchK, self.generation_length), \
            f"{name} has shape {lp.shape}, instead it must be [batch*K = {batchK}, generation_length = {self.generation_length}]"

        # prescribed exit layers (optional)
        if self.prescribed_exit_layers is not None:
            pel = self.prescribed_exit_layers
            assert isinstance(pel, torch.Tensor), "prescribed_exit_layers must be a torch.Tensor"
            # Removing the inf assertion since we are removing the first element in the generate_rollouts_function
            # assert (pel[:, 0] == torch.inf).all(), "prescribed_exit_layers[:, 0] must be torch.inf to indicate no early exit on the first rollout token"
            # TODO: Check if we need long dtype for this
            # assert pel.dtype == torch.long, f"prescribed_exit_layers must be dtype torch.long instead of {pel.dtype}"
            assert pel.dim() == 2 and tuple(pel.shape) == (batchK, self.generation_length),\
            f"prescribed_exit_layers has shape = {pel.shape} must be [batch*K = {batchK}, generation_length = {self.generation_length}]"

        # avg exit layer (optional)
        if self.avg_exit_layer is not None:
            ael = self.avg_exit_layer
            assert isinstance(ael, torch.Tensor), "avg_exit_layer must be a torch.Tensor"
            assert torch.is_floating_point(ael), "avg_exit_layer must be a floating tensor"
            assert ael.dim() == 1 and tuple(ael.shape) == (batchK,), "avg_exit_layer must be [batch*K]"
            
        # student early exit probs (optional)
        if self.student_early_exit_logprobs is not None:
            seep = self.student_early_exit_logprobs
            assert isinstance(seep, torch.Tensor), "student_early_exit_probs must be a torch.Tensor"
            assert torch.is_floating_point(seep), "student_early_exit_probs must be a floating tensor"
            assert seep.dim() == 3 and tuple(seep.shape) == (batchK, self.generation_length, seep.shape[2]), \
                f"student_early_exit_probs has shape = {seep.shape} must be [batch*K = {batchK}, generation_length = {self.generation_length}, num_exit_layers]"


