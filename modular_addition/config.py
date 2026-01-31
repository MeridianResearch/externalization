"""Centralized configuration for modular addition experiments."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    max_seq_len: int = 256
    dropout: float = 0.1


@dataclass
class PretrainConfig:
    epochs: int = 50
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    dataset_size: int = 100_000
    eval_size: int = 5_000
    num_operands_range: tuple[int, int] = (2, 6)
    save_path: str = "checkpoints/pretrain"


@dataclass
class SFTReasoningConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 0.01
    dataset_size: int = 100_000
    eval_size: int = 5_000
    num_operands_range: tuple[int, int] = (2, 8)
    pretrain_checkpoint: str = "checkpoints/pretrain/model.pt"
    save_path: str = "checkpoints/sft_reasoning"


@dataclass
class RLReasoningConfig:
    epochs: int = 20
    batch_size: int = 32
    k: int = 4  # rollouts per prompt
    lr: float = 1e-5
    max_new_tokens: int = 64
    dataset_size: int = 50_000
    num_operands_range: tuple[int, int] = (2, 8)
    sft_checkpoint: str = "checkpoints/sft_reasoning/model.pt"
    save_path: str = "checkpoints/rl_reasoning"


@dataclass
class SFTEarlyExitConfig:
    epochs: int = 20
    batch_size: int = 64
    lr_lora: float = 1e-4
    lr_exit: float = 1e-5
    lora_r: int = 16
    lora_alpha: int = 32
    base_checkpoint: str = "checkpoints/rl_reasoning/model.pt"
    save_path: str = "checkpoints/sft_early_exit"


@dataclass
class RLEarlyExitConfig:
    epochs: int = 20
    batch_size: int = 32
    k: int = 4
    lr_lora: float = 1e-4
    lr_exit: float = 1e-5
    beta_kl: float = 0.05
    lambda_exit: float = 0.5
    max_new_tokens: int = 64
    sft_ee_checkpoint: str = "checkpoints/sft_early_exit/model.pt"
    base_checkpoint: str = "checkpoints/rl_reasoning/model.pt"
    save_path: str = "checkpoints/rl_early_exit"


@dataclass
class Config:
    p: int = 113
    device: str = "cuda"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    sft_reasoning: SFTReasoningConfig = field(default_factory=SFTReasoningConfig)
    rl_reasoning: RLReasoningConfig = field(default_factory=RLReasoningConfig)
    sft_early_exit: SFTEarlyExitConfig = field(default_factory=SFTEarlyExitConfig)
    rl_early_exit: RLEarlyExitConfig = field(default_factory=RLEarlyExitConfig)
