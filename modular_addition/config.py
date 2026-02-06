"""Configuration dataclasses for modular addition training."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    n_positions: int = 64
    n_embd: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_inner: int | None = None
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0


@dataclass
class DataConfig:
    n_samples: int = 50000
    n_operands: tuple[int, int] = (2, 3)
    train_frac: float = 0.6
    seed: int = 42


@dataclass
class PretrainConfig:
    p: int = 113
    device: str = "cuda"
    output_dir: str = "outputs/pretrain"

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.1
    epochs: int = 200

    # Logging: set num_logs OR log_every
    num_logs: int | None = None
    log_every: int | None = 5

    # Checkpoints: set num_checkpoints OR save_every
    num_checkpoints: int | None = 5
    save_every: int | None = None
