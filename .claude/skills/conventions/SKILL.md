---
description: Coding conventions and patterns for this project
---

# Conventions

## Config
- Use `@dataclass` for all configs (see `modular_addition/config.py`)
- YAML files for runtime config, dataclasses for defaults and typing
- `load_config()` parses YAML → nested dataclasses
- Access config via attributes (`cfg.model.n_embd`), never dict access (`cfg["model"]["n_embd"]`)
- No `.get("key", default)` — let it break if missing

## Model
- Use HuggingFace `GPT2LMHeadModel` + `GPT2Config`, not custom models
- Save with `model.save_pretrained(path)`, load with `GPT2LMHeadModel.from_pretrained(path)`
- Tokenizer: custom `ModularAdditionTokenizer` (not HF tokenizer)

## Data
- `Dataset.__getitem__` returns `{"input_ids": [...]}` — single example, no padding
- `DataCollator` handles padding at batch time (dynamic padding)
- Right padding for training, left padding for generation
- Labels: `-100` at pad positions (ignored by loss)
- GPT2 handles label shifting internally

## Training scripts
- No tqdm — use plain print with all metrics on one line:
  `print(f"Epoch {epoch}: train_loss={..:.4f} train_acc={..:.4f} test_loss={..:.4f} test_acc={..:.4f}")`
- Extract `evaluate()` as a reusable function, run on both train and test loaders
- Batch accuracy computation (use `gather`, no per-sample loops)
- Timestamp in `output_dir` path (`outputs/pretrain/20260206_143022/`)
- Save config.yaml to output_dir at start of training
- Support `num_logs` OR `log_every`, `num_checkpoints` OR `save_every`
- Best model saved based on test loss

## General
- Keep code minimal and simple
- No over-engineering or unnecessary abstractions
- Prefer standard HuggingFace patterns over custom implementations
