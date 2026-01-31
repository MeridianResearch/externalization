# Externalization via Early Exit Mechanisms — Modular Addition

A framework for training a small transformer on modular addition (mod p) and then applying early exit mechanisms to externalize reasoning.

## Overview

This branch trains a from-scratch decoder-only transformer to solve modular addition problems (e.g. `23 45 67 = 22` mod 113), then applies early exit mechanisms to force the model to externalize its reasoning into interpretable Chain-of-Thought tokens rather than processing it internally.

## Training Pipeline

### Stage 1: Base Reasoning Model

1. **Pretrain** (`modular_addition/pretrain.py`): Train from scratch to output answers for modular addition (2–8 operands, mod p=113).
2. **SFT on Reasoning** (`modular_addition/sft_reasoning.py`): Fine-tune on step-by-step reasoning traces with `<think>` tags.
3. **RL for Correctness** (`modular_addition/rl_reasoning.py`): RLOO policy gradient with reward = 1 if final answer correct, 0 otherwise.

### Stage 2: Early Exit SFT

**SFT Early Exit** (`modular_addition/sft_train.py`): Apply early exit patching + LoRA, train with teacher/student distillation.

### Stage 3: Early Exit RL

**RL Early Exit** (`modular_addition/rl_train.py`): RL training with rewards for correctness, KL regularization, and exit layer penalty.

## Data Format

- **Tokenizer**: Custom tokenizer with tokens `0..112`, `=`, `<think>`, `</think>`, `<bos>`, `<eos>`, `<pad>` (119 tokens, no `+` — operands are space-separated).
- **Pretrain**: `<bos> 23 45 67 = 22 <eos>`
- **SFT**: `<bos> 23 45 67 89 = <think> 23 112 89 = 23 88 = 111 </think> = 111 <eos>`
- **RL**: `<bos> 23 45 67 89 =` (model generates the rest)

## Repository Structure

```
externalization/
├── modular_addition/     # Modular addition experiment
│   ├── tokenizer.py      # Custom tokenizer (0..p-1 + special tokens)
│   ├── data.py           # Dataset generation (pretrain/sft/rl modes)
│   ├── model.py          # Small decoder-only transformer (HuggingFace compatible)
│   ├── config.py         # Centralized hyperparameters
│   ├── pretrain.py       # Stage 1a: pretrain on answers
│   ├── sft_reasoning.py  # Stage 1b: SFT on reasoning traces
│   ├── rl_reasoning.py   # Stage 1c: RL for correctness
│   ├── sft_train.py      # Stage 2: SFT early exit
│   └── rl_train.py       # Stage 3: RL early exit
├── early_exit/           # Core early exit implementation
│   ├── patching/         # Model and attention layer modifications
│   ├── rl_utils.py       # RL training utilities
│   ├── rewards.py        # Reward functions
│   └── util.py           # Utilities and helper functions
├── shared_utils/         # Common utilities for data processing
└── tests/                # Evaluation scripts
```

## Setup

```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

```bash
# Stage 1a: Pretrain
python modular_addition/pretrain.py

# Stage 1b: SFT reasoning
python modular_addition/sft_reasoning.py

# Stage 1c: RL reasoning
python modular_addition/rl_reasoning.py

# Stage 2: SFT early exit
python modular_addition/sft_train.py

# Stage 3: RL early exit
python modular_addition/rl_train.py
```

### Technical Details

- **Model**: Small decoder-only transformer (default: d=256, 6 layers, 8 heads, ~5M params)
- **Early Exit**: Stochastic scalar readout weights at each layer determine exit probability
- **Residual Stream Freezing**: On exit, residual stream is frozen and passed to final readout weights
- **Patching**: Runtime patching via `early_exit/patching/` — supports both Qwen2 and custom models
