# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project trains language models (DeepSeek) to externalize reasoning through early exit mechanisms. Models can terminate computation at intermediate transformer layers and proceed directly to final readout weights, forcing reasoning to be serialized into interpretable Chain-of-Thought tokens rather than processed internally.

## Setup

```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Key Commands

```bash
# SFT training
python early_exit/sft_train.py --config config_deepseek.yaml

# RL training (experimental)
python early_exit/train_rl.py --config config_deepseek.yaml

# Evaluation
python early_exit/eval.py
```

## Architecture

### Training Modes
- **SFT** (`early_exit/sft_train.py`, `sft_train_gsm8k.py`, `sft_train_teacher_data.py`): Supervised fine-tuning where early exit weights and LoRA adapters learn from teacher reasoning traces.
- **RL** (`early_exit/train_rl.py`, `rl_train_sft_base.py`): Reinforcement learning to optimize exit timing with rewards for earlier exits. Uses `rl_utils.py` for training utilities and `rewards.py` for reward functions. Types in `rl_types.py`.

### Core Mechanism (`early_exit/patching/`)
- `method_patching.py`: Runtime patching of model forward passes to inject early exit logic.
- `dynamical_types.py`: Types for the patching system.
- `model_mixins/`: Mixins that modify model forward behavior to support early exits.
- `attention_mixins/`: Mixins that modify attention layers for early exit support.

The early exit mechanism uses stochastic scalar readout weights at each transformer layer to determine exit probability. When triggered, the residual stream freezes and passes directly to final readout weights.

### Configuration
- `config_deepseek.yaml`: Main config (model loading, LoRA params, generation settings).
- `config_greedy.yaml`: Greedy decoding variant.
- LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`.

### Supporting Code
- `early_exit/util.py`: Shared utilities and helpers.
- `shared_utils/`: Data processing, demo utilities.
- `teacher_data/`: Notebooks for generating teacher model reasoning traces.
- `results_and_data/`: Training datasets and experimental results.
