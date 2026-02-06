# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This branch (`rl_modular_addition`) trains a small decoder-only transformer on modular addition (mod 113) with step-by-step reasoning, then applies early exit mechanisms to externalize reasoning into interpretable Chain-of-Thought tokens.

**Base Model**: `models/sft_think_v1/` — SFT model trained on reasoning traces with `<think>` tags (94.6% accuracy on 4 operands, degrades to 30.7% on 8 operands).

## Setup

```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Key Commands

```bash
# Stage 1a: Pretrain base model on modular addition answers
python modular_addition/pretrain.py

# Stage 1b: SFT on step-by-step reasoning traces
python modular_addition/sft_reasoning.py

# Stage 1c: RL for correctness (RLOO policy gradient)
python modular_addition/rl_reasoning.py

# Stage 2: SFT early exit (apply patching + LoRA)
python modular_addition/sft_train.py

# Stage 3: RL early exit (correctness + KL + exit penalty)
python modular_addition/rl_train.py
```

## Architecture

### Data Format
- **Tokenizer** (`modular_addition/tokenizer.py`): 119 tokens — `0..112` (mod p=113), `=`, `<think>`, `</think>`, `<bos>`, `<eos>`, `<pad>`. Numbers are space-separated (no `+` token).
- **Pretrain**: `<bos> 23 45 67 = 22 <eos>`
- **SFT reasoning**: `<bos> 23 45 67 89 = <think> 23 112 89 = 23 88 = 111 </think> = 111 <eos>`
- **RL**: `<bos> 23 45 67 89 =` (model generates the rest)

### Model
- **Architecture** (`modular_addition/model.py`): Small decoder-only transformer (~5M params), HuggingFace-compatible (`ModularAdditionModel`).
- **Default config** (`modular_addition/config.py`): d_model=256, 4 layers, 4 heads, max_seq_len=16.

### Early Exit Mechanism (`early_exit/patching/`)
- **`method_patching.py`**: Runtime patching of model forward passes. Key function: `replace_attention_layers()` which injects early exit decision heads and wraps the model with LoRA.
- **Mixins**: `model_mixins/modular_addition.py` and `attention_mixins/modular_addition.py` implement model-specific early exit logic.
- **Exit Modes**: `'off'`, `'sft_teacher'`, `'sft_student'`, `'free_generate'` — controlled via `set_transformer_early_exit_mode()`.
- **Mechanism**: Stochastic scalar readout weights at each transformer layer determine exit probability. On exit, residual stream freezes and passes directly to final readout weights.

### Configuration
- All hyperparameters in `modular_addition/config.py` as dataclasses: `ModelConfig`, `PretrainConfig`, `SFTReasoningConfig`, `RLReasoningConfig`, `SFTEarlyExitConfig`, `RLEarlyExitConfig`.
- LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`.

### RL Training
- **RLOO policy gradient**: Reward = 1 if final answer correct, 0 otherwise.
- **Early exit RL**: Correctness reward + KL regularization (`beta_kl`) + exit layer penalty (`lambda_exit`).
- Utilities in `early_exit/rl_utils.py`, reward functions in `early_exit/rewards.py`, types in `early_exit/rl_types.py`.
