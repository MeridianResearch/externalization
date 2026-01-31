"""Dataset generation for modular addition."""

from __future__ import annotations

import random
from typing import Literal

import torch
from torch.utils.data import Dataset

from modular_addition.tokenizer import ModularAdditionTokenizer


def generate_reasoning_trace(operands: list[int], p: int) -> list[list[int]]:
    """Generate a step-by-step reasoning trace by randomly reducing contiguous subsets.

    Returns a list of intermediate states, where each state is the current list of numbers.
    E.g. [23, 45, 67, 89] -> [[23, 45, 67, 89], [23, 112, 89], [23, 88], [111]]
    """
    states = [list(operands)]
    current = list(operands)
    while len(current) > 1:
        # Pick a random contiguous subset of size >= 2
        max_start = len(current) - 2
        start = random.randint(0, max_start)
        max_end = len(current)
        end = random.randint(start + 2, max_end)
        # Reduce the subset
        reduced = sum(current[start:end]) % p
        current = current[:start] + [reduced] + current[end:]
        states.append(list(current))
    return states


class ModularAdditionDataset(Dataset):
    """Dataset for modular addition problems.

    Modes:
        pretrain: <bos> a b c = answer <eos>
        sft:      <bos> a b c = <think> step1 = step2 = ... </think> = answer <eos>
        rl:       <bos> a b c = (prompt only)
    """

    def __init__(
        self,
        tokenizer: ModularAdditionTokenizer,
        mode: Literal["pretrain", "sft", "rl"] = "pretrain",
        size: int = 10_000,
        num_operands_range: tuple[int, int] = (2, 8),
        max_seq_len: int = 256,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.p = tokenizer.p

        rng = random.Random(seed)
        self.samples = []
        for _ in range(size):
            n_ops = rng.randint(*num_operands_range)
            operands = [rng.randint(0, self.p - 1) for _ in range(n_ops)]
            self.samples.append(operands)

    def __len__(self) -> int:
        return len(self.samples)

    def _build_pretrain_tokens(self, operands: list[int]) -> list[int]:
        tok = self.tokenizer
        answer = sum(operands) % self.p
        return [tok.bos_token_id] + operands + [tok.eq_token_id, answer, tok.eos_token_id]

    def _build_sft_tokens(self, operands: list[int]) -> list[int]:
        tok = self.tokenizer
        states = generate_reasoning_trace(operands, self.p)
        # Build: <bos> operands = <think> state1 = state2 = ... </think> = final <eos>
        tokens = [tok.bos_token_id] + operands + [tok.eq_token_id, tok.think_token_id]
        for i, state in enumerate(states[1:]):  # skip initial state (same as operands)
            if i > 0:
                tokens.append(tok.eq_token_id)
            tokens.extend(state)
        tokens += [tok.end_think_token_id, tok.eq_token_id, states[-1][0], tok.eos_token_id]
        return tokens

    def _build_rl_tokens(self, operands: list[int]) -> list[int]:
        tok = self.tokenizer
        return [tok.bos_token_id] + operands + [tok.eq_token_id]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        operands = self.samples[idx]
        if self.mode == "pretrain":
            tokens = self._build_pretrain_tokens(operands)
        elif self.mode == "sft":
            tokens = self._build_sft_tokens(operands)
        elif self.mode == "rl":
            tokens = self._build_rl_tokens(operands)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Pad/truncate
        tokens = tokens[: self.max_seq_len]
        pad_len = self.max_seq_len - len(tokens)
        attention_mask = [1] * len(tokens) + [0] * pad_len
        input_ids = tokens + [self.tokenizer.pad_token_id] * pad_len

        # Labels: -100 for padding and prompt tokens (pretrain: mask everything before =)
        if self.mode == "pretrain":
            eq_pos = tokens.index(self.tokenizer.eq_token_id)
            labels = [-100] * (eq_pos + 1) + tokens[eq_pos + 1 :] + [-100] * pad_len
        elif self.mode == "sft":
            eq_pos = tokens.index(self.tokenizer.eq_token_id)
            labels = [-100] * (eq_pos + 1) + tokens[eq_pos + 1 :] + [-100] * pad_len
        else:  # rl - no labels needed
            labels = [-100] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
