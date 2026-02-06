"""Data generation for modular addition pretraining."""

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from modular_addition.tokenizer import ModularAdditionTokenizer


class PretrainDataset(Dataset):
    """Dataset for modular addition pretraining.

    Format: <bos> a b [c] = result <eos>
    """

    def __init__(
        self,
        tokenizer: ModularAdditionTokenizer,
        n_samples: int = 10000,
        n_operands: tuple[int, int] = (2, 3),
        seed: int | None = None,
    ):
        if seed is not None:
            random.seed(seed)

        self.sequences = []
        for _ in range(n_samples):
            n = random.randint(n_operands[0], n_operands[1])
            operands = [random.randint(0, tokenizer.p - 1) for _ in range(n)]
            result = sum(operands) % tokenizer.p
            seq = [tokenizer.bos_token_id] + operands + [tokenizer.eq_token_id, result, tokenizer.eos_token_id]
            self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"input_ids": self.sequences[idx]}


@dataclass
class DataCollator:
    """Pads sequences. Loss only on answer + <eos> (last 2 tokens)."""

    tokenizer: ModularAdditionTokenizer

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        seqs = [ex["input_ids"] for ex in examples]
        max_len = max(len(s) for s in seqs)

        input_ids = []
        attention_mask = []
        labels = []

        for seq in seqs:
            seq_len = len(seq)
            pad_len = max_len - seq_len
            input_ids.append(seq + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append([1] * seq_len + [0] * pad_len)
            labels.append([-100] * (seq_len - 2) + seq[-2:] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }
