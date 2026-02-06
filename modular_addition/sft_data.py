"""Data generation for SFT reasoning on modular addition with chain-of-thought."""

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from modular_addition.tokenizer import ModularAdditionTokenizer


def generate_flexible_cot_sequence(
    operands: list[int],
    tokenizer: ModularAdditionTokenizer,
    combine_probs: tuple[float, float] = (0.5, 0.5),
) -> list[int]:
    """Generate a CoT sequence where each step randomly combines 2 or 3 operands.

    Format: <bos> a b c d <think> a b c d = result1 c d = result2 d = final </think> final <eos>
    """
    p = tokenizer.p

    # Start: <bos> operands <think> operands
    sequence = [tokenizer.bos_token_id] + list(operands) + [tokenizer.think_token_id] + list(operands)

    current = list(operands)

    while len(current) > 1:
        sequence.append(tokenizer.eq_token_id)

        if len(current) <= 2:
            n_combine = len(current)
        elif random.random() < combine_probs[0]:
            n_combine = 2
        else:
            n_combine = 3

        partial_sum = sum(current[:n_combine]) % p
        current = [partial_sum] + current[n_combine:]
        sequence.extend(current)

    final_answer = current[0]
    sequence.append(tokenizer.end_think_token_id)
    sequence.append(final_answer)
    sequence.append(tokenizer.eos_token_id)

    return sequence


class SFTReasoningDataset(Dataset):
    """Dataset for SFT reasoning with flexible CoT.

    Format: <bos> ops <think> ops = partial ... </think> answer <eos>
    Labels: -100 up to and including <think>, actual tokens after that.
    """

    def __init__(
        self,
        tokenizer: ModularAdditionTokenizer,
        n_samples_per_n: int = 10000,
        n_operands_min: int = 4,
        n_operands_max: int = 8,
        combine_probs: tuple[float, float] = (0.5, 0.5),
        seed: int | None = None,
    ):
        if seed is not None:
            random.seed(seed)

        self.sequences = []
        self.labels_list = []

        for n_operands in range(n_operands_min, n_operands_max + 1):
            for _ in range(n_samples_per_n):
                operands = [random.randint(0, tokenizer.p - 1) for _ in range(n_operands)]
                seq = generate_flexible_cot_sequence(operands, tokenizer, combine_probs)

                # Labels: -100 up to and including <think>, actual tokens after
                think_pos = seq.index(tokenizer.think_token_id)
                labels = [-100] * (think_pos + 1) + seq[think_pos + 1:]

                self.sequences.append(seq)
                self.labels_list.append(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"input_ids": self.sequences[idx], "labels": self.labels_list[idx]}


@dataclass
class SFTDataCollator:
    """Pads input_ids and labels for SFT reasoning."""

    tokenizer: ModularAdditionTokenizer

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        seqs = [ex["input_ids"] for ex in examples]
        labs = [ex["labels"] for ex in examples]
        max_len = max(len(s) for s in seqs)

        input_ids = []
        attention_mask = []
        labels = []

        for seq, lab in zip(seqs, labs):
            pad_len = max_len - len(seq)
            input_ids.append(seq + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append([1] * len(seq) + [0] * pad_len)
            labels.append(lab + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }
