"""Custom tokenizer for modular addition with think tags."""

from __future__ import annotations


class ModularAdditionTokenizer:
    """Tokenizer for modular addition sequences.

    Vocab: 0..p-1 (number tokens), =, <think>, </think>, <bos>, <eos>, <pad>
    Numbers are space-separated (no + token).
    """

    def __init__(self, p: int = 113):
        self.p = p
        # Number tokens: 0..p-1
        # Special tokens start at index p
        self.eq_token_id = p
        self.think_token_id = p + 1
        self.end_think_token_id = p + 2
        self.bos_token_id = p + 3
        self.eos_token_id = p + 4
        self.pad_token_id = p + 5
        self.vocab_size = p + 6

        self._special_tokens = {
            "=": self.eq_token_id,
            "<think>": self.think_token_id,
            "</think>": self.end_think_token_id,
            "<bos>": self.bos_token_id,
            "<eos>": self.eos_token_id,
            "<pad>": self.pad_token_id,
        }
        self._id_to_token = {}
        for i in range(p):
            self._id_to_token[i] = str(i)
        for tok, idx in self._special_tokens.items():
            self._id_to_token[idx] = tok

        self._token_to_id = {v: k for k, v in self._id_to_token.items()}

    def encode(self, text: str) -> list[int]:
        """Encode a string into token ids."""
        tokens = text.split()
        return [self._token_to_id[t] for t in tokens]

    def decode(self, ids: list[int] | 'torch.Tensor') -> str:
        """Decode token ids into a string."""
        import torch
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return " ".join(self._id_to_token[i] for i in ids if i != self.pad_token_id)

    def __len__(self) -> int:
        return self.vocab_size
