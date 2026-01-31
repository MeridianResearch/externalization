"""Small decoder-only transformer for modular addition, HuggingFace compatible."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput


class ModularAdditionConfig(PretrainedConfig):
    model_type = "modular_addition"

    def __init__(
        self,
        vocab_size: int = 119,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        pad_token_id: int = 118,
        bos_token_id: int = 116,
        eos_token_id: int = 117,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_size = d_model  # alias for patching compatibility
        self.torch_dtype = torch.float32
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout


class TransformerBlock(nn.Module):
    def __init__(self, config: ModularAdditionConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(config.d_model)
        self.self_attn = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        # Attach config and layer_idx to self_attn for patching compatibility
        self.self_attn.config = config
        self.self_attn.layer_idx = layer_idx
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        h = self.ln1(hidden_states)
        h, _ = self.self_attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        hidden_states = hidden_states + h
        hidden_states = hidden_states + self.mlp(self.ln2(hidden_states))
        return hidden_states


class ModularAdditionModel(PreTrainedModel, GenerationMixin):
    config_class = ModularAdditionConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: ModularAdditionConfig):
        super().__init__(config)
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        B, T = input_ids.shape
        device = input_ids.device

        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

        # Key padding mask (True = ignore)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        for layer in self.layers:
            out = layer(hidden_states=x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
            x = out[0] if isinstance(out, tuple) else out

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {"input_ids": input_ids, "attention_mask": attention_mask}
