from transformers.utils import logging
logger = logging.get_logger(__name__)

from early_exit.patching.attention_mixins.base import LayerFakeAttentionForwardMixin

import torch
from torch.nn import functional as F
from torch import nn, Tensor as _T, FloatTensor as _FT, LongTensor as _LT

import math

from typing import List, Optional, Tuple

from transformers import Cache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv, Qwen2Attention



class Qwen2DecoderLayerFakeAttentionForwardMixin(LayerFakeAttentionForwardMixin):

    def patched_layer_forward(
        self,
        hidden_states: _T,
        attention_mask: Optional[_T] = None,
        position_ids: Optional[_LT] = None,
        past_key_value: Optional[Tuple[_T]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[_LT] = None,
        position_embeddings: Optional[Tuple[_T, _T]] = None,
        unfrozen_idx_or_mask: Optional[List[int] | _T] = None,
        **kwargs,
    ) -> Tuple[_FT, Optional[Tuple[_FT, _FT]]]:
        
        # Store original hidden states
        original_hidden_states = hidden_states.clone()
        
        bsz, q_len, _ = hidden_states.size()
        
        # Process unfrozen mask
        if isinstance(unfrozen_idx_or_mask, list):
            unfrozen_mask = torch.zeros(bsz, dtype=torch.bool, device=hidden_states.device)
            if len(unfrozen_idx_or_mask) > 0:
                unfrozen_mask[unfrozen_idx_or_mask] = True
            unfrozen_elements = unfrozen_mask
        elif isinstance(unfrozen_idx_or_mask, _T):
            gen_len = unfrozen_idx_or_mask.shape[1]
            padding_required = q_len - gen_len
            unfrozen_elements = F.pad(
                input=unfrozen_idx_or_mask,
                pad=(padding_required, 0),
                value=True  # Pre-rollout (prompt) residual stream never gets frozen
            ).to(hidden_states.device)
        elif unfrozen_idx_or_mask is None:
            unfrozen_elements = torch.ones(bsz, dtype=torch.bool, device=hidden_states.device)
        
        # Call parent's forward method
        outputs = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs
        )
        
        # Extract hidden states from outputs
        if isinstance(outputs, tuple):
            new_hidden_states = outputs[0]
            other_outputs = outputs[1:]
        else:
            new_hidden_states = outputs
            other_outputs = ()

        if unfrozen_elements.ndim == 1:
            mask = unfrozen_elements.view(bsz, 1, 1)
        else:
            mask = unfrozen_elements.unsqueeze(-1)
            
        final_hidden_states = torch.where(
            mask,                              # Expand mask to match hidden dimension
            new_hidden_states,                 # Use new values where unfrozen
            original_hidden_states             # Keep original values where frozen
        )
        
        
        # Reconstruct outputs with updated hidden states
        final_outputs = (final_hidden_states,) + other_outputs
        
        # Debug assertion
        assert (original_hidden_states == final_hidden_states)[~unfrozen_elements].all()
        
        return final_outputs

    