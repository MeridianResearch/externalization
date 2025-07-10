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

        _original_hidden_states = hidden_states.clone()

        bsz, q_len, _ = hidden_states.size()

        if isinstance(unfrozen_idx_or_mask, list):
            unfrozen_elements = unfrozen_idx_or_mask
            
        elif isinstance(unfrozen_idx_or_mask, _T):
            # XXX: CHECK MASK AND ATTENTION ALIGNMENT BY TIME
            gen_len = unfrozen_idx_or_mask.shape[1]
            padding_required = q_len - gen_len
            unfrozen_elements = F.pad(
                input = unfrozen_idx_or_mask,
                pad = (padding_required, 0),
                value = True    # Pre-rollout (prompt) residual stream never gets frozen
            )

        elif unfrozen_idx_or_mask is None:
            unfrozen_elements = torch.ones((bsz, q_len), dtype=torch.bool, device=hidden_states.device)
            
        residual = hidden_states

        hidden_states[unfrozen_elements] = self.input_layernorm(hidden_states[unfrozen_elements])
        # Self Attention
        attention_output = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                unfrozen_idx_or_mask=unfrozen_idx_or_mask       # Key change
            )
        ### TODO: The following statements are very hacky. Please change it in the future versions
        if len(attention_output) == 3:
            hidden_states, self_attn_weights, present_key_value = attention_output
        else:
            hidden_states, self_attn_weights = attention_output
            present_key_value = None
        # print("Unfrozen elements = ", unfrozen_elements) 
        # print("Unfrozen idx or mask = ", unfrozen_idx_or_mask) 
        # print("Not Unfrozen elements = ", ~unfrozen_elements)
        # print("Frozen Hidden states shape = ", hidden_states[~unfrozen_elements].shape)      
        # print("Hidden states sum = ", hidden_states[~unfrozen_elements].sum())
        # print("Residual states sum = ", residual[~unfrozen_elements].sum())
        # print("Original hidden states sum = ", _original_hidden_states[~unfrozen_elements].sum())
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states[unfrozen_elements] = self.post_attention_layernorm(hidden_states[unfrozen_elements])
        hidden_states[unfrozen_elements] = self.mlp(hidden_states[unfrozen_elements])
        hidden_states[unfrozen_elements] = residual[unfrozen_elements] + hidden_states[unfrozen_elements]

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache and present_key_value is not None:
            outputs += (present_key_value,)
        
        assert (_original_hidden_states == hidden_states)[~unfrozen_elements].all()

        return outputs


    @staticmethod
    def patched_attention_forward(
        self: Qwen2Attention,
        hidden_states: _T,
        attention_mask: Optional[_T] = None,
        position_ids: Optional[_LT] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[_LT] = None,
        position_embeddings: Optional[Tuple[_T, _T]] = None,  # will become mandatory in v4.46
        *_,
        unfrozen_idx_or_mask: List[int] | _T,
    ) -> Tuple[_T, Optional[_T], Optional[Tuple[_T]]]:
        """
        Outputs (attn_outputs, self_attn_weights, present_key_value)

        With change to residual stream:
            hidden_states = residual + attn_outputs
        """
        if unfrozen_idx_or_mask is None:
            op = self.base_self_attn_forward(
                hidden_states = hidden_states,
                attention_mask = attention_mask,
                position_ids = position_ids,
                past_key_value = past_key_value,
                output_attentions = output_attentions,
                use_cache = use_cache,
                cache_position = cache_position,
                position_embeddings = position_embeddings,
            )
            return op

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # # print("Number of attention heads = ", self.config)if not hasattr(self, "num_heads"):
        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.config.num_key_value_heads
        if not hasattr(self, "hidden_size"):
            self.hidden_size = self.config.hidden_size

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)


        # Differences begin: only make changes to unfrozen_idx_or_mask, rest has a zero attention output
        # During generation we should only freeze after the prompt scoring (which is just cache building)
        if isinstance(unfrozen_idx_or_mask, list):
            unfrozen_batch_idx = unfrozen_idx_or_mask
            assert q_len == 1, "Cannot use unfrozen_idx_or_mask as List unless sequence length == 1 (i.e. in free generation mode)"
        
        # During SFT with prescribed early exists at each timestep
        # We do the full forward pass with all batch items, but only accept changes from those which would not be frozen at this timestep
        elif isinstance(unfrozen_idx_or_mask, _T):
            gen_len = unfrozen_idx_or_mask.shape[1]
            assert unfrozen_idx_or_mask.shape[0] == bsz
            assert unfrozen_idx_or_mask.dtype == torch.bool
            unfrozen_batch_idx = list(range(bsz))

        else:
            raise TypeError

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states[unfrozen_batch_idx], self.num_key_value_groups)
        value_states = repeat_kv(value_states[unfrozen_batch_idx], self.num_key_value_groups)
        query_states = query_states[unfrozen_batch_idx]

        # print(f"Unfrozen batch indices: {unfrozen_batch_idx}")
        # print(f"Key states shape: {key_states.shape}")
        # print(f"Value states shape: {value_states.shape}")
        # print(f"Query states shape: {query_states.shape}")
        # print(f"Past Key Value: {past_key_value}")

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:  # no matter the length, we just slice it
            attention_mask = attention_mask[unfrozen_batch_idx]
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        
        if attn_output.size() != (len(unfrozen_batch_idx), self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(len(unfrozen_batch_idx), q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        
        attn_output_with_zeros = torch.zeros(bsz, *attn_output.shape[1:], device = attn_output.device, dtype = attn_output.dtype)
        attn_weights_with_zeros = torch.zeros(bsz, *attn_weights.shape[1:], device = attn_weights.device, dtype = attn_weights.dtype)

        # In generative mode (one timestep) we can just index the batch item
        if isinstance(unfrozen_idx_or_mask, list):
            attn_output_with_zeros[unfrozen_batch_idx] = attn_output
            attn_weights_with_zeros[unfrozen_batch_idx] = attn_weights
        
        # In SFT fitting mode (multiple timesteps simulateously), we only
        # update each batch item at timesteps where we haven't frozen it
        # Because timesteps of the mask align with the final timesteps of
        # everything else, we have to use padding
        elif isinstance(unfrozen_idx_or_mask, _T):
            # XXX: CHECK MASK AND ATTENTION ALIGNMENT BY TIME
            padding_required = q_len - gen_len
            padded_unfrozen_idx_or_mask = F.pad(
                input = unfrozen_idx_or_mask,
                pad = (padding_required, 0),
                value = True    # Pre-rollout (prompt) residual stream never gets frozen
            )
            attn_output_with_zeros[padded_unfrozen_idx_or_mask] = attn_output[padded_unfrozen_idx_or_mask]
            # attn_weights_with_zeros[padded_unfrozen_idx_or_mask] = attn_weights[padded_unfrozen_idx_or_mask]

            assert not output_attentions,\
                "Currently not supporting output_attentions for LayerFakeAttentionForwardMixin.patched_attention_forward (don't know how to)"

        if not output_attentions:
            attn_weights = None
        return attn_output_with_zeros, attn_weights_with_zeros, past_key_value



