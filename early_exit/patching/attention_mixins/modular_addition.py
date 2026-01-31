import torch
from torch.nn import functional as F
from torch import Tensor as _T

from typing import List, Optional, Tuple

from early_exit.patching.attention_mixins.base import LayerFakeAttentionForwardMixin


class ModularAdditionLayerFakeAttentionForwardMixin(LayerFakeAttentionForwardMixin):

    def patched_layer_forward(
        self,
        hidden_states: _T,
        attn_mask: Optional[_T] = None,
        key_padding_mask: Optional[_T] = None,
        unfrozen_idx_or_mask: Optional[List[int] | _T] = None,
        **kwargs,
    ) -> Tuple[_T]:

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
                value=True,
            ).to(hidden_states.device)
        elif unfrozen_idx_or_mask is None:
            unfrozen_elements = torch.ones(bsz, dtype=torch.bool, device=hidden_states.device)

        # Call TransformerBlock.forward
        new_hidden_states = super().forward(
            hidden_states,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )

        if unfrozen_elements.ndim == 1:
            mask = unfrozen_elements.view(bsz, 1, 1)
        else:
            mask = unfrozen_elements.unsqueeze(-1)

        final_hidden_states = torch.where(mask, new_hidden_states, original_hidden_states)

        assert (original_hidden_states == final_hidden_states)[~unfrozen_elements].all()

        return (final_hidden_states,)

    @staticmethod
    def patched_attention_forward(self, hidden_states, *_, unfrozen_idx_or_mask=None, **kwargs):
        # Not used — the main version calls super().forward() which handles attention internally
        raise NotImplementedError("patched_attention_forward not needed for ModularAddition")
