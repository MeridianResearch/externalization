from torch import nn, Tensor as _T
from types import MethodType

from typing import Literal


from early_exit.util import *

from abc import ABC, abstractmethod


possible_early_exit_types = Literal['off', 'free_generate', 'sft_student', 'sft_teacher']



class LayerFakeAttentionForwardMixin(ABC, nn.Module):

    early_exit_mode: possible_early_exit_types = 'off'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.self_attn.base_self_attn_forward = self.self_attn.forward
        self.self_attn.patched_self_attn_forward = MethodType(self.patched_attention_forward, self.self_attn)

    @abstractmethod
    def patched_layer_forward(self, hidden_states: _T, *_, unfrozen_idx_or_mask: List[int], **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def patched_attention_forward(self: nn.Module, hidden_states: _T, *_, unfrozen_idx_or_mask: List[int], **kwargs):
        """
        After early stopping has been triggered, we will still need to go through the layers,
        with the residual stream fixed to the value it had when early exiting was triggered.
        
        This is done to update KV caches, but critically we need the attention output to equal zero
        so that when it is added to the residual stream

        Because of the implementation of this caching, it's easier to copy through the whole function, and
        differentially treat each batch item inside the function itself

        Unfortunately, this implementation is not consistent amongst huggingface layer classes,
        so we need a new one for each model

        XXX: kinda a messy solution because this is a mixin for a layer child, but we are implementing
            a patch for a self_attn method!

            Note that while it is a staticmethod, we still pass self, which will be this model's attention layer

            This is quite cursed...

        XXX: is the attention layer the only thing we want to freeze??
        """
        raise NotImplementedError

