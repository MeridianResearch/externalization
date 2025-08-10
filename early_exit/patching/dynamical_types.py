from torch import nn, Tensor as _T

from typing import Type

from transformers import AutoConfig, AutoModelForCausalLM

from transformers.utils import logging
logger = logging.get_logger(__name__)


from early_exit.util import *

from early_exit.patching.attention_mixins.base import LayerFakeAttentionForwardMixin
from early_exit.patching.attention_mixins import ATTN_MIXIN_DICT

from early_exit.patching.model_mixins.base import EarlyExitModelMixin
from early_exit.patching.model_mixins import MODEL_MIXIN_DICT


def generate_layer_type_with_early_exit_decision_head(base_type: Type[AutoModelForCausalLM]):
    """
    Different Hugging Face transformer architectures have different types for their (macro-)layers,
        and these different types also have their own config types.
    
    This function generates a new class which dynamically inherits from the layer classes,
        and also types the config which it expects.
    """
    mixin_type: Type[LayerFakeAttentionForwardMixin] = ATTN_MIXIN_DICT[base_type.__name__]

    class DynamicallyTypedLayerWithExit(mixin_type, base_type):

        exit_state: ExitLogger | ExitPrescription | ExitProbabilities
        
        def __init__(self, config: AutoConfig, exitable_layer_idx: int, **kwargs) -> None:
            super().__init__(config, **kwargs)

            self.config = config
            self.layer_idx = kwargs['layer_idx']
            dtype = config.torch_dtype
            self.exitable_layer_idx = exitable_layer_idx

            self.vocab_size = config.vocab_size

            self.early_exit_decision_weights = nn.Linear(config.hidden_size, 1, dtype=dtype)
            # self.early_readout_weights = nn.Linear(config.hidden_size, self.vocab_size, bias=False, dtype=dtype)

        def update_exit_decision_during_generation(self, hidden_states: _T):
            """
            hidden_states comes in shape [B, 1, D]
            
            Update self.exit_state based on XXX: deterministic rule
            """
            B, L, D = hidden_states.shape
            assert L == 1
            assert self.early_exit_mode == 'free_generate', \
                f"Cannot use update_exit_decision_during_generation when self.early_exit_mode = {self.early_exit_mode}"

            unfrozen_batch_items = torch.tensor(self.exit_state.unfrozen_batch_items, device = hidden_states.device)   # [B_unfrozen]

            if len(unfrozen_batch_items):
                unfrozen_hidden_states = hidden_states[unfrozen_batch_items]       # [B_unfrozen, L, D]

                readout_logits: _T = self.early_exit_decision_weights(unfrozen_hidden_states).squeeze(-1).squeeze(-1)       # [B_unfrozen]

                #raise NotImplementedError(header = 'make this stochastic!')
                exit_probs = torch.sigmoid(readout_logits)  # [B_unfrozen]
                min_threshold = 0.1
                above_threshold_mask = exit_probs >= min_threshold
                exit_decisions = torch.zeros_like(exit_probs, dtype=torch.bool)
                if above_threshold_mask.any():
                    exit_decisions[above_threshold_mask] = torch.bernoulli(exit_probs[above_threshold_mask]).bool()
                #exit_decisions = torch.bernoulli(exit_probs).bool()  #stochastic sample
                items_to_early_exit = unfrozen_batch_items[exit_decisions]

                #early_exit_decision_weights = (readout_logits >= 1.0)      # [B_unfrozen]
                #items_to_early_exit = unfrozen_batch_items[early_exit_decision_weights] # [B_to_exit]
                
                self.exit_state.freeze_batch_item(
                    batch_idx = items_to_early_exit.tolist(),
                    layer_idx = self.layer_idx
                )

        def forward(self, hidden_states: _T, **kwargs):
            """
            hidden_states comes in shape [B, L, D]

            - Assume that if L > 1, we are cache-building in the prompt forwardpass, so neve exit anyway
                XXX: this is quite unsafe though! maybe check cache or something similar?
            """
            B, L, D = hidden_states.shape

            if self.early_exit_mode == 'free_generate':
                if L == 1:
                    self.update_exit_decision_during_generation(hidden_states)
                    return self.patched_layer_forward(
                        hidden_states = hidden_states,
                        **kwargs,
                        unfrozen_idx_or_mask = self.exit_state.unfrozen_batch_items,
                    )

                else:
                    return self.patched_layer_forward(
                        hidden_states = hidden_states, **kwargs
                    )

            elif self.early_exit_mode == 'sft_student':
                
                # XXX: CHECK LOGIT COLLECTION ALIGNMENT BY TIME
                readout_logits: _T = self.early_exit_decision_weights(hidden_states[:,-self.exit_state.generation_length:]).squeeze(-1)       # [B, S]
                assert self.exit_state.collected_exit_logits[...,self.exitable_layer_idx].isnan().all()
                self.exit_state.collected_exit_logits[...,self.exitable_layer_idx] = readout_logits

                unfrozen_mask = self.exit_state.get_unfrozen_mask(self.layer_idx)

                return self.patched_layer_forward(
                    hidden_states = hidden_states,
                    **kwargs,
                    unfrozen_idx_or_mask = unfrozen_mask
                )

            elif self.early_exit_mode == 'sft_teacher':
                if L == 1:
                    # Normal forward pass - everything is normal
                    # Collect hidden state before this layer, because this is where we are making the decision to exit
                    self.exit_state.log_residual_stream(
                        hidden_states = hidden_states,
                        layer_idx = self.layer_idx,
                        exitable_layer_idx = self.exitable_layer_idx,
                    )
                    layer_output = super().forward(hidden_states = hidden_states, **kwargs)
                    return layer_output

                else:
                    # Normal forward pass - everything is normal
                    return super().forward(hidden_states = hidden_states, **kwargs)
            
            elif self.early_exit_mode == 'off':
                return super().forward(hidden_states = hidden_states, **kwargs)

            else:
                raise AttributeError(self.early_exit_mode)

    return DynamicallyTypedLayerWithExit


def generate_layer_type_without_early_exit_decision_head(base_type: Type[AutoModelForCausalLM]):
    """
    Different Hugging Face transformer architectures have different types for their (macro-)layers,
        and these different types also have their own config types.
    
    This function generates a new class which dynamically inherits from the layer classes,
        but WITHOUT early exit decision heads - just uses patched_layer_forward().
    """
    mixin_type: Type[LayerFakeAttentionForwardMixin] = ATTN_MIXIN_DICT[base_type.__name__]

    class DynamicallyTypedLayerWithoutExit(mixin_type, base_type):

        exit_state: ExitLogger | ExitPrescription | ExitProbabilities
        
        def __init__(self, config: AutoConfig, exitable_layer_idx: int, **kwargs) -> None:
            super().__init__(config, **kwargs)

            self.config = config
            self.layer_idx = kwargs['layer_idx']
            self.exitable_layer_idx = exitable_layer_idx
            self.vocab_size = config.vocab_size

            # NOTE: No early_exit_decision_weights - removed early exit decision head

        def forward(self, hidden_states: _T, **kwargs):
            """
            hidden_states comes in shape [B, L, D]

            - Assume that if L > 1, we are cache-building in the prompt forwardpass, so never exit anyway
                XXX: this is quite unsafe though! maybe check cache or something similar?
            """
            B, L, D = hidden_states.shape

            if self.early_exit_mode == 'free_generate':
                if L == 1:
                    # No decision making - just use existing unfrozen items
                    return self.patched_layer_forward(
                        hidden_states = hidden_states,
                        **kwargs,
                        unfrozen_idx_or_mask = self.exit_state.unfrozen_batch_items,
                    )
                else:
                    return self.patched_layer_forward(
                        hidden_states = hidden_states, **kwargs
                    )

            elif self.early_exit_mode == 'sft_student':
                # Get unfrozen mask and call patched_layer_forward
                unfrozen_mask = self.exit_state.get_unfrozen_mask(self.layer_idx)

                return self.patched_layer_forward(
                    hidden_states = hidden_states,
                    **kwargs,
                    unfrozen_idx_or_mask = unfrozen_mask
                )

            elif self.early_exit_mode == 'sft_teacher':
                # Normal forward pass - no logging or decision making
                return super().forward(hidden_states = hidden_states, **kwargs)
            
            elif self.early_exit_mode == 'off':
                return super().forward(hidden_states = hidden_states, **kwargs)

            else:
                raise AttributeError(self.early_exit_mode)

    return DynamicallyTypedLayerWithoutExit




def generate_model_type_with_early_exit_readout_head(base_type: Type[nn.Module]):
    """
    Same idea here but for the whole model, 
    """

    mixin_type: Type[EarlyExitModelMixin] = MODEL_MIXIN_DICT[base_type.__name__]

    class DynamicallyTypedModelWithReadout(mixin_type, base_type):
        """
        Just a placeholder for now so that we can load the pretrained weights cleanly
        """
        
    return DynamicallyTypedModelWithReadout
