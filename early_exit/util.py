import math

import torch
from torch import Tensor as _T

from dataclasses import dataclass, field
from typing import Optional, List




def module_name_is_layer_base(name: str):
    split_by_dots = name.split('.')
    if len(split_by_dots) < 2:
        return False
    is_layer = (split_by_dots[-2] == 'layers')
    if is_layer:
        layer_idx = int(split_by_dots[-1])
        return layer_idx % 5 == 0
    else:
        return False


@dataclass
class ExitLogger:
    """
    Message carrier for exiting early, and the logits associated with an early exit
    """
    batch_size: int
    readout_layer_idx: Optional[List[int | float]] = None

    def __post_init__(self) -> None:
        self.readout_layer_idx = [torch.inf for _ in range(self.batch_size)]
        self.frozen_batch_items = []
    
    def freeze_batch_item(self, batch_idx: List[int], layer_idx: int) -> None:
        for bi in batch_idx:
            assert bi not in self.frozen_batch_items
            assert math.isinf(self.readout_layer_idx[bi])
            self.frozen_batch_items.append(bi)
            self.readout_layer_idx[bi] = layer_idx
        self.frozen_batch_items = sorted(self.frozen_batch_items)

    @property
    def unfrozen_batch_items(self) -> List[int]:
        return [i for i in range(self.batch_size) if i not in self.frozen_batch_items]


@dataclass
class ExitProbabilities:
    """
    This is for the teacher model in SFT

    Keeps a track of early hidden states, which can later be converted to logits *as if* we had exited early
    """
    batch_size: int
    hidden_size: int
    cache: List[_T] = field(default_factory=lambda: [])
    layers_idxs: List[int] = field(default_factory=lambda: [])
    exitable_layers_idxs: List[int] = field(default_factory=lambda: [])

    def log_residual_stream(self, hidden_states: _T, layer_idx: int, exitable_layer_idx: int) -> None:
        """
        Expecting hidden_states of shape [batch, length = 1, hidden size]
        
        Expecting length of 1 because this is only for 'freely generated' hidden states
        """
        assert tuple(hidden_states.shape) == (self.batch_size, 1, self.hidden_size)
        assert exitable_layer_idx == len(self.exitable_layers_idxs)
        
        self.cache.append(hidden_states)
        self.layers_idxs.append(layer_idx)
        self.exitable_layers_idxs.append(exitable_layer_idx)






@dataclass
class ExitPrescription:
    """
    This is for the student model in SFT

    Keeps a track of the probability of exiting, but only exit at the preprescribed time!
    """
    prescribed_exit_layer_idxs: _T
    total_exitable_layers: int
    device: str
    
    batch_size: Optional[int] = None
    generation_length: Optional[int] = None
    collected_exit_logits: Optional[_T] = None

    def __post_init__(self,):
        self.batch_size, self.generation_length = self.prescribed_exit_layer_idxs.shape
        self.collected_exit_logits = torch.nan * torch.ones(
            self.batch_size, self.generation_length, self.total_exitable_layers, device = self.device
        )

    def get_unfrozen_mask(self, current_layer_idx: int) -> _T:
        """
        Layer is frozen if prescribed_exit_layer_idxs (where the early exit happened during generation)
            is XXX: >= the current layer
        """
        return self.prescribed_exit_layer_idxs >= current_layer_idx



