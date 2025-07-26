from torch import nn, Tensor as _T
import torch.nn.functional as F

from types import MethodType

from typing import Optional, List

from early_exit.util import *
from early_exit.patching.attention_mixins.base import possible_early_exit_types

from abc import ABC, abstractmethod


class EarlyExitModelMixin(ABC, nn.Module):

    early_exit_mode: possible_early_exit_types = 'off'

    base_model_forward: MethodType
    patched_forward_generation: MethodType
    patched_forward_sft_student: MethodType
    patched_forward_sft_teacher: MethodType
    base_model_generate: MethodType
    total_exitable_layers: MethodType
    exitable_layer_idxs: _T

    _early_exit_logs: Optional[List[ExitLogger]] = None
    _early_exit_probabilities: Optional[List[ExitProbabilities]] = None

    @abstractmethod
    def early_exit_hidden_state_readout(self, hidden_states: _T) -> _T:
        """
        hidden_states come in shape     [batch, exitable layers, sequence, hidden shape]
        output shape                    [batch, exitable layers, sequence, vocabulary]
        """
        raise NotImplementedError

    @torch.no_grad()
    def early_exit_target_probs(self, early_output_log_probs: _T, teacher_final_layer_log_probs: _T) -> _T:
        """
        Target (SFT teacher) probability of exiting, which will be sampled from and prescribed for the student

        early_output_log_probs is output of early_exit_hidden_state_readout of shape [batch, exitable layers, sequence, vocabulary]
        teacher_final_layer_log_probs of shape [batch, sequence, vocabulary]

        Recipe:
            1. Get KL divergence between early exit and final layers        -> shaped [batch, exitable layers, sequence]
            2. Scale KL divergencees by KL_FACTOR and pass through sigmoid  -> shaped [batch, exitable layers, sequence]
            3. Apply stickbreaking process along layers, i.e.
                stickbreaking_probs of shape [batch, exitable layers + 1, sequence]
                stickbreaking_probs[b, l, t] = sigmoidkls[b, l, t] * sigmoidkls[b, :l, t].prod(1, keepdims = True)
                s.t. stickbreaking_probs.sum(1) = 1.0 everywhere
        
        Return stickbreaking_probs of shape [batch, sequence, exitable layers + 1]
        """

        KL_FACTOR = 1.0

        # 1. Get KL divergence between early exit and final layers
        teacher_expanded = teacher_final_layer_log_probs.unsqueeze(1).exp()  # [batch, 1, sequence, vocab]
        early_output_probs = early_output_log_probs.exp()

        # Sum over vocab -> [batch, exitable layers, sequence]
        # print('CRUDE KL')
        eps = 1e-16
        kl_div = (teacher_expanded * ((teacher_expanded + eps) / (early_output_probs + eps)).log()).sum(-1)
        # kl_div = - (teacher_expanded * (early_output_probs + eps).log()).sum(-1)

        # 2. Scale KL divergencees by KL_FACTOR and pass through sigmoid (0-1)
        sigmoid_kls = torch.sigmoid(KL_FACTOR * kl_div)  # [batch, exitable layers, sequence]
        sigmoid_kls = 2.0 * sigmoid_kls - 1.0
        sigmoid_kls = 1.0 - sigmoid_kls

        # 3. Apply stickbreaking process along layers
        batch_size, num_layers, seq_len = sigmoid_kls.shape
        stickbreaking_probs = torch.zeros(batch_size, num_layers + 1, seq_len, device = sigmoid_kls.device)

        for l in range(num_layers):
            if l == 0:
                prod_term = torch.ones((batch_size, seq_len), device=sigmoid_kls.device)
            else:
                prod_term = torch.prod(1 - sigmoid_kls[:, :l, :], dim=1)
            stickbreaking_probs[:, l, :] = sigmoid_kls[:, l, :] * prod_term
        
        stickbreaking_probs[:,-1,:] = torch.prod(1 - sigmoid_kls, dim=1)

        assert torch.isclose(stickbreaking_probs.sum(1), torch.tensor(1.0)).all()

        return stickbreaking_probs.permute(0, 2, 1)

    def early_exit_student_probs(self, student_early_exit_logits: _T) -> _T:
        """
        TODO: standardise shapes! This is [B, S, L] when early_exit_target_probs has [B, L, S]
        """
        batch_size, seq_len, num_layers = student_early_exit_logits.shape
        assert num_layers == self.total_exitable_layers
        student_early_exit_probs = student_early_exit_logits.sigmoid()
        stickbreaking_probs = torch.zeros(batch_size, seq_len, num_layers + 1, device = student_early_exit_probs.device)

        for l in range(num_layers):
            if l == 0:
                prod_term = torch.ones((batch_size, seq_len), device=student_early_exit_probs.device)
            else:
                prod_term = torch.prod(1 - student_early_exit_probs[:, :, :l], dim=-1)
            stickbreaking_probs[:, :, l] = student_early_exit_probs[:, :, l] * prod_term
        
        stickbreaking_probs[:,:,-1] = torch.prod(1 - student_early_exit_probs, dim=-1)

        assert torch.isclose(stickbreaking_probs.sum(-1), torch.tensor(1.0)).all()

        return stickbreaking_probs
