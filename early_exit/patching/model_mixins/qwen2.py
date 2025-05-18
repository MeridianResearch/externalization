from torch import Tensor as _T

from early_exit.patching.model_mixins.base import EarlyExitModelMixin

class Qwen2EarlyExitModelMixin(EarlyExitModelMixin):

    def early_exit_hidden_state_readout(self, hidden_states: _T) -> _T:
        return self.lm_head(hidden_states).log_softmax(-1)

