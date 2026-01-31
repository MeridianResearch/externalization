from early_exit.patching.model_mixins.qwen2 import Qwen2EarlyExitModelMixin
from early_exit.patching.model_mixins.modular_addition import ModularAdditionEarlyExitModelMixin

MODEL_MIXIN_DICT = {
    'Qwen2ForCausalLM': Qwen2EarlyExitModelMixin,
    'Qwen3ForCausalLM': Qwen2EarlyExitModelMixin,
    'ModularAdditionModel': ModularAdditionEarlyExitModelMixin,
}

