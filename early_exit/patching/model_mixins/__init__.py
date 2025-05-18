from early_exit.patching.model_mixins.qwen2 import Qwen2EarlyExitModelMixin

MODEL_MIXIN_DICT = {
    'Qwen2ForCausalLM': Qwen2EarlyExitModelMixin,
}

