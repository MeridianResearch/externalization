from early_exit.patching.attention_mixins.qwen2 import Qwen2DecoderLayerFakeAttentionForwardMixin
from early_exit.patching.attention_mixins.modular_addition import ModularAdditionLayerFakeAttentionForwardMixin


ATTN_MIXIN_DICT = {
    'Qwen2DecoderLayer': Qwen2DecoderLayerFakeAttentionForwardMixin,
    'Qwen3DecoderLayer': Qwen2DecoderLayerFakeAttentionForwardMixin,
    'TransformerBlock': ModularAdditionLayerFakeAttentionForwardMixin,
}
