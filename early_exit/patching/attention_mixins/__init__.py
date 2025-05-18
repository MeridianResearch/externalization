from early_exit.patching.attention_mixins.qwen2 import Qwen2DecoderLayerFakeAttentionForwardMixin


ATTN_MIXIN_DICT = {
    'Qwen2DecoderLayer': Qwen2DecoderLayerFakeAttentionForwardMixin,
}
