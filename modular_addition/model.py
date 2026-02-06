"""GPT2-based model for modular addition."""

from transformers import GPT2Config, GPT2LMHeadModel

from modular_addition.tokenizer import ModularAdditionTokenizer


def create_model(
    tokenizer: ModularAdditionTokenizer,
    n_positions: int = 64,
    n_embd: int = 256,
    n_layer: int = 6,
    n_head: int = 8,
    n_inner: int | None = None,
    resid_pdrop: float = 0.0,
    embd_pdrop: float = 0.0,
    attn_pdrop: float = 0.0,
    **kwargs,
) -> GPT2LMHeadModel:
    """Create a GPT2LMHeadModel configured for modular addition.

    Args:
        tokenizer: ModularAdditionTokenizer instance
        n_positions: Max sequence length (default: 64)
        n_embd: Hidden size (default: 256)
        n_layer: Number of layers (default: 6)
        n_head: Number of attention heads (default: 8)
        n_inner: FFN inner dim (default: 4*n_embd)
        resid_pdrop: Residual dropout (default: 0.0)
        embd_pdrop: Embedding dropout (default: 0.0)
        attn_pdrop: Attention dropout (default: 0.0)
        **kwargs: Additional GPT2Config arguments
    """
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=n_positions,
        n_ctx=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_inner,
        resid_pdrop=resid_pdrop,
        embd_pdrop=embd_pdrop,
        attn_pdrop=attn_pdrop,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        loss_type="ForCausalLMLoss",
        **kwargs,
    )
    return GPT2LMHeadModel(config)


def create_model_from_config(tokenizer: ModularAdditionTokenizer, model_config: dict) -> GPT2LMHeadModel:
    """Create model from a config dict (e.g., loaded from YAML).

    Args:
        tokenizer: ModularAdditionTokenizer instance
        model_config: Dict with model parameters (n_positions, n_embd, n_layer, etc.)
    """
    return create_model(tokenizer, **model_config)
