"""Working example of a real-time Gradio demo for early-exit token visualization.

Note: kept as a reference and not used for now.
"""

import os
import sys
import io
import contextlib
from typing import Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
import gradio as gr


import sys
sys.path.append("../")
from shared_utils.load import get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text
from early_exit.util import get_model, load_model_from_wandb
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode
from tests.early_exit_teacher.visualization import safe_decode_tokens, visualize_tokens_by_exit_layer
from IPython.display import display, HTML




# Make repository root importable
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Imports matching tests/free_generate_simple.py
from shared_utils.load import get_tokenizer, configs_from_yaml  # type: ignore
from shared_utils.generate import generate_text  # type: ignore
from early_exit.util import get_model, load_model_from_wandb  # type: ignore
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode  # type: ignore

# Visualization utilities
from tests.early_exit_teacher.visualization import (  # type: ignore
    visualize_tokens_by_exit_layer,
    safe_decode_tokens,
)


# Optional: keep the original text-to-image utility (not used in the demo now)
def render_text_to_image(
    text: str,
    canvas_size: Tuple[int, int] = (768, 512),
    background_color: str = "white",
    text_color: str = "black",
    font_size: int = 32,
) -> Image.Image:
    width, height = canvas_size
    image = Image.new("RGB", (width, height), color=background_color)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    max_width = width - 40
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        test_line = (" ".join(current_line + [word])).strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]
        if line_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))

    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1])
    total_text_height = sum(line_heights) + max(0, len(lines) - 1) * 6
    y = max(20, (height - total_text_height) // 2)

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        x = max(20, (width - line_width) // 2)
        draw.text((x, y), line, font=font, fill=text_color)
        y += line_height + 6

    return image


# Global singleton state for model/tokenizer/config
_MODEL = None
_TOKENIZER = None
_CONFIG = None


def _init_model_once():
    global _MODEL, _TOKENIZER, _CONFIG
    if _MODEL is not None and _TOKENIZER is not None and _CONFIG is not None:
        return

    # Configuration matching tests/free_generate_simple.py
    config_path = os.path.join(PROJECT_ROOT, "config_deepseek.yaml")
    device = "cpu"
    model_path = "models/trained_model_v0"
    base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    tokenizer = get_tokenizer(base_model_name)
    config = configs_from_yaml(config_path, tokenizer.eos_token_id)

    base_model = get_model(base_model_name, config['model'], device)
    model = replace_attention_layers(base_model, config['lora'], device)

    # Download artifact and load weights
    model = load_model_from_wandb(
        model,
        model_path=model_path,
        artifact_path='vkarthik095-university-of-amsterdam/early-exit/early-exit-model-fs5ofmzp:v0'
    )

    # Set early-exit free generate mode
    set_transformer_early_exit_mode(model, 'free_generate')

    _MODEL = model
    _TOKENIZER = tokenizer
    _CONFIG = config


def prompt_to_exit_layer_html(prompt: str) -> Tuple[str, str]:
    """Generate text with early-exit and return (HTML visualization, captured log)."""
    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        _init_model_once()
        assert _MODEL is not None and _TOKENIZER is not None and _CONFIG is not None

        # Mirror informative prints from free_generate_simple.py
        print(f"Model loaded w exitable layers: {_MODEL.exitable_layer_idxs}")
        print("Generating...")

        system_prompt = "You are a helpful math tutor."
        prefiller = ""

        # Keep generation budget small by default
        gen_cfg = dict(_CONFIG['generation'])
        gen_cfg['max_new_tokens'] = gen_cfg.get('max_new_tokens', 5)

        with torch.no_grad():
            free_generate_response, exit_info = generate_text(
                model=_MODEL,
                prompt=prompt,
                system_prompt=system_prompt,
                prefiller=prefiller,
                tokenizer=_TOKENIZER,
                generation_config=gen_cfg,
                device="cpu",
            )

        print(f"Free Generate Response: {free_generate_response}")

    # Build visualization outside of redirect to avoid cluttering log
    assert _MODEL is not None and _TOKENIZER is not None
    gen_len = exit_info[1][0].shape[-1]
    token_ids = exit_info[0][0, -gen_len:]
    tokens = safe_decode_tokens(_TOKENIZER, token_ids)
    layers = [27 if item == torch.inf or item == -1 else int(item) for item in exit_info[1][0]]
    exitable_layers = [int(item) for item in _MODEL.exitable_layer_idxs[:-1]]

    html_vis = visualize_tokens_by_exit_layer(
        tokens,
        layers,
        exitable_layers,
        title="Committed Early Exit Token Generation",
    )
    logs = log_buffer.getvalue()
    return html_vis, logs


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Early-Exit Token Visualization") as demo:
        gr.Markdown("## Prompt → Early-Exit Token Visualization")
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Type a prompt (e.g., What is 17 x 19?)",
                    lines=4,
                )
                submit = gr.Button("Generate & Visualize")
            with gr.Column():
                html_output = gr.HTML(label="Visualization")
                log_output = gr.Textbox(label="Log", lines=10)

        submit.click(prompt_to_exit_layer_html, inputs=text_input, outputs=[html_output, log_output])
        text_input.submit(prompt_to_exit_layer_html, inputs=text_input, outputs=[html_output, log_output])
    return demo


if __name__ == "__main__":
    build_demo().launch()


