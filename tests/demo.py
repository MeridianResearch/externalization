import torch
import os, certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
import gradio as gr

from shared_utils.load import get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text
from early_exit.util import get_model, load_model_from_wandb
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode

# ---- Config ----
MODEL_PATH = "models/trained_model_v0"
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CONFIG_PATH = "config_deepseek.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WANDB_ARTIFACT = "vkarthik095-university-of-amsterdam/early-exit/early-exit-model-fs5ofmzp:v0"

# ---- Load once at startup (on GPU) ----
tokenizer = get_tokenizer(BASE_MODEL)
config = configs_from_yaml(CONFIG_PATH, tokenizer.eos_token_id)

base_model = get_model(BASE_MODEL, config["model"], DEVICE)
model = replace_attention_layers(base_model, config["lora"], DEVICE)
model = load_model_from_wandb(
    model, model_path=MODEL_PATH, artifact_path=WANDB_ARTIFACT
)

print(f"Model loaded w exitable layers: {getattr(model, 'exitable_layer_idxs', None)}")

# Warmup (optional, helps first-request latency)
set_transformer_early_exit_mode(model, "free_generate")
_ = torch.cuda.synchronize() if DEVICE.startswith("cuda") else None

def infer(
    user_prompt,
    system_prompt,
    prefiller,
    max_new_tokens,
    temperature,
    top_p,
    do_sample,
    mode
):
    if not user_prompt or user_prompt.strip() == "":
        return "Please enter a prompt."

    # Copy and tweak generation config per request
    gen_cfg = dict(config["generation"])
    gen_cfg["max_new_tokens"] = int(max_new_tokens)
    gen_cfg["temperature"] = float(temperature)
    gen_cfg["top_p"] = float(top_p)
    gen_cfg["do_sample"] = bool(do_sample)

    # Early-exit / generation mode
    set_transformer_early_exit_mode(model, mode)

    with torch.no_grad():
        response, exit_info = generate_text(
            model=model,
            prompt=user_prompt,
            system_prompt=system_prompt or "",
            prefiller=prefiller or "",
            tokenizer=tokenizer,
            generation_config=gen_cfg,
            device=DEVICE,
        )

    # Optionally expose some exit info if you want
    # return f"{response}\n\n---\nExit info: {exit_info}"
    return response

# ---- Gradio UI ----
with gr.Blocks(title="Early-Exit LLM Demo") as demo:
    gr.Markdown("# Early-Exit LLM Demo")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Ask me something…", lines=4)
            system_prompt = gr.Textbox(
                label="System Prompt",
                value="You are a helpful math tutor.",
                lines=2
            )
            prefiller = gr.Textbox(label="Prefiller (optional)", value="", lines=1)

            with gr.Accordion("Generation settings", open=False):
                max_new_tokens = gr.Slider(1, 1024, value=400, step=1, label="max_new_tokens")
                temperature = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="temperature")
                top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="top_p")
                do_sample = gr.Checkbox(value=False, label="do_sample")

                mode = gr.Radio(
                    choices=["free_generate", "off"],
                    value="free_generate",
                    label="Transformer early-exit mode"
                )

            run_btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="Response", lines=18)

    run_btn.click(
        infer,
        inputs=[prompt, system_prompt, prefiller, max_new_tokens, temperature, top_p, do_sample, mode],
        outputs=[output],
        queue=True,  # enables request queueing for concurrent users
        api_name="generate"
    )

# For remote servers, set server_name to "0.0.0.0" and choose a port.
import random
port = random.randint(1000, 9999)
demo.queue(max_size=32).launch(server_name="0.0.0.0", server_port=port, share=True)
