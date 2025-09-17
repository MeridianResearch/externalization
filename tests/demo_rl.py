import torch
import os, certifi

os.environ["SSL_CERT_FILE"] = certifi.where()
import gradio as gr

from shared_utils.load import get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text
from early_exit.util import get_model, load_model, load_model_from_wandb
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode
torch.set_grad_enabled(False)
import argparse

import re  # add this

# ----------------- helpers -----------------
def split_think_and_response(text: str):
    """
    Extract <think>...</think> as CoT, and strip it from the visible response.
    Also strips BOS/EOS wrappers used by some models.
    """
    if not isinstance(text, str):
        return "", ""

    # Strip wrapper tokens like <｜begin▁of▁sentence｜> ... <｜end▁of▁sentence｜>
    text = re.sub(r"<\u2502?begin.*?sentence.*?\u2502?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\u2502?end.*?sentence.*?\u2502?>", "", text, flags=re.IGNORECASE)

    thinks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    cot = "\n\n---\n\n".join(t.strip() for t in thinks) if thinks else ""

    # Remove CoT blocks
    response = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    # If your generator sometimes adds `<｜Assistant｜>` delimiters, keep the tail
    if "<｜Assistant｜>" in response:
        parts = response.split("<｜Assistant｜>")
        response = parts[-1].strip()

    return cot, response

def build_model(base_model_name: str, config: dict, device: str, adapter_path: str | None):
    """
    base-only if adapter_path is None; otherwise apply LoRA and load weights from adapter_path.
    """
    base = get_model(base_model_name, config["model"], device)
    if adapter_path is None:
        # Pure base, no early-exit heads installed
        return base
    # Apply adapter-capable layers and load adapter weights
    model = replace_attention_layers(base, config["lora"], device)
    model = load_model(model, model_path=adapter_path)
    set_transformer_early_exit_mode(model, 'free_generate')
    return model

def generate_once(model, tokenizer, config, device, prompt, system_prompt, prefiller,
                  max_new_tokens, temperature, top_p, do_sample):
    """
    Standardized generation call w/ per-request config.
    """
    gen_cfg = dict(config["generation"])
    gen_cfg["max_new_tokens"] = int(max_new_tokens)
    gen_cfg["temperature"] = float(temperature)
    gen_cfg["top_p"] = float(top_p)
    gen_cfg["do_sample"] = bool(do_sample)

    text, exit_info = generate_text(
        model=model,
        prompt=prompt,
        system_prompt=system_prompt or "",
        prefiller=prefiller or "",
        tokenizer=tokenizer,
        generation_config=gen_cfg,
        device=device,
    )
    cot, resp = split_think_and_response(text)
    return cot, resp

parser = argparse.ArgumentParser()
parser.add_argument("--sft_model", type=str, default="models/gsm8k_old_school_1", help="Path to SFT model directory")
parser.add_argument("--rl_model", type=str, default="models/rl-run-qi7wr8f6/epoch-695", help="Path to RL model directory")
parser.add_argument("--base_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
parser.add_argument("--config_path", type=str, default="config_greedy.yaml")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

BASE_MODEL = args.base_model
CONFIG_PATH = args.config_path
DEVICE = args.device


tokenizer = get_tokenizer(BASE_MODEL)
config = configs_from_yaml(CONFIG_PATH, tokenizer.eos_token_id)

# ----------------- load three models -----------------
# Base: no adapters
base_model = build_model(BASE_MODEL, config, DEVICE, adapter_path=None)
# SFT: if path provided
sft_model = build_model(BASE_MODEL, config, DEVICE, adapter_path=args.sft_model) if args.sft_model else None
# RL : if path provided
rl_model  = build_model(BASE_MODEL, config, DEVICE, adapter_path=args.rl_model) if args.rl_model else None

# Small printouts
print(f"[Base] loaded: {BASE_MODEL}")
if sft_model: print(f"[SFT ] loaded from: {args.sft_model}")
else:         print("[SFT ] not provided")
if rl_model:  print(f"[RL  ] loaded from: {args.rl_model}")
else:         print("[RL  ] not provided")

# ----------------- Gradio app -----------------
def infer_all(
    user_prompt,
    system_prompt,
    prefiller,
    max_new_tokens,
    temperature,
    top_p,
    do_sample
):
    if not user_prompt or user_prompt.strip() == "":
        # Return 6 boxes (CoT base/sft/rl, Resp base/sft/rl)
        return ("Please enter a prompt.", "", "",
                "", "", "")

    # BASE (early-exit off by default)
    base_cot, base_resp = generate_once(
        base_model, tokenizer, config, DEVICE,
        user_prompt, system_prompt, prefiller,
        max_new_tokens, temperature, top_p, do_sample
    )

    # SFT (if missing, leave blank)
    if sft_model is not None:
        sft_cot, sft_resp = generate_once(
            sft_model, tokenizer, config, DEVICE,
            user_prompt, system_prompt, prefiller,
            max_new_tokens, temperature, top_p, do_sample
        )
    else:
        sft_cot, sft_resp = ("(no SFT model loaded)", "")

    # RL (if missing, leave blank)
    if rl_model is not None:
        rl_cot, rl_resp = generate_once(
            rl_model, tokenizer, config, DEVICE,
            user_prompt, system_prompt, prefiller,
            max_new_tokens, temperature, top_p, do_sample
        )
    else:
        rl_cot, rl_resp = ("(no RL model loaded)", "")
    
    print("\n\nGeneration complete!\n\n")
    return base_cot, sft_cot, rl_cot, base_resp, sft_resp, rl_resp


with gr.Blocks(title="Early-Exit LLM Demo: Base vs SFT vs RL") as demo:
    header_md = [
        "# Early-Exit LLM Demo - Base vs SFT vs RL",
        f"- **Base model:** `{BASE_MODEL}`",
        f"- **Config:** `{CONFIG_PATH}`",
        f"- **SFT path:** `{args.sft_model or 'None'}`",
        f"- **RL path:** `{args.rl_model or 'None'}`",
    ]
    gr.Markdown("\n".join(header_md))

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                value="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
                lines=4
            )
            system_prompt = gr.Textbox(label="System Prompt", 
                                       value="I am going to give you a math word problem. Solve it step by step, showing your reasoning. After your work, provide your final numerical answer.",
                                       lines=2)
            prefiller = gr.Textbox(label="Prefiller (optional)", value="", lines=1)

            with gr.Accordion("Generation settings", open=False):
                max_new_tokens = gr.Slider(1, 1024, value=1024, step=1, label="max_new_tokens")
                temperature = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="temperature")
                top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="top_p")
                do_sample = gr.Checkbox(value=False, label="do_sample")


            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Chain of thought")
            with gr.Row():
                base_cot  = gr.Textbox(label="Base - CoT", lines=8)
                sft_cot   = gr.Textbox(label="SFT - CoT",  lines=8)
                rl_cot    = gr.Textbox(label="RL - CoT",   lines=8)

            gr.Markdown("### Assistant")
            with gr.Row():
                base_resp = gr.Textbox(label="Base - Assistant", lines=10)
                sft_resp  = gr.Textbox(label="SFT - Assistant",  lines=10)
                rl_resp   = gr.Textbox(label="RL - Assistant",   lines=10)

    run_btn.click(
        infer_all,
        inputs=[prompt, system_prompt, prefiller, max_new_tokens, temperature, top_p, do_sample],
        outputs=[base_cot, sft_cot, rl_cot, base_resp, sft_resp, rl_resp],
        queue=True,
        api_name="generate_all"
    )

# Launch
import random
port = random.randint(1000, 9999)
demo.queue(max_size=32).launch(server_name="0.0.0.0", server_port=port, share=True)