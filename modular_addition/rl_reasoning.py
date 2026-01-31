"""Stage 1c: RL training for correctness on modular addition."""

import os
import random
import torch
from torch.utils.data import DataLoader

from modular_addition.config import Config
from modular_addition.tokenizer import ModularAdditionTokenizer
from modular_addition.model import ModularAdditionModel
from modular_addition.data import ModularAdditionDataset


def extract_answer(token_ids: list[int], tokenizer: ModularAdditionTokenizer) -> int | None:
    """Extract the final answer from generated tokens.

    Looks for the pattern: </think> = answer or just = answer (no think tags).
    Returns the number token right after the last = token before <eos>.
    """
    eq_id = tokenizer.eq_token_id
    eos_id = tokenizer.eos_token_id

    # Find last = before eos (or end)
    eos_pos = len(token_ids)
    for i, t in enumerate(token_ids):
        if t == eos_id:
            eos_pos = i
            break

    # Walk backwards from eos to find the last =
    for i in range(eos_pos - 1, -1, -1):
        if token_ids[i] == eq_id and i + 1 < eos_pos:
            ans = token_ids[i + 1]
            if 0 <= ans < tokenizer.p:
                return ans
    return None


def compute_log_probs(model, input_ids, attention_mask):
    """Compute per-token log probabilities for generated tokens."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out["logits"][:, :-1]  # [B, T-1, V]
    targets = input_ids[:, 1:]  # [B, T-1]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    return token_log_probs


def main(config: Config | None = None):
    if config is None:
        config = Config()

    torch.manual_seed(config.seed)
    device = config.device
    cfg = config.rl_reasoning

    tokenizer = ModularAdditionTokenizer(p=config.p)

    # Load SFT checkpoint
    checkpoint = torch.load(cfg.sft_checkpoint, map_location=device)
    model_config = checkpoint["config"]
    model = ModularAdditionModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded SFT model from {cfg.sft_checkpoint}")

    # Reference model (frozen)
    ref_checkpoint = torch.load(cfg.sft_checkpoint, map_location=device)
    ref_model = ModularAdditionModel(model_config).to(device)
    ref_model.load_state_dict(ref_checkpoint["model_state_dict"])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # RL dataset (prompts only)
    dataset = ModularAdditionDataset(
        tokenizer, mode="rl", size=cfg.dataset_size,
        num_operands_range=cfg.num_operands_range,
        max_seq_len=config.model.max_seq_len, seed=config.seed + 20,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        total_reward = 0.0
        total_loss = 0.0
        n_steps = 0

        for batch in dataloader:
            prompt_ids = batch["input_ids"].to(device)
            prompt_mask = batch["attention_mask"].to(device)
            B = prompt_ids.shape[0]

            # Find actual prompt length (non-pad)
            prompt_lengths = prompt_mask.sum(dim=1)  # [B]

            all_rewards = []
            all_log_probs = []
            all_masks = []

            for _ in range(cfg.k):
                # Generate completions
                with torch.no_grad():
                    generated = model.generate(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        max_new_tokens=cfg.max_new_tokens,
                        do_sample=True,
                        temperature=1.0,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                # Compute rewards
                rewards = []
                for i in range(B):
                    gen_tokens = generated[i].tolist()
                    operands = dataset.samples[batch["input_ids"][i].tolist().index(tokenizer.bos_token_id):]
                    # Ground truth
                    orig_operands = []
                    ids = prompt_ids[i].tolist()
                    for t in ids:
                        if 0 <= t < tokenizer.p:
                            orig_operands.append(t)
                    correct_answer = sum(orig_operands) % config.p
                    pred_answer = extract_answer(gen_tokens, tokenizer)
                    rewards.append(1.0 if pred_answer == correct_answer else 0.0)
                rewards = torch.tensor(rewards, device=device)
                all_rewards.append(rewards)

                # Compute log probs for the generated sequence
                gen_mask = (generated != tokenizer.pad_token_id).long()
                log_probs = compute_log_probs(model, generated, gen_mask)

                # Mask to only generation tokens
                gen_only_mask = torch.zeros_like(log_probs)
                for i in range(B):
                    pl = prompt_lengths[i].item()
                    gen_len = gen_mask[i].sum().item()
                    if pl < log_probs.shape[1]:
                        gen_only_mask[i, pl - 1 : gen_len - 1] = 1.0

                all_log_probs.append(log_probs)
                all_masks.append(gen_only_mask)

            # Stack across k rollouts: [k, B]
            rewards_stack = torch.stack(all_rewards, dim=0)  # [k, B]
            # RLOO baseline: mean of other rollouts
            baseline = (rewards_stack.sum(dim=0, keepdim=True) - rewards_stack) / (cfg.k - 1)
            advantages = rewards_stack - baseline  # [k, B]

            # Compute loss
            loss = torch.tensor(0.0, device=device)
            for ki in range(cfg.k):
                masked_lp = all_log_probs[ki] * all_masks[ki]
                seq_lp = masked_lp.sum(dim=1)  # [B]
                loss = loss - (advantages[ki].detach() * seq_lp).mean()
            loss = loss / cfg.k

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward += rewards_stack.mean().item()
            total_loss += loss.item()
            n_steps += 1

        avg_reward = total_reward / n_steps if n_steps > 0 else 0.0
        avg_loss = total_loss / n_steps if n_steps > 0 else 0.0
        print(f"Epoch {epoch+1}/{cfg.epochs} | Avg Reward: {avg_reward:.4f} | Loss: {avg_loss:.4f}")

    # Save
    os.makedirs(cfg.save_path, exist_ok=True)
    save_path = os.path.join(cfg.save_path, "model.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": model_config}, save_path)
    print(f"Saved to {save_path}")
    return model


if __name__ == "__main__":
    main()
