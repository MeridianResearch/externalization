"""Pretraining script for modular addition."""

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from modular_addition import (
    ModularAdditionTokenizer,
    create_model_from_config,
    PretrainDataset,
    DataCollator,
)
from modular_addition.config import PretrainConfig, ModelConfig, DataConfig


def load_config(config_path: str) -> PretrainConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Build nested dataclasses from flat YAML
    model_cfg = ModelConfig(**raw.pop("model", {}))
    data_raw = raw.pop("data", {})
    if "n_operands" in data_raw:
        data_raw["n_operands"] = tuple(data_raw["n_operands"])
    data_cfg = DataConfig(**data_raw)

    return PretrainConfig(model=model_cfg, data=data_cfg, **raw)


@torch.no_grad()
def evaluate(model, loader, tokenizer, device):
    """Compute loss and final-answer accuracy over a dataloader."""
    model.eval()
    total_loss = 0
    n_batches = 0
    total_correct = 0
    total_count = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()
        n_batches += 1

        labels = batch["labels"]
        preds = outputs.logits.argmax(dim=-1)

        eos_mask = labels == tokenizer.eos_token_id
        has_eos = eos_mask.any(dim=-1)
        eos_pos = eos_mask.int().argmax(dim=-1)

        ans_pos = (eos_pos - 1).clamp(min=0)
        pred_pos = (eos_pos - 2).clamp(min=0)

        pred_ans = preds.gather(1, pred_pos.unsqueeze(1)).squeeze(1)
        true_ans = labels.gather(1, ans_pos.unsqueeze(1)).squeeze(1)

        correct = (pred_ans == true_ans) & has_eos
        total_correct += correct.sum().item()
        total_count += has_eos.sum().item()

    return {
        "loss": total_loss / max(n_batches, 1),
        "acc": total_correct / max(total_count, 1),
    }


def train(cfg: PretrainConfig):
    output_dir = Path(cfg.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(asdict(cfg), f)

    # Setup
    tokenizer = ModularAdditionTokenizer(p=cfg.p)
    model = create_model_from_config(tokenizer, asdict(cfg.model)).to(cfg.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Data
    full_dataset = PretrainDataset(
        tokenizer,
        n_samples=cfg.data.n_samples,
        n_operands=cfg.data.n_operands,
        seed=cfg.data.seed,
    )
    n_train = int(len(full_dataset) * cfg.data.train_frac)
    n_test = len(full_dataset) - n_train
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [n_train, n_test])

    collator = DataCollator(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Training
    epochs = cfg.epochs

    # Logging: num_logs or log_every (in epochs)
    if cfg.num_logs is not None:
        log_every = max(1, epochs // cfg.num_logs)
    else:
        log_every = cfg.log_every

    # Checkpoints: num_checkpoints or save_every (in epochs)
    if cfg.num_checkpoints is not None:
        save_every = max(1, epochs // cfg.num_checkpoints)
    else:
        save_every = cfg.save_every

    best_test_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Eval
        if epoch % log_every == 0 or epoch == epochs - 1:
            train_metrics = evaluate(model, train_loader, tokenizer, cfg.device)
            test_metrics = evaluate(model, test_loader, tokenizer, cfg.device)
            # Save best
            saved = ""
            if test_metrics["loss"] < best_test_loss:
                best_test_loss = test_metrics["loss"]
                torch.save(model, output_dir / "best_model.pt")
                saved = " ✓"

            print(
                f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f}"
                f" test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['acc']:.4f}{saved}"
            )

        # Save checkpoint
        if save_every and (epoch % save_every == 0 or epoch == epochs - 1):
            torch.save(model, output_dir / f"model_epoch_{epoch}.pt")

    # Save final
    torch.save(model, output_dir / "final_model.pt")
    print(f"Training complete. Model saved to {output_dir / 'final_model.pt'}")


if __name__ == "__main__":
    cfg = PretrainConfig()
    train(cfg)
