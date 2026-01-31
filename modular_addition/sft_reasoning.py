"""Stage 1b: SFT on reasoning traces for modular addition."""

import os
import torch
from torch.utils.data import DataLoader

from modular_addition.config import Config
from modular_addition.tokenizer import ModularAdditionTokenizer
from modular_addition.model import ModularAdditionModel, ModularAdditionConfig
from modular_addition.data import ModularAdditionDataset


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += out["loss"].item()
            n_batches += 1
    model.train()
    return total_loss / n_batches if n_batches > 0 else 0.0


def main(config: Config | None = None):
    if config is None:
        config = Config()

    torch.manual_seed(config.seed)
    device = config.device

    tokenizer = ModularAdditionTokenizer(p=config.p)

    # Load pretrained checkpoint
    checkpoint = torch.load(config.sft_reasoning.pretrain_checkpoint, map_location=device)
    model_config = checkpoint["config"]
    model = ModularAdditionModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded pretrained model from {config.sft_reasoning.pretrain_checkpoint}")

    train_ds = ModularAdditionDataset(
        tokenizer, mode="sft", size=config.sft_reasoning.dataset_size,
        num_operands_range=config.sft_reasoning.num_operands_range,
        max_seq_len=config.model.max_seq_len, seed=config.seed + 10,
    )
    eval_ds = ModularAdditionDataset(
        tokenizer, mode="sft", size=config.sft_reasoning.eval_size,
        num_operands_range=config.sft_reasoning.num_operands_range,
        max_seq_len=config.model.max_seq_len, seed=config.seed + 11,
    )
    train_loader = DataLoader(train_ds, batch_size=config.sft_reasoning.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=config.sft_reasoning.batch_size)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.sft_reasoning.lr, weight_decay=config.sft_reasoning.weight_decay
    )

    for epoch in range(config.sft_reasoning.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        eval_loss = evaluate(model, eval_loader, device)
        print(f"Epoch {epoch+1}/{config.sft_reasoning.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Eval Loss: {eval_loss:.4f}")

    # Save
    os.makedirs(config.sft_reasoning.save_path, exist_ok=True)
    save_path = os.path.join(config.sft_reasoning.save_path, "model.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": model_config}, save_path)
    print(f"Saved to {save_path}")
    return model


if __name__ == "__main__":
    main()
