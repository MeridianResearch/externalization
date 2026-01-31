"""Stage 1a: Pretrain model to output answers for modular addition."""

import os
import torch
from torch.utils.data import DataLoader

from modular_addition.config import Config
from modular_addition.tokenizer import ModularAdditionTokenizer
from modular_addition.model import ModularAdditionModel, ModularAdditionConfig
from modular_addition.data import ModularAdditionDataset


def batch_accuracy(logits, input_ids, eq_token_id):
    """Vectorized accuracy: check if prediction at the first = position matches the next token."""
    # Find first = in each row
    eq_mask = input_ids == eq_token_id  # [B, T]
    # Get index of first = per row (T if none found)
    has_eq = eq_mask.any(dim=1)
    eq_pos = eq_mask.float().argmax(dim=1)  # [B] — first True index per row
    valid = has_eq & (eq_pos + 1 < input_ids.shape[1])
    if not valid.any():
        return 0, 0
    eq_pos_valid = eq_pos[valid]  # [V]
    preds = logits[valid].gather(1, eq_pos_valid.unsqueeze(1).unsqueeze(2).expand(-1, 1, logits.shape[2])).squeeze(1).argmax(dim=1)
    targets = input_ids[valid].gather(1, (eq_pos_valid + 1).unsqueeze(1)).squeeze(1)
    correct = (preds == targets).sum().item()
    return correct, valid.sum().item()


def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += out.loss.item()
            n_batches += 1
            c, t = batch_accuracy(out.logits, input_ids, tokenizer.eq_token_id)
            correct += c
            total += t
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    model.train()
    return avg_loss, accuracy


def main(config: Config | None = None):
    if config is None:
        config = Config()

    torch.manual_seed(config.seed)
    device = config.device

    tokenizer = ModularAdditionTokenizer(p=config.p)

    model_config = ModularAdditionConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = ModularAdditionModel(model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_ds = ModularAdditionDataset(
        tokenizer, mode="pretrain", size=config.pretrain.dataset_size,
        num_operands_range=config.pretrain.num_operands_range,
        max_seq_len=config.model.max_seq_len, seed=config.seed,
    )
    eval_ds = ModularAdditionDataset(
        tokenizer, mode="pretrain", size=config.pretrain.eval_size,
        num_operands_range=config.pretrain.num_operands_range,
        max_seq_len=config.model.max_seq_len, seed=config.seed + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=config.pretrain.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=config.pretrain.batch_size)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.pretrain.lr, weight_decay=config.pretrain.weight_decay
    )

    for epoch in range(config.pretrain.epochs):
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                c, t = batch_accuracy(out.logits, input_ids, tokenizer.eq_token_id)
                train_correct += c
                train_total += t

        avg_train_loss = total_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        eval_loss, eval_acc = evaluate(model, eval_loader, tokenizer, device)
        print(f"Epoch {epoch+1}/{config.pretrain.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.4f}")

    # Save
    os.makedirs(config.pretrain.save_path, exist_ok=True)
    save_path = os.path.join(config.pretrain.save_path, "model.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": model_config}, save_path)
    print(f"Saved to {save_path}")
    return model


if __name__ == "__main__":
    main()
