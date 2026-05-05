"""
Training Pipeline — Osmotic Attention
Connects OsmoticTransformer to real datasets with W&B tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import wandb
import argparse
import os
import time
from src.model import OsmoticTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Osmotic Attention Transformer")
    
    # Model args
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--ffn_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_seq_len", type=int, default=128)
    
    # Training args
    parser.add_argument("--task", type=str, default="sst2",
                        choices=["sst2", "mrpc", "cola"],
                        help="GLUE task to train on")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    # Logging args
    parser.add_argument("--wandb_project", type=str, default="osmotic-attention")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="experiments/checkpoints")
    
    return parser.parse_args()


def get_task_config(task):
    """GLUE task specific configuration"""
    configs = {
        "sst2": {
            "dataset": "glue",
            "subset": "sst2",
            "text_col": "sentence",
            "label_col": "label",
            "num_labels": 2,
            "metric": "accuracy",
        },
        "mrpc": {
            "dataset": "glue",
            "subset": "mrpc",
            "text_col": ["sentence1", "sentence2"],
            "label_col": "label",
            "num_labels": 2,
            "metric": "f1",
        },
        "cola": {
            "dataset": "glue",
            "subset": "cola",
            "text_col": "sentence",
            "label_col": "label",
            "num_labels": 2,
            "metric": "matthews_correlation",
        },
    }
    return configs[task]


def tokenize_dataset(dataset, tokenizer, config, max_seq_len):
    """Tokenize dataset based on task config"""
    
    def tokenize_single(examples):
        return tokenizer(
            examples[config["text_col"]],
            truncation=True,
            padding="max_length",
            max_length=max_seq_len,
        )

    def tokenize_pair(examples):
        return tokenizer(
            examples[config["text_col"][0]],
            examples[config["text_col"][1]],
            truncation=True,
            padding="max_length",
            max_length=max_seq_len,
        )

    tokenize_fn = tokenize_pair if isinstance(config["text_col"], list) else tokenize_single

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column(config["label_col"], "labels")
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    return tokenized


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def compute_accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def train_epoch(model, loader, optimizer, scheduler, device, args, epoch):
    model.train()
    total_loss = 0
    total_acc = 0
    start = time.time()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward
        output = model(input_ids, attention_mask, labels)
        loss = output["loss"]
        logits = output["logits"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Metrics
        acc = compute_accuracy(logits, labels)
        total_loss += loss.item()
        total_acc += acc

        # Log to W&B
        if step % args.log_every == 0:
            avg_loss = total_loss / (step + 1)
            avg_acc = total_acc / (step + 1)
            elapsed = time.time() - start

            # Log lambda values per layer
            lambda_logs = {}
            for i, layer in enumerate(model.layers):
                for h, lam in enumerate(layer.self_attn.lambda_h):
                    lambda_logs[f"lambda/layer{i}_head{h}"] = lam.item()

            wandb.log({
                "train/loss": avg_loss,
                "train/accuracy": avg_acc,
                "train/lr": scheduler.get_last_lr()[0],
                "train/step": epoch * len(loader) + step,
                "train/elapsed_sec": elapsed,
                **lambda_logs,
            })

            print(
                f"Epoch {epoch+1} | Step {step}/{len(loader)} | "
                f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    return total_loss / len(loader), total_acc / len(loader)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            output = model(input_ids, attention_mask, labels)
            total_loss += output["loss"].item()
            total_acc += compute_accuracy(output["logits"], labels)

    return total_loss / len(loader), total_acc / len(loader)


def main():
    args = parse_args()
    task_config = get_task_config(args.task)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # W&B init
    run_name = args.run_name or f"osmotic-{args.task}-d{args.embed_dim}-h{args.num_heads}-l{args.num_layers}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
    )

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Dataset
    print(f"Loading {args.task} dataset...")
    raw_dataset = load_dataset(task_config["dataset"], task_config["subset"])
    tokenized = tokenize_dataset(raw_dataset, tokenizer, task_config, args.max_seq_len)

    train_loader = DataLoader(
        tokenized["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        tokenized["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Model
    print("Initializing OsmoticTransformer...")
    model = OsmoticTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        num_labels=task_config["num_labels"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    wandb.log({"model/total_params": total_params})

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Save dir
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0

    # Training loop
    print(f"\nStarting training on {args.task} for {args.epochs} epochs!")
    print(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}\n")

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, args, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"{'='*60}\n")

        wandb.log({
            "epoch": epoch + 1,
            "epoch/train_loss": train_loss,
            "epoch/train_acc": train_acc,
            "epoch/val_loss": val_loss,
            "epoch/val_acc": val_acc,
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(args.save_dir, f"best_{args.task}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "args": vars(args),
            }, checkpoint_path)
            print(f"New best model saved! Val Acc: {val_acc:.4f}")
            wandb.log({"best/val_acc": best_val_acc})

    print(f"\nTraining complete! Best Val Acc: {best_val_acc:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()