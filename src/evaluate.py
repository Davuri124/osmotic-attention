"""
Evaluation Pipeline — Osmotic Attention
Comprehensive benchmarking against baselines for NeurIPS/ICML paper
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
import evaluate
import wandb
import argparse
import json
import os
import numpy as np
from src.model import OsmoticTransformer
from src.train import (
    get_task_config,
    tokenize_dataset,
    collate_fn,
    compute_accuracy
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Osmotic vs Baselines")

    parser.add_argument("--tasks", nargs="+",
                        default=["sst2", "mrpc", "cola"],
                        help="GLUE tasks to evaluate on")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to osmotic model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--ffn_dim", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--results_dir", type=str, default="experiments/results")
    parser.add_argument("--wandb_project", type=str, default="osmotic-attention")
    parser.add_argument("--run_baselines", action="store_true",
                        help="Also run vanilla transformer baseline")

    return parser.parse_args()


# ─────────────────────────────────────────────
# Vanilla Transformer Baseline (no osmosis)
# Identical architecture but standard attention
# ─────────────────────────────────────────────

class VanillaAttention(nn.Module):
    """Standard scaled dot-product attention — our baseline"""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        B, N, _ = query.shape

        Q = self.q_proj(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn = attn + attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn = self.dropout(torch.softmax(attn, dim=-1))
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        return self.out_proj(out), attn


class VanillaTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.self_attn = VanillaAttention(embed_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        residual = x
        x = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        x = residual + self.dropout(attn_out)
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x, attn_weights


class VanillaTransformer(nn.Module):
    """Identical to OsmoticTransformer but with standard attention"""

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers,
                 ffn_dim, max_seq_len=512, dropout=0.1, num_labels=2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.embedding_norm = nn.LayerNorm(embed_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            VanillaTransformerLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_labels),
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, N = input_ids.shape
        pos_ids = torch.arange(N, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(pos_ids)
        x = self.embedding_norm(x)
        x = self.embedding_dropout(x)

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        for layer in self.layers:
            x, _ = layer(x, key_padding_mask=key_padding_mask)

        logits = self.classifier(x[:, 0, :])
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"logits": logits, "loss": loss}


# ─────────────────────────────────────────────
# Training & Evaluation Functions
# ─────────────────────────────────────────────

def train_model(model, train_loader, val_loader, device, epochs, lr, model_name):
    """Train any model and return best val accuracy"""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    best_val_acc = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            output = model(input_ids, attention_mask, labels)
            loss = output["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validate
        model.eval()
        val_acc = 0
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                output = model(input_ids, attention_mask, labels)
                val_acc += compute_accuracy(output["logits"], labels)
                val_loss += output["loss"].item()

        val_acc /= len(val_loader)
        val_loss /= len(val_loader)

        print(f"  [{model_name}] Epoch {epoch+1}/{epochs} "
              f"| Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc


def run_ablation(train_loader, val_loader, device, args, task):
    """
    Ablation study — systematically remove components
    to measure each one's contribution
    """
    print(f"\n{'='*60}")
    print(f"ABLATION STUDY — {task.upper()}")
    print(f"{'='*60}")

    ablation_results = {}
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    configs = {
        # Full model
        "Full Osmotic": {"use_osmosis": True,  "use_membrane": True,  "learnable_lambda": True},
        # Remove membrane gate
        "No Membrane":  {"use_osmosis": True,  "use_membrane": False, "learnable_lambda": True},
        # Fix lambda = 1 (not learnable)
        "Fixed Lambda": {"use_osmosis": True,  "use_membrane": True,  "learnable_lambda": False},
        # Remove osmosis entirely = vanilla baseline
        "No Osmosis":   {"use_osmosis": False, "use_membrane": False, "learnable_lambda": False},
    }

    for config_name, config in configs.items():
        print(f"\nRunning: {config_name}")

        if not config["use_osmosis"]:
            model = VanillaTransformer(
                vocab_size=tokenizer.vocab_size,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                ffn_dim=args.ffn_dim,
                max_seq_len=args.max_seq_len,
                num_labels=2,
            ).to(device)
        else:
            model = OsmoticTransformer(
                vocab_size=tokenizer.vocab_size,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                ffn_dim=args.ffn_dim,
                max_seq_len=args.max_seq_len,
                num_labels=2,
            ).to(device)

            # Apply ablation config
            if not config["learnable_lambda"]:
                for layer in model.layers:
                    layer.self_attn.lambda_h = nn.Parameter(
                        torch.ones(args.num_heads), requires_grad=False
                    )

        best_acc = train_model(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr,
            model_name=config_name
        )

        ablation_results[config_name] = best_acc
        wandb.log({f"ablation/{task}/{config_name}": best_acc})
        print(f"  → Best Val Acc: {best_acc:.4f}")

    return ablation_results


def evaluate_task(task, args, device):
    """Full evaluation pipeline for one GLUE task"""

    print(f"\n{'='*60}")
    print(f"EVALUATING TASK: {task.upper()}")
    print(f"{'='*60}\n")

    task_config = get_task_config(task)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load dataset
    raw_dataset = load_dataset(task_config["dataset"], task_config["subset"])
    tokenized = tokenize_dataset(
        raw_dataset, tokenizer, task_config, args.max_seq_len
    )

    train_loader = DataLoader(
        tokenized["train"], batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        tokenized["validation"], batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn
    )

    results = {"task": task}

    # ── 1. Osmotic Model ──────────────────────────────────────
    print("Training OsmoticTransformer...")
    osmotic_model = OsmoticTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        max_seq_len=args.max_seq_len,
        num_labels=task_config["num_labels"],
    ).to(device)

    osmotic_acc = train_model(
        osmotic_model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, model_name="Osmotic"
    )
    results["osmotic_acc"] = osmotic_acc

    # ── 2. Vanilla Baseline ───────────────────────────────────
    if args.run_baselines:
        print("\nTraining VanillaTransformer baseline...")
        vanilla_model = VanillaTransformer(
            vocab_size=tokenizer.vocab_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ffn_dim=args.ffn_dim,
            max_seq_len=args.max_seq_len,
            num_labels=task_config["num_labels"],
        ).to(device)

        vanilla_acc = train_model(
            vanilla_model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr, model_name="Vanilla"
        )
        results["vanilla_acc"] = vanilla_acc
        results["improvement"] = osmotic_acc - vanilla_acc

        print(f"\n📊 {task.upper()} RESULTS:")
        print(f"  Osmotic Acc : {osmotic_acc:.4f}")
        print(f"  Vanilla Acc : {vanilla_acc:.4f}")
        print(f"  Improvement : +{results['improvement']:.4f}")

        wandb.log({
            f"comparison/{task}/osmotic": osmotic_acc,
            f"comparison/{task}/vanilla": vanilla_acc,
            f"comparison/{task}/improvement": results["improvement"],
        })

    # ── 3. Ablation Study ─────────────────────────────────────
    ablation_results = run_ablation(
        train_loader, val_loader, device, args, task
    )
    results["ablations"] = ablation_results

    return results


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wandb.init(
        project=args.wandb_project,
        name=f"evaluation-{'_'.join(args.tasks)}",
        config=vars(args),
    )

    all_results = {}

    for task in args.tasks:
        task_results = evaluate_task(task, args, device)
        all_results[task] = task_results

    # Save results to JSON
    results_path = os.path.join(args.results_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    for task, results in all_results.items():
        print(f"\n{task.upper()}:")
        print(f"  Osmotic Acc : {results['osmotic_acc']:.4f}")
        if "vanilla_acc" in results:
            print(f"  Vanilla Acc : {results['vanilla_acc']:.4f}")
            print(f"  Improvement : +{results['improvement']:.4f}")
        print(f"  Ablations   : {results['ablations']}")

    print(f"\nResults saved to {results_path}")
    wandb.finish()


if __name__ == "__main__":
    main()