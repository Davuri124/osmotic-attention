"""
Stage 4 — Deep Ablation Studies
10 ablation configurations for NeurIPS/ICML paper
All run locally — no GPU needed for small-scale ablations!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset


# ─────────────────────────────────────────────
# Ablation Model Variants
# ─────────────────────────────────────────────

class OsmoticAttentionAblation(nn.Module):
    """
    Flexible OsmoticAttention supporting all 10 ablation configs.
    Controlled by config dict.
    """

    def __init__(self, embed_dim, num_heads, config):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.config = config
        self.eps = 1e-8

        # Standard projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Osmotic parameters
        if config.get("use_osmosis", True):
            self.rho_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)

            # Membrane
            if config.get("use_membrane", True):
                self.membrane_proj_i = nn.Linear(self.head_dim, 1, bias=True)
                self.membrane_proj_j = nn.Linear(self.head_dim, 1, bias=False)

            # Lambda configuration
            lambda_config = config.get("lambda_config", "per_head")

            if lambda_config == "per_head":
                # Different lambda per head (our full model)
                self.lambda_h = nn.Parameter(torch.zeros(num_heads))

            elif lambda_config == "shared_heads":
                # Single shared lambda across all heads
                self.lambda_shared = nn.Parameter(torch.zeros(1))

            elif lambda_config == "shared_layers":
                # Lambda fixed at 1.0, not learnable
                self.register_buffer('lambda_fixed', torch.ones(num_heads))

            elif lambda_config == "fixed_positive":
                # Fixed positive lambda
                self.register_buffer('lambda_fixed', torch.ones(num_heads) * 0.1)

            elif lambda_config == "fixed_negative":
                # Fixed negative lambda
                self.register_buffer('lambda_fixed', -torch.ones(num_heads) * 0.1)

            elif lambda_config == "random_init":
                # Random initialization (not zeros)
                self.lambda_h = nn.Parameter(torch.randn(num_heads) * 0.1)

    def get_lambda(self, head_idx):
        lambda_config = self.config.get("lambda_config", "per_head")
        if lambda_config == "per_head":
            return self.lambda_h[head_idx]
        elif lambda_config == "shared_heads":
            return self.lambda_shared[0]
        elif lambda_config == "random_init":
            return self.lambda_h[head_idx]
        else:
            return self.lambda_fixed[head_idx]

    def compute_density(self, h):
        """Compute information density using configured estimator"""
        estimator = self.config.get("density_estimator", "entropy")

        if estimator == "entropy":
            # Shannon entropy (our method)
            p = F.softmax(self.rho_proj(h), dim=-1)
            return -(p * torch.log(p + self.eps)).sum(dim=-1)

        elif estimator == "l2_norm":
            # L2 norm as density proxy
            return torch.norm(h, dim=-1)

        elif estimator == "variance":
            # Variance as density proxy
            return h.var(dim=-1)

        elif estimator == "max_activation":
            # Max activation as density proxy
            return h.abs().max(dim=-1).values

    def forward(self, query, key, value,
                attn_mask=None, key_padding_mask=None):

        B, N, _ = query.shape

        Q = self.q_proj(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if self.config.get("use_osmosis", True):
            osmotic_term = torch.zeros_like(attn_scores)

            for h in range(self.num_heads):
                q_h = Q[:, h]
                k_h = K[:, h]

                # Information density
                rho = self.compute_density(q_h)

                # Osmotic gradient
                delta_pi = rho.unsqueeze(1) - rho.unsqueeze(2)

                # Membrane
                if self.config.get("use_membrane", True):
                    m_i = self.membrane_proj_i(q_h)
                    m_j = self.membrane_proj_j(k_h)
                    M = torch.sigmoid(m_i + m_j.transpose(1, 2))
                else:
                    M = torch.ones(B, N, N, device=query.device)

                lam = self.get_lambda(h)
                osmotic_term[:, h] = M * lam * delta_pi

            attn_scores = attn_scores + osmotic_term

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        return self.out_proj(output), attn_weights


class AblationTransformer(nn.Module):
    """Full transformer using ablation attention"""

    def __init__(self, vocab_size, embed_dim, num_heads,
                 num_layers, ffn_dim, max_seq_len,
                 num_labels, ablation_config):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.embedding_norm = nn.LayerNorm(embed_dim)
        self.embedding_dropout = nn.Dropout(0.1)

        self.layers = nn.ModuleList([
            AblationLayer(embed_dim, num_heads, ffn_dim, ablation_config)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_labels),
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, N = input_ids.shape
        pos = torch.arange(N, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        x = self.embedding_norm(x)
        x = self.embedding_dropout(x)

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        logits = self.classifier(x[:, 0, :])
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"logits": logits, "loss": loss}


class AblationLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, ablation_config):
        super().__init__()
        self.attn = OsmoticAttentionAblation(embed_dim, num_heads, ablation_config)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ffn_dim, embed_dim), nn.Dropout(0.1),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, key_padding_mask=None):
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask
        )
        x = residual + self.dropout(attn_out)
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x


# ─────────────────────────────────────────────
# 10 Ablation Configurations
# ─────────────────────────────────────────────

ABLATION_CONFIGS = {
    # ── Core ablations ──────────────────────
    "A1_Full_Osmotic": {
        "use_osmosis": True,
        "use_membrane": True,
        "lambda_config": "per_head",
        "density_estimator": "entropy",
        "description": "Full model — all components active"
    },
    "A2_No_Membrane": {
        "use_osmosis": True,
        "use_membrane": False,
        "lambda_config": "per_head",
        "density_estimator": "entropy",
        "description": "Remove membrane gate M_ij"
    },
    "A3_Shared_Lambda_Heads": {
        "use_osmosis": True,
        "use_membrane": True,
        "lambda_config": "shared_heads",
        "density_estimator": "entropy",
        "description": "Single shared lambda across all heads"
    },
    "A4_Shared_Lambda_Layers": {
        "use_osmosis": True,
        "use_membrane": True,
        "lambda_config": "shared_layers",
        "density_estimator": "entropy",
        "description": "Fixed lambda=1, not learnable"
    },
    "A5_Fixed_Positive_Lambda": {
        "use_osmosis": True,
        "use_membrane": True,
        "lambda_config": "fixed_positive",
        "density_estimator": "entropy",
        "description": "Fixed lambda=0.1 (positive direction)"
    },
    "A6_Fixed_Negative_Lambda": {
        "use_osmosis": True,
        "use_membrane": True,
        "lambda_config": "fixed_negative",
        "density_estimator": "entropy",
        "description": "Fixed lambda=-0.1 (negative direction)"
    },
    # ── Density estimator ablations ─────────
    "A7_L2_Density": {
        "use_osmosis": True,
        "use_membrane": True,
        "lambda_config": "per_head",
        "density_estimator": "l2_norm",
        "description": "L2 norm instead of entropy for density"
    },
    "A8_Variance_Density": {
        "use_osmosis": True,
        "use_membrane": True,
        "lambda_config": "per_head",
        "density_estimator": "variance",
        "description": "Variance instead of entropy for density"
    },
    "A9_Random_Init": {
        "use_osmosis": True,
        "use_membrane": True,
        "lambda_config": "random_init",
        "density_estimator": "entropy",
        "description": "Random lambda init instead of zeros"
    },
    # ── Baseline ────────────────────────────
    "A10_No_Osmosis": {
        "use_osmosis": False,
        "use_membrane": False,
        "lambda_config": "per_head",
        "density_estimator": "entropy",
        "description": "Vanilla transformer — no osmosis"
    },
}


# ─────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────

def run_ablation_config(config_name, ablation_config, train_loader,
                        val_loader, device, epochs=3, lr=2e-4):
    """Train one ablation config and return best accuracy"""

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model = AblationTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ffn_dim=512,
        max_seq_len=128,
        num_labels=2,
        ablation_config=ablation_config,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01
    )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    best_acc = 0.0

    for epoch in range(epochs):
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

        # Evaluate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                output = model(input_ids, attention_mask)
                preds = output["logits"].argmax(-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        if acc > best_acc:
            best_acc = acc

        print(f"  [{config_name}] Epoch {epoch+1}/{epochs} | Acc: {acc:.4f}")

    return best_acc


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


# ─────────────────────────────────────────────
# Sensitivity Analysis
# ─────────────────────────────────────────────

def run_sensitivity_analysis(train_loader, val_loader, device):
    """
    Test how sensitive osmosis is to hyperparameter choices.
    Varies: num_heads, sequence_length, lambda_init
    """
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS")
    print("="*60)

    sensitivity_results = {}

    # ── Vary number of heads ─────────────────
    print("\n1. Sensitivity to number of heads:")
    head_results = {}
    for num_heads in [1, 2, 4, 8]:
        config = {
            "use_osmosis": True,
            "use_membrane": True,
            "lambda_config": "per_head",
            "density_estimator": "entropy",
        }
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        embed_dim = 128

        # Adjust embed_dim to be divisible by num_heads
        adjusted_dim = (embed_dim // num_heads) * num_heads

        model = AblationTransformer(
            vocab_size=tokenizer.vocab_size,
            embed_dim=adjusted_dim,
            num_heads=num_heads,
            num_layers=2,
            ffn_dim=512,
            max_seq_len=128,
            num_labels=2,
            ablation_config=config,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        acc = quick_train_eval(model, optimizer, train_loader,
                               val_loader, device, epochs=2)
        head_results[num_heads] = acc
        print(f"  heads={num_heads}: {acc:.4f}")

    sensitivity_results["num_heads"] = head_results

    # ── Vary lambda learning rate ────────────
    print("\n2. Sensitivity to lambda learning rate:")
    lr_results = {}
    for lambda_lr_mult in [1, 5, 10, 20, 50]:
        config = {
            "use_osmosis": True,
            "use_membrane": True,
            "lambda_config": "per_head",
            "density_estimator": "entropy",
        }
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AblationTransformer(
            vocab_size=tokenizer.vocab_size,
            embed_dim=128, num_heads=4, num_layers=2,
            ffn_dim=512, max_seq_len=128, num_labels=2,
            ablation_config=config,
        ).to(device)

        # Differential learning rate for lambda
        optimizer = torch.optim.AdamW([
            {"params": [p for n, p in model.named_parameters()
                        if "lambda" not in n], "lr": 2e-4},
            {"params": [p for n, p in model.named_parameters()
                        if "lambda" in n], "lr": 2e-4 * lambda_lr_mult},
        ])

        acc = quick_train_eval(model, optimizer, train_loader,
                               val_loader, device, epochs=2)
        lr_results[lambda_lr_mult] = acc
        print(f"  lambda_lr_mult={lambda_lr_mult}x: {acc:.4f}")

    sensitivity_results["lambda_lr"] = lr_results
    return sensitivity_results


def quick_train_eval(model, optimizer, train_loader,
                     val_loader, device, epochs=2):
    """Quick train + eval for sensitivity analysis"""
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            output = model(input_ids, attention_mask, labels)
            optimizer.zero_grad()
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            output = model(input_ids, attention_mask)
            preds = output["logits"].argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_ablation_results(results, save_path):
    """Create publication-ready ablation figure"""

    configs = list(results.keys())
    scores = list(results.values())
    full_score = results.get("A1_Full_Osmotic", max(scores))

    colors = ['#2ecc71' if s >= full_score
              else '#e74c3c' if s < full_score - 0.01
              else '#f39c12'
              for s in scores]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(range(len(configs)), scores, color=colors, alpha=0.8)

    ax.axvline(x=full_score, color='green', linestyle='--',
               linewidth=2, label=f'Full Model ({full_score:.4f})')
    ax.axvline(x=results.get("A10_No_Osmosis", 0),
               color='red', linestyle='--',
               linewidth=2, label='No Osmosis baseline')

    ax.set_yticks(range(len(configs)))
    short_names = [
        c.replace("A1_", "").replace("A2_", "").replace("A3_", "")
         .replace("A4_", "").replace("A5_", "").replace("A6_", "")
         .replace("A7_", "").replace("A8_", "").replace("A9_", "")
         .replace("A10_", "").replace("_", " ")
        for c in configs
    ]
    ax.set_yticklabels(short_names, fontsize=10)

    for bar, score in zip(bars, scores):
        ax.text(score + 0.001, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', va='center', fontsize=9)

    ax.set_xlabel("Validation Accuracy", fontsize=12)
    ax.set_title(
        "Ablation Study — Contribution of Each Component\n"
        "Green = improves over baseline | Red = hurts performance",
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Ablation figure saved to {save_path}")
    plt.close()
    return fig


def plot_sensitivity(sensitivity_results, save_path):
    """Plot sensitivity analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heads sensitivity
    heads = list(sensitivity_results["num_heads"].keys())
    head_scores = list(sensitivity_results["num_heads"].values())
    axes[0].plot(heads, head_scores, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel("Number of Attention Heads")
    axes[0].set_ylabel("Validation Accuracy")
    axes[0].set_title("Sensitivity to Number of Heads", fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(heads)

    # Lambda LR sensitivity
    lr_mults = list(sensitivity_results["lambda_lr"].keys())
    lr_scores = list(sensitivity_results["lambda_lr"].values())
    axes[1].plot(lr_mults, lr_scores, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel("Lambda Learning Rate Multiplier")
    axes[1].set_ylabel("Validation Accuracy")
    axes[1].set_title("Sensitivity to Lambda Learning Rate", fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Sensitivity Analysis — Osmotic Attention",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Sensitivity figure saved to {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────

def run_all_ablations():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load SST2 subset — small enough for CPU!
    print("Loading dataset...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    raw = load_dataset("glue", "sst2")

    def tokenize(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    train_data = raw["train"].select(range(3000)).map(tokenize, batched=True)
    val_data = raw["validation"].map(tokenize, batched=True)

    for split in [train_data, val_data]:
        split.set_format(type="torch",
                         columns=["input_ids", "attention_mask", "label"])

    # Rename label column
    def add_labels(batch):
        batch["labels"] = batch["label"]
        return batch

    train_data = train_data.map(add_labels)
    val_data = val_data.map(add_labels)
    train_data.set_format(type="torch",
                          columns=["input_ids", "attention_mask", "labels"])
    val_data.set_format(type="torch",
                        columns=["input_ids", "attention_mask", "labels"])

    train_loader = DataLoader(train_data, batch_size=32,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=64,
                            shuffle=False, collate_fn=collate_fn)

    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # ── Run all 10 ablation configs ──────────
    print("\n" + "="*60)
    print("RUNNING 10 ABLATION CONFIGURATIONS")
    print("="*60)

    ablation_results = {}
    results_path = "experiments/results/ablation_results.json"
    os.makedirs("experiments/results", exist_ok=True)

    # Load existing if any
    if os.path.exists(results_path):
        with open(results_path) as f:
            ablation_results = json.load(f)
        print(f"Loaded {len(ablation_results)} existing results")

    for config_name, ablation_config in ABLATION_CONFIGS.items():
        if config_name in ablation_results:
            print(f"  ✅ {config_name}: {ablation_results[config_name]:.4f} (cached)")
            continue

        print(f"\nRunning: {config_name}")
        print(f"  {ablation_config['description']}")

        acc = run_ablation_config(
            config_name=config_name,
            ablation_config=ablation_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=3,
            lr=2e-4,
        )

        ablation_results[config_name] = acc

        # Save after EVERY config!
        with open(results_path, "w") as f:
            json.dump(ablation_results, f, indent=2)
        print(f"  → Best Acc: {acc:.4f} 💾 Saved!")

    # ── Sensitivity Analysis ─────────────────
    print("\nRunning sensitivity analysis...")
    sensitivity = run_sensitivity_analysis(
        train_loader, val_loader, device
    )

    # Save sensitivity
    with open("experiments/results/sensitivity_results.json", "w") as f:
        json.dump(sensitivity, f, indent=2)

    # ── Print Final Table ─────────────────────
    print("\n" + "="*65)
    print("ABLATION STUDY RESULTS")
    print("="*65)
    full_score = ablation_results.get("A1_Full_Osmotic", 0)
    baseline = ablation_results.get("A10_No_Osmosis", 0)

    for config_name, score in ablation_results.items():
        delta = score - baseline
        config = ABLATION_CONFIGS[config_name]
        marker = "⭐" if score >= full_score - 0.005 else (
                 "❌" if score < baseline else "  ")
        print(f"{marker} {config_name:<35} {score:.4f}  "
              f"Δ={delta:+.4f}  {config['description'][:30]}")

    print(f"\nFull model advantage over baseline: "
          f"{full_score - baseline:+.4f}")

    # ── Generate Figures ─────────────────────
    print("\nGenerating figures...")
    plot_ablation_results(
        ablation_results,
        "experiments/figures/ablation_results.png"
    )
    plot_sensitivity(
        sensitivity,
        "experiments/figures/sensitivity_analysis.png"
    )

    return ablation_results, sensitivity


if __name__ == "__main__":
    ablation_results, sensitivity = run_all_ablations()