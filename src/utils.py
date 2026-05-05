"""
Utilities — Osmotic Attention
Visualization, analysis, and helper functions for the paper
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import wandb
import os
from src.model import OsmoticTransformer


# ─────────────────────────────────────────────
# 1. Attention Map Visualization
# ─────────────────────────────────────────────

def plot_attention_maps(
    model,
    input_ids,
    tokenizer,
    layer_idx=0,
    save_path=None,
    wandb_log=True,
):
    """
    Visualize osmotic attention maps for all heads in a layer.
    Shows how information flows between tokens.
    """
    model.eval()
    with torch.no_grad():
        output = model(input_ids)

    # Get attention weights for specified layer
    attn_weights = output["attention_weights"][layer_idx]
    # Shape: [batch, heads, seq_len, seq_len]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    # Clean up tokens
    tokens = [t.replace("##", "") for t in tokens]

    num_heads = attn_weights.shape[1]
    fig, axes = plt.subplots(
        2, num_heads // 2,
        figsize=(4 * num_heads // 2, 8)
    )
    axes = axes.flatten()

    for h in range(num_heads):
        attn = attn_weights[0, h].cpu().numpy()
        sns.heatmap(
            attn,
            ax=axes[h],
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="Blues",
            vmin=0, vmax=1,
            cbar=True,
        )
        axes[h].set_title(
            f"Head {h}\nλ={model.layers[layer_idx].self_attn.lambda_h[h].item():.4f}",
            fontsize=9
        )
        axes[h].tick_params(axis='x', rotation=45, labelsize=7)
        axes[h].tick_params(axis='y', rotation=0, labelsize=7)

    plt.suptitle(
        f"Osmotic Attention Maps — Layer {layer_idx}",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention map to {save_path}")

    if wandb_log and wandb.run is not None:
        wandb.log({f"viz/attention_map_layer{layer_idx}": wandb.Image(fig)})

    plt.show()
    return fig


# ─────────────────────────────────────────────
# 2. Lambda Evolution Visualization
# ─────────────────────────────────────────────

def plot_lambda_evolution(lambda_history, save_path=None, wandb_log=True):
    """
    Plot how osmotic coupling coefficients (lambda) evolve during training.
    lambda_history: dict of {step: {layer_head: value}}
    """
    if not lambda_history:
        print("No lambda history to plot!")
        return

    steps = sorted(lambda_history.keys())
    keys = list(lambda_history[steps[0]].keys())

    num_layers = len(set(k.split("_head")[0] for k in keys))
    num_heads = len(set(k.split("head")[1] for k in keys))

    fig, axes = plt.subplots(
        1, num_layers,
        figsize=(6 * num_layers, 4),
        sharey=True
    )
    if num_layers == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, num_heads))

    for l_idx, ax in enumerate(axes):
        for h in range(num_heads):
            key = f"layer_{l_idx}_head{h}"
            if key in lambda_history[steps[0]]:
                values = [lambda_history[s][key] for s in steps]
                ax.plot(steps, values,
                       label=f"Head {h}",
                       color=colors[h],
                       linewidth=2)

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_title(f"Layer {l_idx}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Training Step")
        ax.set_ylabel("λ (Osmotic Coupling)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Osmotic Coupling (λ) Evolution During Training\n"
        "Positive λ = attracts high-density tokens | Negative λ = repels",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved lambda evolution to {save_path}")

    if wandb_log and wandb.run is not None:
        wandb.log({"viz/lambda_evolution": wandb.Image(fig)})

    plt.show()
    return fig


# ─────────────────────────────────────────────
# 3. Information Density Visualization
# ─────────────────────────────────────────────

def plot_information_density(model, input_ids, tokenizer, save_path=None, wandb_log=False):
    """
    Visualize per-token information density (rho) across layers.
    Shows which tokens are information-rich vs sparse.
    """
    model.eval()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = [t.replace("##", "") for t in tokens]
    seq_len = len(tokens)

    all_densities = []

    with torch.no_grad():
        # Get embeddings
        pos_ids = torch.arange(input_ids.size(1)).unsqueeze(0)
        x = model.token_embedding(input_ids) + model.position_embedding(pos_ids)
        x = model.embedding_norm(x)

        for layer_idx, layer in enumerate(model.layers):
            # Compute rho for first head as representative
            x_norm = layer.norm1(x)
            q_proj = layer.self_attn.q_proj(x_norm)
            B, N, D = q_proj.shape
            head_dim = D // layer.self_attn.num_heads
            q_h = q_proj[:, :, :head_dim]  # First head

            rho = layer.self_attn.compute_information_density(q_h)
            all_densities.append(rho[0].cpu().numpy())

            # Forward through layer
            attn_out, _ = layer.self_attn(x_norm, x_norm, x_norm)
            x = x + attn_out
            x = x + layer.ffn(layer.norm2(x))

    # Plot
    densities_matrix = np.array(all_densities)  # [layers, seq_len]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Heatmap across layers
    sns.heatmap(
        densities_matrix,
        ax=ax1,
        xticklabels=tokens,
        yticklabels=[f"Layer {i}" for i in range(len(model.layers))],
        cmap="YlOrRd",
        cbar_kws={"label": "Information Density (ρ)"}
    )
    ax1.set_title("Information Density per Token per Layer", fontweight='bold')
    ax1.tick_params(axis='x', rotation=45, labelsize=8)

    # Average density per token
    avg_density = densities_matrix.mean(axis=0)
    bars = ax2.bar(range(seq_len), avg_density,
                   color=plt.cm.YlOrRd(avg_density / avg_density.max()))
    ax2.set_xticks(range(seq_len))
    ax2.set_xticklabels(tokens, rotation=45, fontsize=8)
    ax2.set_ylabel("Average Information Density (ρ)")
    ax2.set_title("Average Token Information Density Across Layers", fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle("Osmotic Information Density Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved density plot to {save_path}")

    if wandb_log and wandb.run is not None:
        wandb.log({"viz/information_density": wandb.Image(fig)})

    plt.show()
    return fig


# ─────────────────────────────────────────────
# 4. Osmotic Gradient Visualization
# ─────────────────────────────────────────────

def plot_osmotic_gradient(model, input_ids, tokenizer, layer_idx=0, save_path=None):
    """
    Visualize the osmotic gradient matrix (delta_pi).
    Shows information flow direction between token pairs.
    """
    model.eval()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = [t.replace("##", "") for t in tokens]

    with torch.no_grad():
        pos_ids = torch.arange(input_ids.size(1)).unsqueeze(0)
        x = model.token_embedding(input_ids) + model.position_embedding(pos_ids)
        x = model.embedding_norm(x)

        # Forward to target layer
        for i in range(layer_idx):
            x_norm = model.layers[i].norm1(x)
            attn_out, _ = model.layers[i].self_attn(x_norm, x_norm, x_norm)
            x = x + attn_out
            x = x + model.layers[i].ffn(model.layers[i].norm2(x))

        # Get osmotic gradient at target layer
        layer = model.layers[layer_idx]
        x_norm = layer.norm1(x)
        q_proj = layer.self_attn.q_proj(x_norm)
        head_dim = q_proj.shape[-1] // layer.self_attn.num_heads
        q_h = q_proj[:, :, :head_dim]

        rho = layer.self_attn.compute_information_density(q_h)
        delta_pi = layer.self_attn.compute_osmotic_gradient(rho)

    gradient_matrix = delta_pi[0].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Osmotic gradient heatmap
    vmax = np.abs(gradient_matrix).max()
    sns.heatmap(
        gradient_matrix,
        ax=axes[0],
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax, vmax=vmax,
        cbar_kws={"label": "Δπ (Osmotic Gradient)"}
    )
    axes[0].set_title(
        f"Osmotic Gradient Matrix — Layer {layer_idx}\n"
        "Blue = information flows TO j | Red = information flows FROM j",
        fontweight='bold'
    )
    axes[0].tick_params(axis='x', rotation=45, labelsize=8)
    axes[0].tick_params(axis='y', rotation=0, labelsize=8)

    # Net osmotic pressure per token
    net_pressure = gradient_matrix.sum(axis=0)
    colors = ['steelblue' if p > 0 else 'tomato' for p in net_pressure]
    axes[1].bar(range(len(tokens)), net_pressure, color=colors)
    axes[1].set_xticks(range(len(tokens)))
    axes[1].set_xticklabels(tokens, rotation=45, fontsize=8)
    axes[1].axhline(y=0, color='black', linewidth=1)
    axes[1].set_ylabel("Net Osmotic Pressure")
    axes[1].set_title(
        "Net Osmotic Pressure per Token\n"
        "Blue = information sink | Red = information source",
        fontweight='bold'
    )
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle("Osmotic Gradient Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
    return fig


# ─────────────────────────────────────────────
# 5. Results Comparison Table
# ─────────────────────────────────────────────

def plot_results_table(results_dict, save_path=None):
    """
    Create publication-ready results comparison table.
    results_dict: {model_name: {task: accuracy}}
    """
    import pandas as pd

    df = pd.DataFrame(results_dict).T
    df = df.round(4)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 2)

    # Highlight best per column
    for col_idx in range(len(df.columns)):
        col_values = df.iloc[:, col_idx].values
        best_row = col_values.argmax()
        table[best_row + 1, col_idx].set_facecolor('#90EE90')

    plt.title(
        "Osmotic Attention — Results Comparison\n(Green = Best per task)",
        fontsize=14, fontweight='bold', pad=20
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved results table to {save_path}")

    plt.show()
    return fig


# ─────────────────────────────────────────────
# 6. Quick Demo — Run All Visualizations
# ─────────────────────────────────────────────

def run_all_visualizations(checkpoint_path, sample_text, args):
    """
    Run all visualizations from a saved checkpoint.
    Call this after training is complete.
    """
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load model
    model = OsmoticTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        max_seq_len=getattr(args, 'max_seq_len', 128),
        num_labels=2,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Checkpoint val acc: {checkpoint['val_acc']:.4f}")

    # Tokenize sample
    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        truncation=True,
        max_length=32,
        padding="max_length",
    )
    input_ids = inputs["input_ids"].to(device)

    print(f"\nSample: '{sample_text}'")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}\n")

    os.makedirs("experiments/figures", exist_ok=True)

    # 1. Attention maps
    print("Generating attention maps...")
    plot_attention_maps(
        model, input_ids, tokenizer,
        layer_idx=0,
        save_path="experiments/figures/attention_maps.png",
        wandb_log=False,
    )

    # 2. Information density
    print("Generating information density plots...")
    plot_information_density(
        model, input_ids, tokenizer,
        save_path="experiments/figures/information_density.png",
    )

    # 3. Osmotic gradient
    print("Generating osmotic gradient plots...")
    plot_osmotic_gradient(
        model, input_ids, tokenizer,
        layer_idx=0,
        save_path="experiments/figures/osmotic_gradient.png",
    )

    print("\nAll visualizations saved to experiments/figures/")