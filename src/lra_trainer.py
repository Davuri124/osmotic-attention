"""
LRA Trainer — Long Range Arena Benchmark
Stage 2 of Osmotic Attention Research

Key design: saves after EVERY epoch so no progress is lost!
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import wandb
import argparse
import json
import os
import numpy as np
import time


# ─────────────────────────────────────────────
# Checkpoint Manager — saves everything!
# ─────────────────────────────────────────────

class CheckpointManager:
    """
    Saves progress after every epoch.
    Never lose work again!
    """

    def __init__(self, save_dir, task, model_name):
        self.save_dir = save_dir
        self.task = task
        self.model_name = model_name
        self.progress_file = os.path.join(
            save_dir, f"progress_{task}_{model_name}.json"
        )
        os.makedirs(save_dir, exist_ok=True)

    def save_epoch(self, epoch, score, model_state, optimizer_state, scheduler_state):
        """Save everything needed to resume from this epoch"""

        # Save model checkpoint
        ckpt_path = os.path.join(
            self.save_dir,
            f"ckpt_{self.task}_{self.model_name}_epoch{epoch}.pt"
        )
        torch.save({
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "scheduler_state_dict": scheduler_state,
            "score": score,
        }, ckpt_path)

        # Save progress JSON
        progress = self.load_progress()
        progress[f"epoch_{epoch}"] = {
            "score": score,
            "checkpoint": ckpt_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        progress["best_score"] = max(
            [v["score"] for v in progress.values() if isinstance(v, dict) and "score" in v],
            default=0
        )
        progress["last_epoch"] = epoch

        with open(self.progress_file, "w") as f:
            json.dump(progress, f, indent=2)

        print(f"  💾 Saved checkpoint: epoch {epoch}, score {score:.4f}")
        return ckpt_path

    def load_progress(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file) as f:
                return json.load(f)
        return {}

    def get_last_epoch(self):
        progress = self.load_progress()
        return progress.get("last_epoch", 0)

    def get_best_score(self):
        progress = self.load_progress()
        return progress.get("best_score", 0)

    def load_best_checkpoint(self, model):
        progress = self.load_progress()
        if not progress:
            return model, 0

        # Find best epoch
        best_epoch = max(
            [int(k.split("_")[1]) for k in progress.keys()
             if k.startswith("epoch_")],
            default=0
        )
        if best_epoch == 0:
            return model, 0

        ckpt_path = progress[f"epoch_{best_epoch}"]["checkpoint"]
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  ✅ Loaded checkpoint from epoch {best_epoch}")

        return model, best_epoch


# ─────────────────────────────────────────────
# LRA Dataset Classes
# ─────────────────────────────────────────────

class LRATextDataset(Dataset):
    """
    Character-level text classification for LRA Text task.
    Sequences up to 4000 characters.
    """

    def __init__(self, data, max_len=4000):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]

        # Character-level tokenization
        char_ids = [ord(c) % 256 for c in text[:self.max_len]]

        # Pad to max_len
        pad_len = self.max_len - len(char_ids)
        attention_mask = [1] * len(char_ids) + [0] * pad_len
        char_ids = char_ids + [0] * pad_len

        return {
            "input_ids": torch.tensor(char_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


class LRAListOpsDataset(Dataset):
    """
    ListOps — logical reasoning with nested brackets.
    Tests long-range dependency understanding.
    """

    def __init__(self, data, vocab, max_len=2000):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["source"].split()[:self.max_len]
        token_ids = [self.vocab.get(t, 1) for t in tokens]

        pad_len = self.max_len - len(token_ids)
        attention_mask = [1] * len(token_ids) + [0] * pad_len
        token_ids = token_ids + [0] * pad_len

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(item["target"], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# LRA Model — OsmoticBERT adapted for long sequences
# ─────────────────────────────────────────────

def create_lra_osmotic_model(vocab_size, max_seq_len, num_labels, model_type="osmotic"):
    """
    Create model for LRA tasks.
    model_type: 'osmotic' or 'vanilla' for comparison
    """
    import sys
    from src.model import OsmoticTransformer
    from src.evaluate import VanillaTransformer

    if model_type == "osmotic":
        model = OsmoticTransformer(
            vocab_size=vocab_size,
            embed_dim=256,
            num_heads=8,
            num_layers=4,
            ffn_dim=1024,
            max_seq_len=max_seq_len,
            dropout=0.1,
            num_labels=num_labels,
        )
    else:
        model = VanillaTransformer(
            vocab_size=vocab_size,
            embed_dim=256,
            num_heads=8,
            num_layers=4,
            ffn_dim=1024,
            max_seq_len=max_seq_len,
            dropout=0.1,
            num_labels=num_labels,
        )
    return model


# ─────────────────────────────────────────────
# Training with Epoch-by-Epoch Saving
# ─────────────────────────────────────────────

def train_lra_task(
    model,
    train_loader,
    val_loader,
    device,
    task_name,
    model_name,
    epochs,
    lr,
    save_dir,
    resume=True,
):
    """
    Train with checkpoint saving after EVERY epoch.
    Automatically resumes from last saved epoch if interrupted!
    """

    ckpt_manager = CheckpointManager(save_dir, task_name, model_name)

    # Check if we can resume
    start_epoch = 0
    if resume:
        last_epoch = ckpt_manager.get_last_epoch()
        if last_epoch > 0:
            model, start_epoch = ckpt_manager.load_best_checkpoint(model)
            print(f"  ▶️  Resuming {task_name} {model_name} from epoch {start_epoch}")

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01
    )
    total_steps = len(train_loader) * (epochs - start_epoch)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    best_score = ckpt_manager.get_best_score()

    for epoch in range(start_epoch, epochs):
        print(f"\n  Epoch {epoch+1}/{epochs} — {task_name} {model_name}")

        # ── Train ─────────────────────────────────────────────
        model.train()
        train_loss = 0
        start = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            output = model(input_ids, attention_mask, labels)
            loss = output["loss"]
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                avg_loss = train_loss / (step + 1)
                elapsed = time.time() - start
                print(
                    f"    Step {step}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Time: {elapsed:.0f}s"
                )

        # ── Evaluate ──────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                output = model(input_ids, attention_mask)
                all_preds.append(output["logits"].cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc = (all_preds.argmax(-1) == all_labels).float().mean().item()

        print(f"\n  ═══════════════════════════════════")
        print(f"  [{model_name}] {task_name} Epoch {epoch+1} | Acc: {acc:.4f}")
        print(f"  ═══════════════════════════════════")

        # ── Save after EVERY epoch ────────────────────────────
        ckpt_manager.save_epoch(
            epoch=epoch + 1,
            score=acc,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
        )

        # Log to W&B
        wandb.log({
            f"lra/{task_name}/{model_name}/acc": acc,
            f"lra/{task_name}/{model_name}/loss": train_loss / len(train_loader),
            "epoch": epoch + 1,
        })

        if acc > best_score:
            best_score = acc
            print(f"  ⭐ New best: {best_score:.4f}")

    return best_score


# ─────────────────────────────────────────────
# LRA Results Manager — saves globally
# ─────────────────────────────────────────────

class LRAResultsManager:
    """Saves LRA results after each task — never lose results!"""

    def __init__(self, results_path):
        self.results_path = results_path
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

    def load(self):
        if os.path.exists(self.results_path):
            with open(self.results_path) as f:
                return json.load(f)
        return {}

    def save_task(self, task, model_name, score):
        results = self.load()
        if task not in results:
            results[task] = {}
        results[task][model_name] = score

        # Compute improvement if both models done
        if "osmotic" in results[task] and "vanilla" in results[task]:
            results[task]["improvement"] = (
                results[task]["osmotic"] - results[task]["vanilla"]
            )

        with open(self.results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  💾 LRA results saved: {task} {model_name} = {score:.4f}")

    def print_table(self):
        results = self.load()
        if not results:
            print("No results yet!")
            return

        print(f"\n{'═'*60}")
        print("LRA RESULTS — Osmotic vs Vanilla Transformer")
        print(f"{'═'*60}")
        print(f"{'Task':<15} {'Osmotic':>10} {'Vanilla':>10} {'Δ':>8}")
        print(f"{'-'*60}")

        improvements = []
        for task, r in results.items():
            osmotic = r.get("osmotic", "—")
            vanilla = r.get("vanilla", "—")
            delta = r.get("improvement", "—")

            osmotic_str = f"{osmotic:.4f}" if isinstance(osmotic, float) else osmotic
            vanilla_str = f"{vanilla:.4f}" if isinstance(vanilla, float) else vanilla
            delta_str = f"{delta:+.4f}" if isinstance(delta, float) else delta
            star = "⭐" if isinstance(delta, float) and delta > 0 else ""

            print(f"{task:<15} {osmotic_str:>10} {vanilla_str:>10} {delta_str:>8} {star}")
            if isinstance(delta, float):
                improvements.append(delta)

        if improvements:
            avg = np.mean(improvements)
            print(f"{'-'*60}")
            print(f"{'Average':<15} {'':>10} {'':>10} {avg:>+8.4f}")
        print(f"{'═'*60}")


# ─────────────────────────────────────────────
# Main LRA Runner
# ─────────────────────────────────────────────

def run_lra_benchmark(args):
    """
    Run full LRA benchmark with automatic resume support.
    Saves after every epoch — never lose progress!
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    wandb.init(
        project=args.wandb_project,
        name=f"lra-osmotic-{'_'.join(args.tasks)}",
        config=vars(args),
        resume="allow",
    )

    results_manager = LRAResultsManager(
        os.path.join(args.results_dir, "lra_results.json")
    )

    # Load existing results to skip completed tasks
    existing = results_manager.load()

    for task in args.tasks:
        print(f"\n{'#'*60}")
        print(f"# LRA TASK: {task.upper()}")
        print(f"{'#'*60}")

        # Skip if fully completed
        if (task in existing and
            "osmotic" in existing[task] and
            "vanilla" in existing[task]):
            print(f"  ✅ Already completed! Skipping...")
            continue

        # Load dataset
        train_data, val_data, vocab_size, num_labels = load_lra_task(
            task, args.max_seq_len
        )

        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=2,
        )

        print(f"  Train: {len(train_data)} | Val: {len(val_data)}")

        for model_type in ["osmotic", "vanilla"]:

            # Skip if this model already done for this task
            if task in existing and model_type in existing[task]:
                print(f"  ✅ {model_type} already done for {task}. Skipping...")
                continue

            print(f"\n  --- Training {model_type} on {task} ---")

            model = create_lra_osmotic_model(
                vocab_size=vocab_size,
                max_seq_len=args.max_seq_len,
                num_labels=num_labels,
                model_type=model_type,
            )

            best_score = train_lra_task(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                task_name=task,
                model_name=model_type,
                epochs=args.epochs,
                lr=args.lr,
                save_dir=os.path.join(args.results_dir, "lra_checkpoints"),
                resume=True,
            )

            # Save immediately after each model!
            results_manager.save_task(task, model_type, best_score)

            # Clear GPU memory
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    results_manager.print_table()
    wandb.finish()
    return results_manager.load()


def load_lra_task(task, max_seq_len):
    """Load LRA dataset for a given task"""
    from datasets import load_dataset

    print(f"  Loading {task} dataset...")

    if task == "listops":
        dataset = load_dataset("nyu-mll/listops-1000")
        vocab = build_listops_vocab(dataset["train"])
        train_data = LRAListOpsDataset(dataset["train"], vocab, max_seq_len)
        val_data = LRAListOpsDataset(dataset["validation"], vocab, max_seq_len)
        return train_data, val_data, len(vocab) + 2, 10

    elif task == "text":
        dataset = load_dataset("imdb")
        train_data = LRATextDataset(
            [{"text": x["text"], "label": x["label"]}
             for x in dataset["train"]], max_seq_len
        )
        val_data = LRATextDataset(
            [{"text": x["text"], "label": x["label"]}
             for x in dataset["test"]], max_seq_len
        )
        return train_data, val_data, 256, 2

    elif task == "pathfinder":
        dataset = load_dataset("lvwerra/pathfinder32", trust_remote_code=True)
        train_data = PathfinderDataset(dataset["train"], max_seq_len)
        val_data = PathfinderDataset(dataset["validation"], max_seq_len)
        return train_data, val_data, 256, 2

    else:
        raise ValueError(f"Unknown LRA task: {task}")


def build_listops_vocab(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    for item in dataset:
        for token in item["source"].split():
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


class PathfinderDataset(Dataset):
    def __init__(self, data, max_len=1024):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        pixels = item["image"].convert("L")
        import numpy as np
        pixels = np.array(pixels).flatten()[:self.max_len]
        pixel_ids = (pixels // 4).astype(int).tolist()
        pad_len = self.max_len - len(pixel_ids)
        mask = [1] * len(pixel_ids) + [0] * pad_len
        pixel_ids = pixel_ids + [0] * pad_len
        return {
            "input_ids": torch.tensor(pixel_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+",
                        default=["text", "listops", "pathfinder"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--results_dir", type=str,
                        default="experiments/lra_results")
    parser.add_argument("--wandb_project", type=str,
                        default="osmotic-attention")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_lra_benchmark(args)