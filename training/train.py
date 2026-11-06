#!/usr/bin/env python3
"""Fine-tune EleutherAI/pythia-70m student model on teacher distillations.

Usage:
    python training/train.py [--config training/config.yaml] [--subset 50] [--out-dir models/v1]
"""

import argparse
import hashlib
import json
import os
import random
import sys
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


class CommandDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=128):
        self.examples = []
        for r in records:
            cmd = r.get("text", "")
            target = r.get("target")
            if isinstance(target, dict):
                target_str = json.dumps(target)
            else:
                target_str = str(target)
            
            # Target sequence format
            text = f"Command: {cmd}\nJSON: {target_str}"
            encodings = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].squeeze(0)
            attention_mask = encodings["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def compute_dir_sha256(directory, output_file):
    """Compute sha256 checksums of saved model files."""
    entries = []
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            if f.endswith(".sha256"):
                continue
            fpath = os.path.join(root, f)
            rel_path = os.path.relpath(fpath, directory)
            h = hashlib.sha256()
            with open(fpath, "rb") as fp:
                while chunk := fp.read(65536):
                    h.update(chunk)
            entries.append(f"{h.hexdigest()}  {rel_path}")

    with open(output_file, "w", encoding="utf-8") as out:
        out.write("\n".join(entries) + "\n")
    print(f"Checksums saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Train Pythia-70M student model.")
    parser.add_argument("--config", default="training/config.yaml", help="Path to config.yaml")
    parser.add_argument("--subset", type=int, default=None, help="Subset size for testing (e.g. 50)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs override")
    parser.add_argument("--out-dir", default="models/v1", help="Output directory for model weights")
    parser.add_argument("--s3-sync", action="store_true", help="Upload artifacts to S3")
    args = parser.parse_args()

    # Load configuration
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    seed = cfg.get("training", {}).get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    base_model_name = cfg.get("model", {}).get("base_model", "EleutherAI/pythia-70m")
    max_seq_len = cfg.get("model", {}).get("max_seq_len", 128)
    lr = float(cfg.get("training", {}).get("learning_rate", 5e-4))
    epochs = args.epochs if args.epochs is not None else int(cfg.get("training", {}).get("epochs", 3))
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("training", {}).get("batch_size", 16))
    max_train_samples = cfg.get("data", {}).get("max_train_samples", 1500)
    train_file = cfg.get("data", {}).get("train_file", "datasets/v1/train.jsonl")

    if args.subset:
        max_train_samples = args.subset

    print(f"Loading base model: {base_model_name}")
    print(f"Output directory: {args.out_dir}")
    print(f"Training samples limit: {max_train_samples}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load training records
    records = []
    if os.path.exists(train_file):
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
                if len(records) >= max_train_samples:
                    break
    else:
        print(f"[WARN] Train file {train_file} not found. Creating placeholder synthetic records.")
        records = [
            {
                "text": "set an alarm for seven am",
                "target": {"intent": "alarm_set", "date": None, "time": "seven am", "place": None}
            }
        ] * (max_train_samples or 10)

    print(f"Loaded {len(records)} training samples.", flush=True)
    dataset = CommandDataset(records, tokenizer, max_length=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cpu")
    
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps
    )

    print(f"Starting training on {device} for {epochs} epochs ({total_steps} steps)...", flush=True)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs} | Step {step+1}/{len(dataloader)} | Loss: {loss.item():.4f}", flush=True)

        avg_loss = epoch_loss / len(dataloader)
        print(f"--> Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}", flush=True)

    # Save model artifacts
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Saving model and tokenizer to {args.out_dir}...", flush=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    checksum_file = os.path.join(args.out_dir, "model.sha256")
    compute_dir_sha256(args.out_dir, checksum_file)
    print(f"Training completed successfully. Artifacts ready in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

