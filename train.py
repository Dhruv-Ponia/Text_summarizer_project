import argparse
import os
import time

import torch
from torch.cuda.amp import GradScaler, autocast
from transformers import BartForConditionalGeneration, BartTokenizer

from data.dataset import get_dataloader


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Model & tokenizer ---
    print("Loading model …")
    tokenizer = BartTokenizer.from_pretrained(args.model_name)
    model = BartForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)

    # --- Data ---
    train_loader = get_dataloader(
        split="train",
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        max_samples=args.max_samples,
    )

    # --- Optimizer & scaler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=(device == "cuda"))
    accumulation_steps = args.gradient_accumulation

    # --- Training ---
    model.train()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type=device, enabled=(device == "cuda")):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / accumulation_steps

            scaler.scale(loss).backward()

            if step % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * accumulation_steps

            if step % args.log_every == 0:
                avg = epoch_loss / step
                elapsed = format_time(time.time() - t0)
                print(
                    f"Epoch {epoch} | Step {step}/{len(train_loader)} | "
                    f"Loss {avg:.4f} | Time {elapsed}"
                )

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch} finished | Avg loss {avg_epoch_loss:.4f} ---\n")

    # --- Save ---
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Model saved to {args.save_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune BART for summarization")
    p.add_argument("--model_name", default="facebook/bart-large-cnn")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max_input_length", type=int, default=1024)
    p.add_argument("--max_target_length", type=int, default=200)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit training samples (useful for quick testing)")
    p.add_argument("--save_dir", default="checkpoints/bart-summarizer")
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())