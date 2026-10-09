"""
Dataset module for CNN/DailyMail news summarization.
Handles loading, cleaning, tokenization, and DataLoader creation.
"""

import re
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BartTokenizer


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace, remove URLs, and strip CNN/DailyMail artefacts."""
    text = re.sub(r"http\S+", "", text)                   # URLs
    text = re.sub(r"\(CNN\)\s*--?\s*", "", text)           # CNN prefix
    text = re.sub(r"\s+", " ", text).strip()               # extra spaces
    return text


# ---------------------------------------------------------------------------
# Hugging Face → PyTorch Dataset
# ---------------------------------------------------------------------------

class CNNDailyMailDataset(Dataset):
    """Wraps a Hugging Face split into a map-style PyTorch Dataset."""

    def __init__(
        self,
        split: str = "train",
        max_input_length: int = 1024,
        max_target_length: int = 200,
        tokenizer_name: str = "facebook/bart-large-cnn",
        max_samples: int | None = None,
    ):
        print(f"Loading CNN/DailyMail ({split}) …")
        raw = load_dataset("cnn_dailymail", "3.0.0", split=split)
        if max_samples is not None:
            raw = raw.select(range(min(max_samples, len(raw))))

        self.data = raw
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        article = clean_text(item["article"])
        summary = clean_text(item["highlights"])

        # Tokenize article (encoder input)
        src = self.tokenizer(
            article,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize summary (decoder target)
        tgt = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = src["input_ids"].squeeze()
        attention_mask = src["attention_mask"].squeeze()
        labels = tgt["input_ids"].squeeze()

        # Replace padding token ids in labels with -100 so they are
        # ignored by CrossEntropyLoss inside the model.
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    split: str = "train",
    batch_size: int = 2,
    max_input_length: int = 1024,
    max_target_length: int = 200,
    max_samples: int | None = None,
    shuffle: bool | None = None,
) -> DataLoader:
    """Return a DataLoader for the requested split."""
    ds = CNNDailyMailDataset(
        split=split,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        max_samples=max_samples,
    )
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = get_dataloader(split="train", batch_size=2, max_samples=4)
    batch = next(iter(loader))
    print("input_ids shape :", batch["input_ids"].shape)
    print("attention_mask  :", batch["attention_mask"].shape)
    print("labels shape    :", batch["labels"].shape)