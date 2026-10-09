# 📰 Transformer-Based Abstractive News Summarization

An AI-powered news summarization system that generates concise, meaningful summaries from long news articles using a **BART** (Bidirectional and Auto-Regressive Transformer) Seq2Seq model with **chunk-based long document handling**.

---

## 📋 Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Chunk-Based Summarization](#chunk-based-summarization)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Viva Preparation](#viva-preparation)

---

## 🎯 Project Description

This system takes long news articles (500–2000 words) and produces abstractive summaries that:

- Capture the most important information
- Maintain factual correctness
- Avoid repetition (via `no_repeat_ngram_size`)
- Allow control over summary length
- Handle documents **longer than the model's context window** via chunk-based summarization

---

## 📊 Dataset

**CNN/DailyMail v3.0.0** from Hugging Face `datasets`:

| Split      | Samples   |
|------------|-----------|
| Train      | 287,113   |
| Validation | 13,368    |
| Test       | 11,490    |

- **Input**: Full news article (`article` field)
- **Output**: Human-written bullet-point summary (`highlights` field)
- Average article length: ~800 words (~1,100 tokens)
- Average summary length: ~55 words (~75 tokens)

```python
from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
```

---

## 🧠 Model Architecture

We use **BART-base** (`facebook/bart-base`), a Transformer-based **Seq2Seq** (Encoder-Decoder) model.

### Architecture Overview

```
Input Article
     │
     ▼
┌──────────────────────┐
│      ENCODER         │
│  6 Transformer Blocks│
│  ┌────────────────┐  │
│  │ Multi-Head     │  │
│  │ Self-Attention │  │
│  │ (12 heads)     │  │
│  ├────────────────┤  │
│  │ Feed-Forward   │  │
│  │ Network        │  │
│  ├────────────────┤  │
│  │ Layer Norm +   │  │
│  │ Residual Conn. │  │
│  └────────────────┘  │
└──────────┬───────────┘
           │ Hidden States
           ▼
┌──────────────────────┐
│      DECODER         │
│  6 Transformer Blocks│
│  ┌────────────────┐  │
│  │ Masked Self-   │  │
│  │ Attention      │  │
│  ├────────────────┤  │
│  │ Cross-Attention│  │  ← Attends to encoder output
│  ├────────────────┤  │
│  │ Feed-Forward   │  │
│  │ Network        │  │
│  └────────────────┘  │
└──────────┬───────────┘
           │
           ▼
     Output Summary
```

**Key components:**

- **Multi-Head Self-Attention**: Allows the model to attend to different parts of the input simultaneously (12 attention heads).
- **Positional Encoding**: Learned absolute positional embeddings tell the model the order of tokens.
- **Feed-Forward Layers**: Two-layer MLP with GELU activation applied to each position independently.
- **Cross-Attention** (decoder only): Lets the decoder focus on relevant parts of the encoded input.

---

## 🧩 Chunk-Based Summarization

### The Problem

BART's maximum input length is 1,024 tokens (we use 512 for GPU memory). Many news articles exceed this limit. **Naïve truncation discards the latter half of the article**, losing critical information.

### The Solution – Hierarchical Chunking

```
Long Article (1500 tokens)
     │
     ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Chunk 1    │  │  Chunk 2    │  │  Chunk 3    │
│  (512 tok)  │  │  (512 tok)  │  │  (512 tok)  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
  Stage 1: Summarize each chunk independently
       │                │                │
       ▼                ▼                ▼
   Summary 1        Summary 2        Summary 3
       │                │                │
       └────────┬───────┘────────────────┘
                │
     Stage 2: Merge & Refine
                │
                ▼
         Final Summary
```

**Steps:**

1. **Split** the article into overlapping chunks of 512 tokens (50-token overlap preserves context at boundaries).
2. **Summarize** each chunk independently with the model.
3. **Concatenate** chunk summaries into one block of text.
4. **Refine** by running the concatenated text through the model a second time, producing a coherent final summary.

### Why Chunking > Truncation

| Aspect               | Truncation           | Chunk-Based              |
|-----------------------|----------------------|--------------------------|
| Information coverage  | First 512 tokens only| Entire article           |
| Ending details        | Lost                 | Preserved                |
| Summary coherence     | May be one-sided     | Balanced via refinement  |
| Compute cost          | 1 forward pass       | N+1 forward passes       |

---

## ⚙️ Installation

```bash
# Clone or copy the project
cd news_summarizer

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, NVIDIA GPU with ≥ 6 GB VRAM (or CPU for inference).

---

## 🚀 Usage

### Training

```bash
# Quick test (100 samples, 1 epoch)
python train.py --max_samples 100 --epochs 1

# Full training (all data, 1 epoch – takes several hours on a single GPU)
python train.py --epochs 1 --batch_size 2 --gradient_accumulation 4

# Resume with a fine-tuned checkpoint
python train.py --model_name checkpoints/bart-summarizer --epochs 1
```

**Training parameters (6 GB GPU safe):**

| Parameter               | Value     |
|--------------------------|-----------|
| Batch size               | 2         |
| Gradient accumulation    | 4 (effective batch = 8) |
| Learning rate            | 3e-5      |
| Mixed precision (fp16)   | Enabled   |
| Epochs                   | 1–2       |

### Inference

```bash
# Demo: show 3 test examples with ROUGE scores
python inference.py --demo

# Evaluate on 100 test samples
python inference.py --evaluate --num_samples 100

# Summarize a text file
python inference.py --input_file my_article.txt

# Interactive mode
python inference.py --interactive

# Use a fine-tuned checkpoint
python inference.py --model_dir checkpoints/bart-summarizer --demo
```

---

## 📈 Evaluation

We use **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) metrics:

| Metric    | Measures                           | Example                          |
|-----------|------------------------------------|----------------------------------|
| ROUGE-1   | Unigram (word) overlap             | How many individual words match  |
| ROUGE-2   | Bigram (two-word phrase) overlap   | How many word pairs match        |
| ROUGE-L   | Longest Common Subsequence (LCS)   | Structural similarity            |

All scores are reported as **F-measure** (harmonic mean of precision and recall).

### Expected Results (BART-base, pre-trained, no fine-tuning)

| Metric  | Score (approx.) |
|---------|-----------------|
| ROUGE-1 | 0.30 – 0.35     |
| ROUGE-2 | 0.10 – 0.15     |
| ROUGE-L | 0.22 – 0.28     |

After fine-tuning for 1 epoch on CNN/DailyMail, scores improve by 5–10 points.

### Sample Output

```
📰 ARTICLE: (CNN) -- The Federal Reserve announced Wednesday that it
will keep interest rates unchanged, citing ongoing economic uncertainty...

✅ REFERENCE SUMMARY:
Federal Reserve keeps rates unchanged. Economic uncertainty cited as
primary reason. Markets react positively to the decision.

🤖 GENERATED SUMMARY:
The Federal Reserve has decided to maintain current interest rates amid
concerns about economic uncertainty. The decision was announced on
Wednesday.

📊 ROUGE:  1=0.4521  2=0.2143  L=0.3889
```

---

## 📁 Project Structure

```
news_summarizer/
│
├── data/
│   ├── __init__.py
│   └── dataset.py          # Dataset loading, cleaning, tokenization
│
├── model/
│   ├── __init__.py
│   └── model.py             # BART model loading & single-pass generation
│
├── utils/
│   ├── __init__.py
│   └── chunking.py          # Chunk splitting & hierarchical summarization
│
├── train.py                 # Training loop (fp16, grad accumulation)
├── inference.py             # Inference, evaluation, interactive mode
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🎤 Viva Preparation

### Q: Explain the Transformer-based Seq2Seq model.

BART is a denoising autoencoder built on the Transformer architecture. The **encoder** reads the entire input article using bidirectional self-attention (every token can attend to every other token). The **decoder** generates the summary left-to-right using masked self-attention (each token can only attend to previous tokens) plus cross-attention to the encoder's output. This separation lets the encoder deeply understand the input while the decoder focuses on fluent generation.

### Q: Why is chunk-based summarization better than truncation?

Truncation cuts the article at 512 tokens, discarding everything after. For a 1,500-token article, you lose two-thirds of the content — often the conclusion and key details. Chunk-based summarization processes the *entire* article by splitting it into overlapping windows, summarizing each, then merging and refining into a single coherent summary. This preserves information from all parts of the article.

### Q: Explain ROUGE evaluation metrics.

ROUGE measures overlap between a generated summary and a human-written reference:
- **ROUGE-1**: Counts matching individual words (unigrams). High ROUGE-1 means good topical coverage.
- **ROUGE-2**: Counts matching two-word phrases (bigrams). High ROUGE-2 indicates the summary captures specific details and phrasing.
- **ROUGE-L**: Finds the Longest Common Subsequence, measuring structural similarity without requiring consecutive matches. It rewards summaries that follow the reference's logical flow.

All three are reported as F-scores (balancing precision and recall). Higher is better, with state-of-the-art models achieving ROUGE-1 ≈ 0.44 on CNN/DailyMail.

### Q: How do you handle GPU memory constraints?

Four techniques: **(1)** Mixed-precision training (fp16) halves memory for activations and gradients. **(2)** Gradient accumulation simulates a batch of 8 using actual batches of 2. **(3)** Input truncation to 512 tokens per chunk. **(4)** Using `bart-base` (139M params) instead of `bart-large` (406M params).

---

## 📝 License

This project is for educational / academic purposes only. The CNN/DailyMail dataset and BART model are subject to their respective licenses.
