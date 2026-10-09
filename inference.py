import argparse
import json
import os
import textwrap

import torch
from datasets import load_dataset
from rouge_score import rouge_scorer

from data.dataset import clean_text
from model.model import load_model, generate_summary
from utils.chunking import chunk_summarize, split_into_chunks


def compute_rouge(prediction: str, reference: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F-scores."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, prediction)
    return {k: round(v.fmeasure, 4) for k, v in scores.items()}


def evaluate_batch(model, tokenizer, device, num_samples: int = 100,
                   max_length: int = 200, min_length: int = 60):
    """Evaluate the model on the test split and print aggregate ROUGE."""
    print(f"\nEvaluating on {num_samples} test samples …\n")

    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    results = []

    for i, item in enumerate(dataset):
        article = clean_text(item["article"])
        reference = clean_text(item["highlights"])

        # Decide: single-pass or chunk-based
        token_count = len(tokenizer.encode(article, add_special_tokens=False))
        if token_count > 1024:
            prediction = chunk_summarize(
                model, tokenizer, article, device,
                max_chunk_tokens=1024,
                max_output_length=max_length,
            )
        else:
            prediction = generate_summary(
                model, tokenizer, article, device,
                max_input_length=1024,
                max_output_length=max_length,
                min_output_length=min_length,
            )

        scores = compute_rouge(prediction, reference)
        for k in totals:
            totals[k] += scores[k]

        results.append({
            "index": i,
            "prediction": prediction,
            "reference": reference,
            "rouge": scores,
        })

        if (i + 1) % 10 == 0 or (i + 1) == num_samples:
            print(f"  [{i+1}/{num_samples}]  ROUGE-1={scores['rouge1']:.4f}  "
                  f"ROUGE-2={scores['rouge2']:.4f}  ROUGE-L={scores['rougeL']:.4f}")

    # --- Aggregate ---
    n = len(results)
    avg = {k: round(v / n, 4) for k, v in totals.items()}

    print("\n" + "=" * 60)
    print("AGGREGATE ROUGE SCORES")
    print("=" * 60)
    print(f"  ROUGE-1 (word overlap)      : {avg['rouge1']}")
    print(f"  ROUGE-2 (bigram overlap)    : {avg['rouge2']}")
    print(f"  ROUGE-L (longest common seq): {avg['rougeL']}")
    print("=" * 60)

    # Save detailed results
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval_results.json", "w") as f:
        json.dump({"average_rouge": avg, "samples": results[:20]}, f, indent=2)
    print("Detailed results saved to outputs/eval_results.json\n")

    return avg


def summarize_single(model, tokenizer, device, text: str,
                     max_length: int = 200, min_length: int = 60):
    """Summarize a single article, printing generated vs reference."""
    text = clean_text(text)
    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    print(f"\nArticle tokens: {token_count}")
    print(f"Target summary length: {min_length}-{max_length} tokens")

    if token_count > 1024:
        print("Using chunk-based summarization …")
        summary = chunk_summarize(
            model, tokenizer, text, device,
            max_chunk_tokens=1024,
            max_output_length=max_length,
        )
    else:
        summary = generate_summary(
            model, tokenizer, text, device,
            max_input_length=1024,
            max_output_length=max_length,
            min_output_length=min_length,
        )

    summary_words = len(summary.split())

    print("\n" + "=" * 60)
    print(f"GENERATED SUMMARY  ({summary_words} words)")
    print("=" * 60)
    print(textwrap.fill(summary, width=80))
    print("=" * 60)
    return summary


def interactive(model, tokenizer, device,
                max_length: int = 200, min_length: int = 60):
    """Read articles from stdin in a loop."""
    print("\n--- Interactive mode (type 'quit' to exit) ---")
    print(f"Summary length: {min_length}-{max_length} tokens")
    while True:
        print("\nPaste your article (end with an empty line):")
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            if line.strip().lower() == "quit":
                return
            lines.append(line)
        if not lines:
            continue
        article = " ".join(lines)
        summarize_single(model, tokenizer, device, article,
                         max_length=max_length, min_length=min_length)


def demo(model, tokenizer, device, num_examples: int = 3,
         max_length: int = 200, min_length: int = 60):
    """Print side-by-side summaries for a few test articles."""
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")

    print("\n" + "#" * 70)
    print("  DEMO – Summarization Examples with ROUGE Scores")
    print("#" * 70)

    for i in range(num_examples):
        item = dataset[i]
        article = clean_text(item["article"])
        reference = clean_text(item["highlights"])

        token_count = len(tokenizer.encode(article, add_special_tokens=False))

        if token_count > 1024:
            prediction = chunk_summarize(
                model, tokenizer, article, device,
                max_chunk_tokens=1024,
                max_output_length=max_length,
            )
        else:
            prediction = generate_summary(
                model, tokenizer, article, device,
                max_input_length=1024,
                max_output_length=max_length,
                min_output_length=min_length,
            )

        scores = compute_rouge(prediction, reference)

        print(f"\n{'─' * 70}")
        print(f"EXAMPLE {i+1}  (tokens: {token_count})")
        print(f"{'─' * 70}")
        print(f"\n📰 ARTICLE (first 300 chars):\n{article[:300]}…\n")
        print(f"✅ REFERENCE SUMMARY:\n{textwrap.fill(reference, 80)}\n")
        print(f"🤖 GENERATED SUMMARY:\n{textwrap.fill(prediction, 80)}\n")
        print(f"📊 ROUGE:  1={scores['rouge1']:.4f}  "
              f"2={scores['rouge2']:.4f}  L={scores['rougeL']:.4f}")

    print(f"\n{'#' * 70}\n")


def parse_args():
    p = argparse.ArgumentParser(description="News summarization inference")
    p.add_argument("--model_dir", default="facebook/bart-large-cnn",
                   help="Path to fine-tuned checkpoint or HF model name")
    p.add_argument("--evaluate", action="store_true",
                   help="Run ROUGE evaluation on test split")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--input_file", type=str, default=None,
                   help="Path to a .txt file containing one article")
    p.add_argument("--demo", action="store_true",
                   help="Show 3 example summaries with ROUGE")
    p.add_argument("--max_length", type=int, default=200,
                   help="Maximum summary length in tokens (default: 200)")
    p.add_argument("--min_length", type=int, default=60,
                   help="Minimum summary length in tokens (default: 60)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, tokenizer, device = load_model(args.model_dir)

    if args.evaluate:
        evaluate_batch(model, tokenizer, device, args.num_samples,
                       max_length=args.max_length,
                       min_length=args.min_length)

    elif args.interactive:
        interactive(model, tokenizer, device,
                    max_length=args.max_length,
                    min_length=args.min_length)

    elif args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            text = f.read()
        summarize_single(model, tokenizer, device, text,
                         max_length=args.max_length,
                         min_length=args.min_length)

    elif args.demo:
        demo(model, tokenizer, device,
             max_length=args.max_length,
             min_length=args.min_length)

    else:
        # Default: run demo
        demo(model, tokenizer, device,
             max_length=args.max_length,
             min_length=args.min_length)