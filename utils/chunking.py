from __future__ import annotations

from transformers import BartTokenizer

from model.model import generate_summary


def split_into_chunks(
    text: str,
    tokenizer: BartTokenizer,
    max_tokens: int = 1024,
    overlap_tokens: int = 100,
) -> list[str]:
    """Split text into token-bounded overlapping chunks."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) <= max_tokens:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(token_ids):
        end = start + max_tokens
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += max_tokens - overlap_tokens

    return chunks


def chunk_summarize(
    model,
    tokenizer: BartTokenizer,
    article: str,
    device: str = "cpu",
    max_chunk_tokens: int = 1024,
    overlap_tokens: int = 100,
    max_output_length: int = 200,
    min_output_length: int = 60,
    num_beams: int = 4,
) -> str:
    """Summarize a long article using hierarchical chunking."""
    chunks = split_into_chunks(
        article, tokenizer,
        max_tokens=max_chunk_tokens,
        overlap_tokens=overlap_tokens,
    )

    if len(chunks) == 1:
        return generate_summary(
            model, tokenizer, chunks[0], device,
            max_input_length=max_chunk_tokens,
            max_output_length=max_output_length,
            min_output_length=min_output_length,
            num_beams=num_beams,
        )

    # --- Stage 1: per-chunk summaries ---
    chunk_summaries: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}/{len(chunks)} …")
        summary = generate_summary(
            model, tokenizer, chunk, device,
            max_input_length=max_chunk_tokens,
            max_output_length=max_output_length,
            min_output_length=min_output_length,
            num_beams=num_beams,
        )
        chunk_summaries.append(summary)

    # --- Stage 2: refinement pass ---
    merged = " ".join(chunk_summaries)
    print("  Refinement pass …")
    final_summary = generate_summary(
        model, tokenizer, merged, device,
        max_input_length=max_chunk_tokens,
        max_output_length=max_output_length,
        min_output_length=min_output_length,
        num_beams=num_beams,
    )

    return final_summary


if __name__ == "__main__":
    from model.model import load_model

    model, tokenizer, device = load_model()

    long_article = (
        "Climate change continues to reshape global weather patterns. "
        "Scientists have observed record-breaking temperatures across "
        "multiple continents this year. The Arctic ice sheet has shrunk "
        "to its smallest extent since satellite measurements began. "
    ) * 30

    print(f"\nArticle length: {len(long_article.split())} words")
    summary = chunk_summarize(model, tokenizer, long_article, device)
    print(f"\nFinal summary:\n{summary}")