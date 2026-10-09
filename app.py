"""
app.py — Streamlit web interface for the news summarizer.

Features:
  • Upload a PDF file  → extract text → summarize
  • Upload a .txt file → summarize
  • Paste text directly → summarize
  • Automatic chunk-based summarization for long documents
  • Adjustable summary length and beam search parameters
  • Download the generated summary

Run with:
  streamlit run app.py
"""

import io
import time

import streamlit as st
import torch
from pypdf import PdfReader

from data.dataset import clean_text
from model.model import load_model, generate_summary
from utils.chunking import chunk_summarize, split_into_chunks


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="News Summarizer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Load model (cached so it only loads once)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading BART model (first run downloads ~558 MB)…")
def get_model(model_dir: str = "facebook/bart-base"):
    """Load model & tokenizer once and cache."""
    model, tokenizer, device = load_model(model_dir)
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF file's bytes."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")

    model_choice = st.selectbox(
        "Model",
        options=[
            "facebook/bart-base",
            "facebook/bart-large-cnn",
        ],
        index=0,
        help="bart-base is fastest. bart-large-cnn gives best quality but uses more VRAM.",
    )

    st.markdown("### Summary length")
    max_length = st.slider(
        "Max length (tokens)", min_value=50, max_value=300, value=128, step=10,
        help="Upper bound on summary length",
    )
    min_length = st.slider(
        "Min length (tokens)", min_value=10, max_value=150, value=30, step=5,
        help="Lower bound on summary length",
    )

    st.markdown("### Generation")
    num_beams = st.slider(
        "Beam search width", min_value=1, max_value=8, value=4,
        help="Higher = better quality, slower",
    )
    no_repeat_ngram_size = st.slider(
        "No-repeat n-gram size", min_value=0, max_value=5, value=3,
        help="Prevents repetition of phrases of this length",
    )
    length_penalty = st.slider(
        "Length penalty", min_value=0.5, max_value=3.0, value=2.0, step=0.1,
        help="> 1.0 favors longer summaries, < 1.0 favors shorter",
    )

    st.markdown("---")
    st.markdown("### Device")
    device_info = "🚀 CUDA GPU" if torch.cuda.is_available() else "💻 CPU"
    st.info(device_info)
    if torch.cuda.is_available():
        st.caption(f"Device: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("📰 Transformer-based News Summarizer")
st.markdown(
    "Upload a **PDF**, upload a **text file**, or **paste text** directly. "
    "The app uses BART with chunk-based summarization for long documents."
)

# Load model (cached)
try:
    model, tokenizer, device = get_model(model_choice)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- Input tabs ---
tab_pdf, tab_txt, tab_paste = st.tabs([
    "📄 Upload PDF",
    "📝 Upload .txt",
    "✍️ Paste text",
])

article_text = ""

with tab_pdf:
    pdf_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="pdf_uploader",
    )
    if pdf_file is not None:
        with st.spinner("Extracting text from PDF…"):
            try:
                article_text = extract_text_from_pdf(pdf_file.read())
                st.success(
                    f"Extracted {len(article_text.split())} words from "
                    f"`{pdf_file.name}`"
                )
                with st.expander("Preview extracted text"):
                    st.text_area(
                        "Text", article_text[:3000],
                        height=200, disabled=True,
                    )
            except Exception as e:
                st.error(f"Failed to parse PDF: {e}")

with tab_txt:
    txt_file = st.file_uploader(
        "Choose a .txt file",
        type=["txt"],
        key="txt_uploader",
    )
    if txt_file is not None:
        article_text = txt_file.read().decode("utf-8", errors="ignore")
        st.success(
            f"Loaded {len(article_text.split())} words from `{txt_file.name}`"
        )
        with st.expander("Preview text"):
            st.text_area(
                "Text", article_text[:3000],
                height=200, disabled=True,
            )

with tab_paste:
    pasted = st.text_area(
        "Paste your article here",
        height=300,
        placeholder="Paste the full news article text…",
    )
    if pasted.strip():
        article_text = pasted

# ---------------------------------------------------------------------------
# Summarize button
# ---------------------------------------------------------------------------

st.markdown("---")
col1, col2 = st.columns([1, 3])

with col1:
    summarize_btn = st.button(
        "🚀 Generate Summary",
        type="primary",
        use_container_width=True,
        disabled=not article_text.strip(),
    )

with col2:
    if article_text.strip():
        word_count = len(article_text.split())
        token_count = len(tokenizer.encode(article_text, add_special_tokens=False))
        chunk_count = max(1, (token_count + 511) // 462)  # approx with overlap
        st.caption(
            f"📊 {word_count} words · {token_count} tokens · "
            f"will process as {chunk_count} chunk(s)"
        )

# ---------------------------------------------------------------------------
# Run summarization
# ---------------------------------------------------------------------------

if summarize_btn and article_text.strip():
    cleaned = clean_text(article_text)
    token_count = len(tokenizer.encode(cleaned, add_special_tokens=False))

    t0 = time.time()

    with st.spinner("Generating summary…"):
        if token_count > 512:
            # Long document → chunk-based summarization
            st.info(
                f"📚 Document has {token_count} tokens (> 512). "
                "Using **chunk-based summarization**."
            )

            chunks = split_into_chunks(
                cleaned, tokenizer, max_tokens=512, overlap_tokens=50,
            )

            progress = st.progress(0.0, text="Summarizing chunks…")
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                summary = generate_summary(
                    model, tokenizer, chunk, device,
                    max_input_length=512,
                    max_output_length=max_length,
                    min_output_length=min_length,
                    num_beams=num_beams,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    length_penalty=length_penalty,
                )
                chunk_summaries.append(summary)
                progress.progress(
                    (i + 1) / (len(chunks) + 1),
                    text=f"Chunk {i+1}/{len(chunks)} done",
                )

            # Refinement pass
            progress.progress(
                len(chunks) / (len(chunks) + 1),
                text="Final refinement pass…",
            )
            merged = " ".join(chunk_summaries)
            final_summary = generate_summary(
                model, tokenizer, merged, device,
                max_input_length=512,
                max_output_length=max_length,
                min_output_length=min_length,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty,
            )
            progress.progress(1.0, text="Done!")
            progress.empty()

            # Show per-chunk summaries
            with st.expander(f"🔍 Per-chunk summaries ({len(chunks)} chunks)"):
                for i, s in enumerate(chunk_summaries, 1):
                    st.markdown(f"**Chunk {i}:** {s}")
        else:
            final_summary = generate_summary(
                model, tokenizer, cleaned, device,
                max_input_length=512,
                max_output_length=max_length,
                min_output_length=min_length,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty,
            )

    elapsed = time.time() - t0

    # --- Display result ---
    st.markdown("## ✨ Generated Summary")
    st.success(final_summary)

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Input words", len(cleaned.split()))
    with c2:
        st.metric("Summary words", len(final_summary.split()))
    with c3:
        ratio = len(final_summary.split()) / max(1, len(cleaned.split())) * 100
        st.metric("Compression", f"{ratio:.1f}%")
    with c4:
        st.metric("Time", f"{elapsed:.1f}s")

    # Download button
    st.download_button(
        "⬇️ Download summary",
        data=final_summary,
        file_name="summary.txt",
        mime="text/plain",
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "Built with BART (Facebook AI) · Chunk-based summarization for long documents · "
    "Class project demo"
)
