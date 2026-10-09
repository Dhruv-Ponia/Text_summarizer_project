import torch
from transformers import BartForConditionalGeneration, BartTokenizer


MODEL_NAME = "facebook/bart-large-cnn"


def load_model(model_name: str = MODEL_NAME, device: str | None = None):
    """Load pre-trained BART model and tokenizer."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_name} on {device} …")
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer, device


def generate_summary(
    model,
    tokenizer,
    text: str,
    device: str = "cpu",
    max_input_length: int = 1024,
    max_output_length: int = 200,
    min_output_length: int = 60,
    num_beams: int = 4,
    no_repeat_ngram_size: int = 3,
    length_penalty: float = 2.0,
) -> str:
    """Generate a summary for a single passage."""
    inputs = tokenizer(
        text,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_output_length,
            min_length=min_output_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            early_stopping=True,
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)



if __name__ == "__main__":
    model, tokenizer, device = load_model()
    sample = (
        "The stock market rallied sharply on Monday after the Federal Reserve "
        "hinted at a possible pause in interest rate hikes. Major indices rose "
        "by over 2 percent, with technology shares leading the gains."
    )
    print("Summary:", generate_summary(model, tokenizer, sample, device))