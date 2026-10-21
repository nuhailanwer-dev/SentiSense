"""Model loading and inference for the SentiSense application."""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = Path(__file__).parent / "Model"
MAX_TEXT_LENGTH = 10_000
CLASS_ORDER = ("negative", "neutral", "positive")


def select_device() -> torch.device:
    """Use CUDA if this PyTorch build can run kernels on the installed GPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        return torch.device("cuda")
    except RuntimeError as error:
        print(f"CUDA is unavailable ({error}); falling back to CPU.")
        return torch.device("cpu")


class SentimentAnalyzer:
    """Loads the local Hugging Face model and predicts sentiment scores."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        self.device = select_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.id2label = {
            int(label_id): label.lower()
            for label_id, label in self.model.config.id2label.items()
        }

    def predict(self, text: str) -> dict:
        """Return predicted sentiment and scores for a non-empty text string."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter some text to analyze.")
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text must be {MAX_TEXT_LENGTH:,} characters or fewer.")

        inputs = self.tokenizer(text.strip(), return_tensors="pt", truncation=True, max_length=128)
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(**inputs).logits[0], dim=-1).cpu().tolist()
        scores = {self.id2label[i]: probabilities[i] for i in range(len(probabilities))}
        sentiment = max(scores, key=scores.get)
        return {
            "sentiment": sentiment,
            "score": scores[sentiment],
            "scores": {label: scores[label] for label in CLASS_ORDER},
        }
