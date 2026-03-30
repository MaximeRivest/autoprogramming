from difflib import SequenceMatcher
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.strip().lower())
    # normalize french punctuation spacing: "mot ?" -> "mot?"
    for punct in ("?", "!", ";", ":"):
        text = text.replace(f" {punct}", punct)
    return text


def metric(predicted: str, expected: str) -> float:
    return SequenceMatcher(None, normalize(predicted), normalize(expected)).ratio()
