import re
from typing import Any


NONEISH_VALUES = {"", "none", "null", "n/a", "na"}


def is_noneish_text(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in NONEISH_VALUES


def sanitize_commentary_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    # Remove leaked placeholders from model or context artifacts.
    text = re.sub(r"\b(?:none|null)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,;:-")
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\.\s*\.", ".", text)
    text = text.strip()

    if is_noneish_text(text):
        return ""
    return text


def ensure_sentence_punctuation(value: Any) -> str:
    text = sanitize_commentary_text(value)
    if not text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return text
