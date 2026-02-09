import re


STEADY_PATTERNS = (
    r"\bposition remaining steady\b",
    r"\bposition remains steady\b",
    r"\bremaining steady\b",
    r"\bremains steady\b",
    r"\bbalanced continuation\b",
    r"\bposition remains balanced\b",
)


def enforce_momentum_consistency(commentary: str, momentum_update: str) -> str:
    text = (commentary or "").strip()
    if not text:
        return text

    if (momentum_update or "").strip().lower() != "none":
        return text

    lowered = text.lower()
    if not any(re.search(pattern, lowered) for pattern in STEADY_PATTERNS):
        return text

    cleaned = text
    cleaned = re.sub(r",?\s*with the position [^.,;]*steady[^.,;]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",?\s*while the position [^.,;]*steady[^.,;]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",?\s*as the position [^.,;]*steady[^.,;]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",?\s*leading to a balanced continuation[^.,;]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;")

    if not cleaned:
        return "A useful move."
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned
