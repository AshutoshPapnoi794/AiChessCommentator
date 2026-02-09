import re
from typing import Optional, Tuple

from commentary_text import is_noneish_text, sanitize_commentary_text


INVALID_OPENINGS = {"", "unknown", "starting position", "none"}


def _normalize_opening(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    text = str(name).strip()
    if not text:
        return None
    if text.lower() in INVALID_OPENINGS:
        return None
    return text


def build_opening_update(current_opening: Optional[str], last_announced_opening: Optional[str]) -> Tuple[str, Optional[str]]:
    current = _normalize_opening(current_opening)
    if not current:
        return "None", last_announced_opening

    previous = _normalize_opening(last_announced_opening)
    if not previous:
        return f"Opening identified: {current}.", current

    if current == previous:
        return "None", previous

    prev_lower = previous.lower()
    current_lower = current.lower()
    if current_lower.startswith(prev_lower):
        return f"Opening refined: {current}.", current

    return f"Opening changed by transposition: {current}.", current


def enforce_opening_update_consistency(commentary: str, opening_update: str, opening_name: Optional[str]) -> str:
    raw_text = (commentary or "").strip()
    text = "" if is_noneish_text(raw_text) else sanitize_commentary_text(raw_text)

    if (opening_update or "").strip().lower() == "none":
        if not text:
            return "A useful move."
        opening = _normalize_opening(opening_name)
        cleaned = text
        if opening:
            cleaned = re.sub(
                rf",?\s*{re.escape(opening)}[^.,;]*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
        cleaned = re.sub(r",?\s*in (the )?[^.,;]*opening[^.,;]*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^(in|from)\s+the\s*,?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r",\s*,", ",", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;")
        if cleaned:
            if cleaned[-1] not in ".!?":
                cleaned += "."
            return cleaned
        return text

    opening = _normalize_opening(opening_name)
    if opening and opening.lower() in text.lower():
        return text

    update = (opening_update or "").strip()
    if not update:
        return text or "A useful move."
    if not text or is_noneish_text(text):
        return update
    return f"{update} {text}".strip()
