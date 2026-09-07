"""
Pacing Module — filler word removal + dead air detection
=========================================================
Cleans word lists for caption rendering:
  - Strips filler words (um, uh, like, you know, basically, literally)
  - Detects dead air gaps for the pipeline to use
  - Returns cleaned word lists so captions are crisp and professional
"""


# ─── Filler word patterns ────────────────────────────────────────────────────

# Single-word fillers (case-insensitive, stripped of punctuation)
FILLER_WORDS = {
    "um", "uh", "uhh", "umm", "erm", "er",
    "like",  # as filler (heuristic: "like" between commas / at sentence start)
    "basically", "literally", "actually",
    "right", "okay", "ok",
}

# Multi-word fillers (matched as consecutive word sequences)
FILLER_PHRASES = [
    ["you", "know"],
    ["i", "mean"],
    ["sort", "of"],
    ["kind", "of"],
]

# Words that are fillers ONLY when they appear at the very start of a sentence
# (not when used mid-sentence as real content)
START_ONLY_FILLERS = {"so", "well", "yeah", "yes", "no", "alright"}


# ─── Public API ───────────────────────────────────────────────────────────────

def clean_words_for_captions(
    words: list[dict],
    remove_fillers: bool = True,
    min_gap_for_dead_air: float = 0.8,
) -> tuple[list[dict], list[dict]]:
    """
    Clean a word list for caption rendering.

    Parameters
    ----------
    words : list of word dicts with keys: word, start, end
    remove_fillers : if True, strip filler words from the output
    min_gap_for_dead_air : minimum gap (seconds) to flag as dead air

    Returns
    -------
    (cleaned_words, dead_air_segments)
    
    cleaned_words : same format as input, fillers removed
    dead_air_segments : list of {"start": float, "end": float, "duration": float}
    """
    if not words:
        return [], []

    # ── 1. Detect dead air ────────────────────────────────────────────────────
    dead_air = _detect_dead_air(words, min_gap_for_dead_air)

    if not remove_fillers:
        return list(words), dead_air

    # ── 2. Remove filler words ────────────────────────────────────────────────
    cleaned = _remove_fillers(words)

    return cleaned, dead_air


def _detect_dead_air(
    words: list[dict],
    min_gap: float = 0.8,
) -> list[dict]:
    """Detect gaps between words that exceed min_gap seconds."""
    gaps = []

    for i in range(1, len(words)):
        gap_start = words[i - 1]["end"]
        gap_end = words[i]["start"]
        gap_dur = gap_end - gap_start

        if gap_dur >= min_gap:
            gaps.append({
                "start": round(gap_start, 3),
                "end": round(gap_end, 3),
                "duration": round(gap_dur, 3),
            })

    return gaps


def _remove_fillers(words: list[dict]) -> list[dict]:
    """
    Remove filler words from a word list.
    
    Uses heuristics to avoid removing legitimate uses:
    - "like" is only removed if it appears between pauses or as a standalone filler
    - Start-only fillers are only removed at sentence beginnings
    """
    if not words:
        return []

    # First pass: mark indices to remove
    remove_indices = set()
    n = len(words)

    for i, w in enumerate(words):
        clean_word = _normalize(w["word"])

        # Skip empty after normalization
        if not clean_word:
            continue

        # Check single-word fillers
        if clean_word in FILLER_WORDS:
            # Special case: "like" — only remove if it looks like a filler
            if clean_word == "like":
                if _is_filler_like(words, i):
                    remove_indices.add(i)
            else:
                remove_indices.add(i)

        # Check start-only fillers (only at sentence boundaries)
        elif clean_word in START_ONLY_FILLERS:
            if _is_sentence_start(words, i):
                remove_indices.add(i)

    # Check multi-word filler phrases
    for phrase in FILLER_PHRASES:
        phrase_len = len(phrase)
        for i in range(n - phrase_len + 1):
            match = all(
                _normalize(words[i + j]["word"]) == phrase[j]
                for j in range(phrase_len)
            )
            if match:
                for j in range(phrase_len):
                    remove_indices.add(i + j)

    # Build cleaned list
    cleaned = [w for i, w in enumerate(words) if i not in remove_indices]

    return cleaned


def _normalize(word: str) -> str:
    """Lowercase and strip punctuation for comparison."""
    return word.lower().strip(".,!?;:\"'—–-()[]{}…")


def _is_filler_like(words: list[dict], idx: int) -> bool:
    """
    Heuristic: "like" is a filler if:
    - It appears between pauses (gap > 0.3s on either side)
    - Or it's followed by another filler
    - Or it's at the start of a clause after a pause
    """
    if idx == 0:
        return True  # "Like, this is crazy" — filler at start

    # Check for pause before
    gap_before = words[idx]["start"] - words[idx - 1]["end"]
    if gap_before > 0.3:
        return True

    # Check for pause after
    if idx < len(words) - 1:
        gap_after = words[idx + 1]["start"] - words[idx]["end"]
        if gap_after > 0.3:
            return True

        # "like" followed by another filler
        next_word = _normalize(words[idx + 1]["word"])
        if next_word in FILLER_WORDS:
            return True

    return False


def _is_sentence_start(words: list[dict], idx: int) -> bool:
    """Check if a word is at a natural sentence boundary."""
    if idx == 0:
        return True

    # Check for a significant gap before this word
    gap = words[idx]["start"] - words[idx - 1]["end"]
    if gap > 0.6:
        return True

    # Check if the previous word ends with sentence-ending punctuation
    prev_word = words[idx - 1]["word"].strip()
    if prev_word and prev_word[-1] in ".!?":
        return True

    return False
