import subprocess
import json
import os

def get_video_codec_args(cfg=None) -> list[str]:
    """Get the appropriate FFmpeg video codec arguments based on config."""
    use_gpu = getattr(cfg, "USE_GPU", False) if cfg else False
    if use_gpu:
        # NVENC for NVIDIA GPUs: Massive speedup for video encoding
        return ["-c:v", "h264_nvenc", "-preset", "p6", "-cq", "23"]
    else:
        # CPU Fallback
        return ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def get_video_info(video_path: str) -> dict:
    """Get video width, height, fps, duration."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        {}
    )
    audio_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "audio"),
        None
    )

    # Parse fps (can be "30/1" or "29.97" format)
    fps_raw = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_raw:
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) > 0 else 30.0
    else:
        fps = float(fps_raw)

    return {
        "width":    int(video_stream.get("width", 1920)),
        "height":   int(video_stream.get("height", 1080)),
        "fps":      fps,
        "duration": float(data.get("format", {}).get("duration", 0)),
        "has_audio": audio_stream is not None
    }


def build_windows(
    words: list[dict],
    window: float = 60.0,
    overlap: float = 10.0
) -> list[dict]:
    """
    Slice word list into overlapping time windows.
    Each window: {"start": float, "end": float, "words": [...]}
    """
    if not words:
        return []

    windows  = []
    end_time = words[-1]["end"]
    pos      = words[0]["start"]

    while pos < end_time:
        w_end = pos + window
        chunk = [w for w in words if pos <= w["start"] < w_end]
        if chunk:
            windows.append({
                "start": chunk[0]["start"],
                "end":   chunk[-1]["end"],
                "words": chunk
            })
        pos += (window - overlap)

    return windows


def build_prompt(windows: list[dict], duration: float) -> str:
    """Build the transcript prompt string for the AI scorer.
    
    Includes per-word timestamps so the AI can accurately pinpoint
    clip boundaries instead of guessing from word positions.
    """
    prompt = f"Total video duration: {duration:.1f}s\n\nTranscript with per-word timestamps:\n\n"
    for w in windows:
        prompt += f"[Segment {w['start']:.1f}s – {w['end']:.1f}s]\n"
        # Include timestamp for each word so AI can pick precise boundaries
        for word in w["words"]:
            prompt += f"[{word['start']:.2f}] {word['word']} "
        prompt += "\n\n"
    return prompt


def filter_words_for_clip(
    all_words: list[dict],
    clip_start: float,
    clip_end: float
) -> list[dict]:
    """Return words that fall within a clip's time range."""
    return [
        w for w in all_words
        if clip_start <= w["start"] < clip_end
    ]


def safe_remove(path: str):
    """Delete a file if it exists, silently ignore errors."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def build_chunk_summary(words: list[dict], max_words: int = 120) -> str:
    """
    Build a condensed summary of a word list for two-pass coarse scanning.
    Picks key sentences rather than sending the full transcript.
    """
    if not words:
        return ""

    full_text = " ".join(w["word"] for w in words)
    all_words = full_text.split()

    if len(all_words) <= max_words:
        return full_text

    # Sample evenly: beginning, middle, end
    third = max_words // 3
    beginning = " ".join(all_words[:third])
    mid_start = len(all_words) // 2 - third // 2
    middle = " ".join(all_words[mid_start : mid_start + third])
    ending = " ".join(all_words[-third:])

    return f"{beginning} [...] {middle} [...] {ending}"


def build_coarse_prompt(
    chunk_summaries: list[dict],
    duration: float,
    audio_hints: str = "",
) -> str:
    """
    Build a prompt for two-pass coarse scanning (Pass 1).
    Each chunk_summary has: {index, start, end, summary}
    """
    prompt = (
        f"Total video duration: {duration:.1f}s\n\n"
        f"Below are condensed summaries of {len(chunk_summaries)} "
        f"segments from the video transcript.\n"
        f"Rate each segment's VIRAL POTENTIAL for short-form video (TikTok/Reels/Shorts).\n\n"
    )

    for cs in chunk_summaries:
        prompt += (
            f"--- Segment {cs['index']} [{cs['start']:.0f}s – {cs['end']:.0f}s] ---\n"
            f"{cs['summary']}\n\n"
        )

    if audio_hints:
        prompt += audio_hints + "\n"

    prompt += (
        "Return ONLY a valid JSON array ranking the segments by viral potential.\n"
        "Each object: {\"index\": <segment number>, \"score\": <1-10>, "
        "\"reason\": \"brief reason\"}\n"
        "Sort by score descending. Return ALL segments.\n"
        "Return ONLY the JSON array. No markdown, no explanation.\n"
    )
    return prompt


def snap_clip_boundaries(
    words: list[dict],
    start: float,
    end: float,
    min_dur: float = 25.0,
    max_dur: float = 95.0,
    pause_threshold: float = 0.4,
    search_window: float = 3.0,
) -> tuple[float, float]:
    """
    Snap clip start/end to natural speech boundaries (pauses between words).

    Looks for the nearest silence gap (>pause_threshold seconds) near
    the clip's start and end, so the clip doesn't cut mid-sentence.

    Returns (snapped_start, snapped_end).
    """
    if not words:
        return start, end

    # Find words near the start/end bounds
    start_candidates = [
        w for w in words
        if (start - search_window) <= w["start"] <= (start + search_window)
    ]
    end_candidates = [
        w for w in words
        if (end - search_window) <= w["end"] <= (end + search_window)
    ]

    # ── Snap START to a pause gap ─────────────────────────────────────────────
    best_start = start
    if len(start_candidates) >= 2:
        # Find gaps between consecutive words near the start
        gaps = []
        for i in range(1, len(start_candidates)):
            gap_start = start_candidates[i - 1]["end"]
            gap_end = start_candidates[i]["start"]
            gap_dur = gap_end - gap_start
            if gap_dur >= pause_threshold:
                # Prefer gaps closest to the original start
                gaps.append((abs(gap_end - start), gap_end))

        if gaps:
            gaps.sort()  # Sort by distance to original start
            best_start = gaps[0][1]  # Snap to the closest pause
        else:
            # Fallback: snap exactly to the start of the closest word
            closest_idx = min(range(len(start_candidates)), key=lambda i: abs(start_candidates[i]["start"] - start))
            best_start = start_candidates[closest_idx]["start"]

    # ── Snap END to a pause gap ───────────────────────────────────────────────
    best_end = end
    if len(end_candidates) >= 2:
        gaps = []
        for i in range(1, len(end_candidates)):
            gap_start = end_candidates[i - 1]["end"]
            gap_end = end_candidates[i]["start"]
            gap_dur = gap_end - gap_start
            if gap_dur >= pause_threshold:
                gaps.append((abs(gap_start - end), gap_start))

        if gaps:
            gaps.sort()
            best_end = gaps[0][1]
        else:
            # Fallback: snap exactly to the end of the closest word
            closest_idx = min(range(len(end_candidates)), key=lambda i: abs(end_candidates[i]["end"] - end))
            best_end = end_candidates[closest_idx]["end"]

    # Enforce duration constraints (SMART EXTENSION)
    dur = best_end - best_start
    if dur < min_dur:
        # Instead of blindly adding seconds (which cuts mid-word), scan forward
        # in the word list to find a natural pause AFTER the min_dur mark.
        target_end = best_start + min_dur
        found_pause = False
        if len(end_candidates) > 0:
            # Get index of the last word we checked
            last_idx = words.index(end_candidates[-1]) if end_candidates[-1] in words else 0
            for i in range(last_idx, len(words) - 1):
                gap = words[i+1]["start"] - words[i]["end"]
                if words[i]["end"] >= target_end and gap >= pause_threshold:
                    best_end = words[i]["end"]
                    found_pause = True
                    break
        
        # If we couldn't find a pause, just take the end of the word nearest to target_end
        if not found_pause:
            for w in words:
                if w["end"] >= target_end:
                    best_end = w["end"]
                    break
            else:
                best_end = best_start + min_dur # ultimate fallback

    elif dur > max_dur:
        # For max_dur, we scan BACKWARDS to find the last pause before max_dur
        target_end = best_start + max_dur
        found_pause = False
        for i in range(len(words)-2, -1, -1):
            gap = words[i+1]["start"] - words[i]["end"]
            if words[i]["end"] <= target_end and gap >= pause_threshold:
                best_end = words[i]["end"]
                found_pause = True
                break
        if not found_pause:
            best_end = best_start + max_dur

    return round(max(0.0, best_start), 3), round(best_end, 3)

