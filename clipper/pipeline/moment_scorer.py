"""
Moment Scorer — Multi-Signal Flash Forward Selection
=====================================================
Fuses text, audio, and video signals to find the most compelling
3-5 second window within a clip for the cold-open flash forward.

Scoring formula per second:
    Moment Score = (Text × 0.30) + (Audio × 0.40) + (Video × 0.30)

Text:   Power word density, rhetorical hooks, LLM prior
Audio:  RMS energy (loudness) + speech rate (ZCR)
Video:  Lip movement + eye openness + frontality + face size
"""

import math
import cv2
import numpy as np

# ─── Power words that indicate high-engagement moments ─────────────────────────
_POWER_WORDS = {
    # Shock / surprise
    "never", "shocking", "insane", "crazy", "unbelievable", "incredible",
    "impossible", "mindblowing", "wild", "absurd", "ridiculous",
    # Revelation
    "secret", "truth", "actually", "really", "literally", "exactly",
    "finally", "discovered", "revealed", "hidden", "exposed",
    # Urgency / importance
    "critical", "important", "essential", "dangerous", "warning",
    "urgent", "immediately", "now", "today", "forever",
    # Emotion
    "love", "hate", "fear", "terrified", "excited", "amazing",
    "beautiful", "horrible", "worst", "best", "greatest",
    # Contrarian / debate
    "wrong", "mistake", "lie", "myth", "fake", "real", "true",
    "disagree", "controversial", "unpopular",
    # Hook structures
    "imagine", "think", "remember", "listen", "watch", "look",
    "question", "answer", "reason", "why", "how", "what",
    # Future / transformation
    "future", "change", "revolution", "breakthrough", "transform",
    "billion", "million", "trillion",
}

# Rhetorical punctuation that signals hooks
_HOOK_PUNCTUATION = {"?", "!"}


def score_moments(
    video_path: str,
    clip_start: float,
    clip_end: float,
    words: list[dict],
    audio_data: dict,
    llm_flash_start: float = None,
    llm_flash_end: float = None,
    window_dur: float = 4.0,
    min_window: float = 3.0,
    max_window: float = 5.0,
) -> tuple[float, float]:
    """
    Find the best flash-forward window within a clip using multi-signal scoring.

    Args:
        video_path: Path to the source video file.
        clip_start: Start time of the clip (seconds).
        clip_end: End time of the clip (seconds).
        words: Full word list with timestamps from transcription.
        audio_data: Dict from analyze_audio_energy() with peaks, speech_rate_spikes.
        llm_flash_start: The LLM's original flash forward start (used as a prior).
        llm_flash_end: The LLM's original flash forward end.
        window_dur: Target window duration in seconds.
        min_window: Minimum window duration.
        max_window: Maximum window duration.

    Returns:
        (best_start, best_end) — the optimal flash-forward window.
    """
    clip_dur = clip_end - clip_start
    if clip_dur < min_window * 2:
        # Clip too short for a meaningful flash forward
        return clip_start, clip_start + min(clip_dur, window_dur)

    # Build per-second scores for each signal
    num_seconds = int(math.ceil(clip_dur))
    text_scores = _score_text(clip_start, clip_end, words, llm_flash_start, llm_flash_end, num_seconds)
    audio_scores = _score_audio(clip_start, clip_end, audio_data, num_seconds)
    video_scores = _score_video(video_path, clip_start, clip_end, num_seconds)

    # Fuse signals
    fused = []
    for i in range(num_seconds):
        moment = (
            0.30 * text_scores[i] +
            0.40 * audio_scores[i] +
            0.30 * video_scores[i]
        )
        fused.append(moment)

    # Find the best sliding window
    win_size = max(int(min_window), min(int(window_dur), int(max_window), num_seconds))
    best_score = -1.0
    best_idx = 0

    for i in range(num_seconds - win_size + 1):
        window_score = sum(fused[i:i + win_size]) / win_size
        if window_score > best_score:
            best_score = window_score
            best_idx = i

    best_start = clip_start + best_idx
    best_end = best_start + win_size

    # Clamp to clip boundaries
    best_end = min(best_end, clip_end)

    print(f"    MomentScorer: best window [{best_start:.1f}s–{best_end:.1f}s] "
          f"score={best_score:.3f} (T={text_scores[best_idx]:.2f} "
          f"A={audio_scores[best_idx]:.2f} V={video_scores[best_idx]:.2f})")

    return best_start, best_end


# ─── Text Signal ──────────────────────────────────────────────────────────────

def _score_text(
    clip_start: float,
    clip_end: float,
    words: list[dict],
    llm_flash_start: float,
    llm_flash_end: float,
    num_seconds: int,
) -> list[float]:
    """Score each second based on power word density and LLM prior."""
    scores = [0.0] * num_seconds

    # Filter words to this clip's range
    clip_words = [w for w in words if clip_start <= w.get("start", 0) < clip_end]

    if not clip_words:
        return scores

    # Count power words per second
    for w in clip_words:
        sec_idx = int(w["start"] - clip_start)
        if 0 <= sec_idx < num_seconds:
            word_lower = w.get("word", "").lower().strip(".,!?;:'\"")

            # Power word boost
            if word_lower in _POWER_WORDS:
                scores[sec_idx] += 0.4

            # Rhetorical punctuation boost
            raw_word = w.get("word", "")
            if any(p in raw_word for p in _HOOK_PUNCTUATION):
                scores[sec_idx] += 0.3

            # Base word presence (more words per second = more content density)
            scores[sec_idx] += 0.05

    # LLM prior: give a bonus to the seconds the AI originally picked
    if llm_flash_start is not None and llm_flash_end is not None:
        llm_start_idx = max(0, int(llm_flash_start - clip_start))
        llm_end_idx = min(num_seconds, int(llm_flash_end - clip_start) + 1)
        for i in range(llm_start_idx, llm_end_idx):
            if 0 <= i < num_seconds:
                scores[i] += 0.5  # Strong prior — the AI likely had good reason

    # Normalize to 0-1
    max_score = max(scores) if max(scores) > 0 else 1.0
    return [min(s / max_score, 1.0) for s in scores]


# ─── Audio Signal ─────────────────────────────────────────────────────────────

def _score_audio(
    clip_start: float,
    clip_end: float,
    audio_data: dict,
    num_seconds: int,
) -> list[float]:
    """Score each second based on audio energy peaks and speech rate spikes."""
    scores = [0.0] * num_seconds

    if not audio_data:
        return scores

    # Energy peaks
    peaks = audio_data.get("peaks", [])
    avg_energy = audio_data.get("avg_energy", 0.001)

    for peak in peaks:
        t = peak.get("time", 0)
        energy = peak.get("energy", 0)
        if clip_start <= t < clip_end:
            sec_idx = int(t - clip_start)
            if 0 <= sec_idx < num_seconds:
                # Normalize energy relative to the video's average
                normalized = min(energy / max(avg_energy, 0.001), 3.0) / 3.0
                scores[sec_idx] = max(scores[sec_idx], normalized)

    # Speech rate spikes
    spikes = audio_data.get("speech_rate_spikes", [])
    for t in spikes:
        if clip_start <= t < clip_end:
            sec_idx = int(t - clip_start)
            if 0 <= sec_idx < num_seconds:
                scores[sec_idx] += 0.4  # Fast speech = excitement

    # Normalize to 0-1
    max_score = max(scores) if max(scores) > 0 else 1.0
    return [min(s / max_score, 1.0) for s in scores]


# ─── Video Signal ─────────────────────────────────────────────────────────────

def _score_video(
    video_path: str,
    clip_start: float,
    clip_end: float,
    num_seconds: int,
    sample_fps: float = 4.0,
) -> list[float]:
    """
    Score each second based on facial expressiveness.
    Runs a lightweight face analysis pass at ~4 FPS over just the clip range.
    """
    scores = [0.0] * num_seconds
    counts = [0] * num_seconds

    try:
        from pipeline.face_analyzer import FaceAnalyzer
        from pipeline.face_scorer import score_face

        analyzer = FaceAnalyzer(max_faces=4, min_confidence=0.35)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("    MomentScorer: Could not open video for face analysis")
            return scores

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(fps / sample_fps))
        start_frame = int(clip_start * fps)
        end_frame = int(clip_end * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_num = start_frame

        while frame_num < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process every Nth frame
            if (frame_num - start_frame) % frame_interval == 0:
                current_time = frame_num / fps
                sec_idx = int(current_time - clip_start)

                if 0 <= sec_idx < num_seconds:
                    faces = analyzer.analyze_frame(frame)

                    if faces:
                        # Get the max face area for normalization
                        max_area = max(f.norm_area for f in faces)

                        # Score each face and take the best
                        best_face_score = 0.0
                        for face in faces:
                            fs = score_face(face, max_area)
                            best_face_score = max(best_face_score, fs)

                        scores[sec_idx] += best_face_score
                        counts[sec_idx] += 1

            frame_num += 1

        cap.release()

        # Average the scores per second
        for i in range(num_seconds):
            if counts[i] > 0:
                scores[i] /= counts[i]

    except Exception as e:
        print(f"    MomentScorer: Video scoring failed ({e}), using zeros")
        return [0.0] * num_seconds

    # Normalize to 0-1
    max_score = max(scores) if max(scores) > 0 else 1.0
    return [min(s / max_score, 1.0) for s in scores]
