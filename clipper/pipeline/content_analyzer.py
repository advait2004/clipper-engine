"""
Content Analyzer — per-clip intelligence gathering
====================================================
Analyzes a single video file and produces a rich content profile:
  - Transcript (word-level timestamps)
  - Audio energy signals
  - Scene classification
  - AI-generated summary
"""

import os
import traceback

from config import Config
from pipeline import transcriber
from pipeline.audio_analyzer import analyze_audio_energy, format_audio_hints


def analyze_clip(
    video_path: str,
    cfg: Config,
    skip_face_analysis: bool = True,
) -> dict:
    """
    Analyze a single video clip and return a content profile.
    
    Returns:
        {
            "file": "001.mp4",
            "path": "/full/path/to/001.mp4",
            "duration": 62.3,
            "transcript_text": "So the director walks in...",
            "words": [...],
            "language": "en",
            "confidence": 0.98,
            "scene_type": "talking_head",
            "audio_signals": {...},
            "audio_hints": "...",
            "summary": "Speaker tells a story about...",
        }
    """
    filename = os.path.basename(video_path)
    print(f"\n  ── Analyzing: {filename} ──")

    profile = {
        "file": filename,
        "path": video_path,
        "duration": 0.0,
        "transcript_text": "",
        "words": [],
        "language": "en",
        "confidence": 0.0,
        "scene_type": "unknown",
        "audio_signals": {},
        "audio_hints": "",
        "summary": "",
    }

    # ── 1. Transcribe ─────────────────────────────────────────────────────────
    try:
        words, duration = transcriber.transcribe(video_path, cfg.WHISPER_MODEL)
        profile["words"] = words
        profile["duration"] = duration
        profile["transcript_text"] = " ".join(w["word"] for w in words)
        print(f"    Transcribed: {len(words)} words, {duration:.1f}s")
    except Exception as e:
        print(f"    ⚠ Transcription failed: {e}")
        traceback.print_exc()
        return profile

    # ── 2. Audio energy analysis ──────────────────────────────────────────────
    try:
        audio_data = analyze_audio_energy(video_path)
        profile["audio_signals"] = audio_data
        profile["audio_hints"] = format_audio_hints(audio_data)
    except Exception as e:
        print(f"    ⚠ Audio analysis failed: {e}")

    # ── 3. Scene classification (optional — requires face analysis) ───────────
    if not skip_face_analysis:
        try:
            from pipeline.face_analyzer import analyze_faces
            from pipeline.scene_classifier import classify_scene
            face_timeline = analyze_faces(video_path)
            profile["scene_type"] = classify_scene(face_timeline)
            print(f"    Scene type: {profile['scene_type']}")
        except Exception as e:
            print(f"    ⚠ Scene classification skipped: {e}")
    else:
        profile["scene_type"] = "unknown"

    # ── 4. AI summary ─────────────────────────────────────────────────────────
    try:
        summary = _ai_summarize(profile["transcript_text"], cfg)
        profile["summary"] = summary
        print(f"    Summary: {summary[:80]}...")
    except Exception as e:
        print(f"    ⚠ AI summary failed: {e}")
        # Fallback: first 100 chars of transcript
        profile["summary"] = profile["transcript_text"][:100] + "..."

    return profile


def analyze_folder(
    video_paths: list[str],
    cfg: Config,
    progress_callback=None,
) -> list[dict]:
    """
    Analyze all video clips in a folder.
    Returns a list of content profiles.
    """
    profiles = []

    for i, path in enumerate(video_paths):
        if progress_callback:
            pct = int(5 + (40 * i / max(1, len(video_paths))))
            progress_callback(pct, f"Analyzing clip {i+1}/{len(video_paths)}: {os.path.basename(path)}")

        profile = analyze_clip(path, cfg)
        profiles.append(profile)

    return profiles


def _ai_summarize(transcript_text: str, cfg: Config) -> str:
    """
    Use the configured AI provider to generate a 1-2 sentence summary.
    """
    if not transcript_text or len(transcript_text) < 20:
        return "Very short or silent clip."

    # Truncate very long transcripts
    text = transcript_text[:2000]

    prompt = (
        "Summarize this video transcript in 1-2 sentences. "
        "Focus on the main topic, key statement, or story being told. "
        "Be concise and specific.\n\n"
        f"Transcript: {text}"
    )

    # Try NVIDIA
    if cfg.AI_PROVIDER == "nvidia" or cfg.NVIDIA_API_KEY not in ("", "YOUR_NVIDIA_API_KEY_HERE"):
        try:
            return _summarize_nvidia(prompt, cfg.NVIDIA_API_KEY, cfg.NVIDIA_MODEL)
        except Exception as e:
            print(f"    NVIDIA summary failed: {e}")

    # Try Groq first (fast)
    if cfg.AI_PROVIDER == "groq" or cfg.GROQ_API_KEY not in ("", "YOUR_GROQ_API_KEY_HERE"):
        try:
            return _summarize_groq(prompt, cfg.GROQ_API_KEY, cfg.GROQ_MODEL)
        except Exception as e:
            print(f"    Groq summary failed: {e}")

    # Try Gemini
    if cfg.GEMINI_API_KEY not in ("", "YOUR_GEMINI_API_KEY_HERE"):
        try:
            return _summarize_gemini(prompt, cfg.GEMINI_API_KEY, cfg.GEMINI_MODEL)
        except Exception as e:
            print(f"    Gemini summary failed: {e}")

    # Fallback: first 100 chars
    return transcript_text[:100] + "..."


def _summarize_nvidia(prompt: str, api_key: str, model: str = "nvidia/nemotron-3-super-120b-a12b") -> str:
    """Summarize via NVIDIA API."""
    import requests
    import time

    time.sleep(2)  # Rate limit protection
    resp = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 100,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _summarize_groq(prompt: str, api_key: str, model: str) -> str:
    """Summarize via Groq API."""
    import requests
    import time

    time.sleep(2)  # Rate limit protection
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 100,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _summarize_gemini(prompt: str, api_key: str, model: str) -> str:
    """Summarize via Gemini API."""
    import google.generativeai as genai
    import time

    time.sleep(4)  # Rate limit protection
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model)
    response = gemini_model.generate_content(prompt)
    return response.text.strip()
