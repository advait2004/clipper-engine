"""
Sound Effects Module
====================
Adds three layers of audio to each clip:
  1. Intro whoosh  — descending sweep at t=0
  2. Impact booms  — sub-bass punch at hook keyword moments
  3. Background pad — subtle ambient music under speech

All sounds are generated synthetically (numpy) — zero downloads,
zero copyright, works fully offline.
"""

import os
import subprocess
import tempfile

import numpy as np

try:
    import soundfile as sf
    _SF_OK = True
except ImportError:
    _SF_OK = False

SR = 44100  # Sample rate (Hz)

# ─── Hook keywords that trigger an impact boom ─────────────────────────────────
HOOK_WORDS = {
    # Contrast / surprise
    "but", "wait", "actually", "however", "except", "instead",
    "wrong", "false", "myth", "lie",
    # Revelation
    "secret", "truth", "fact", "real", "really", "turns",
    "reveals", "discovered", "found", "hidden",
    # Absolute emphasis
    "never", "always", "every", "nobody", "everyone",
    "most", "all", "none", "zero", "only",
    # Emotional charge
    "insane", "crazy", "shocking", "unbelievable", "incredible",
    "amazing", "terrible", "worst", "best", "biggest",
    # Big numbers (often signal a key stat)
    "million", "billion", "thousand", "crore", "lakh",
}

MIN_IMPACT_GAP   = 4.0   # Minimum seconds between two impacts
MAX_IMPACTS      = 4     # Cap per clip


# ─── Public API ───────────────────────────────────────────────────────────────

def add_sound_effects(
    src: str,
    words: list[dict],
    clip_start: float,
    out_path: str,
    *,
    bg_volume:     float = 0.08,
    whoosh_volume: float = 0.60,
    impact_volume: float = 0.50,
) -> str:
    """
    Layer sound effects onto a clip and save to out_path.

    Parameters
    ----------
    src          : input clip path (already captioned / reframed)
    words        : word-level transcript for this clip (full-video timestamps)
    clip_start   : start time of the clip in the original video (seconds)
    out_path     : where to write the output
    bg_volume    : background music level  (0.0 – 1.0)
    whoosh_volume: intro whoosh level
    impact_volume: impact boom level
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if not _SF_OK:
        import shutil
        shutil.copy2(src, out_path)
        print("    SFX: soundfile not installed — skipped (pip install soundfile)")
        return out_path

    # Get clip duration from ffprobe
    from pipeline.utils import get_video_info
    duration = get_video_info(src)["duration"]
    if duration <= 0:
        import shutil
        shutil.copy2(src, out_path)
        return out_path

    total_samples = int(SR * duration)

    # ── Build effects bus ─────────────────────────────────────────────────────
    bus = np.zeros((total_samples, 2), dtype=np.float32)

    # 1. Background music
    bg = _gen_bg_music(duration)
    _mix_at(bus, bg * bg_volume, 0)

    # 2. Intro whoosh
    whoosh = _gen_whoosh()
    _mix_at(bus, whoosh * whoosh_volume, 0)

    # 3. Impact booms at hook keyword moments
    impact        = _gen_impact()
    impact_times  = _detect_hook_moments(words, clip_start)
    for t_sec in impact_times:
        sample_pos = int(t_sec * SR)
        _mix_at(bus, impact * impact_volume, sample_pos)

    # Soft limiter — prevent clipping
    peak = np.abs(bus).max()
    if peak > 0.92:
        bus *= 0.92 / peak

    # ── Write effects bus to temp WAV ─────────────────────────────────────────
    tmp_wav = tempfile.mktemp(suffix=".wav")
    sf.write(tmp_wav, bus, SR)

    try:
        _ffmpeg_mix(src, tmp_wav, out_path)
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)

    print(f"    SFX: whoosh + {len(impact_times)} impact(s) + ambient music")
    return out_path


# ─── Keyword detection ────────────────────────────────────────────────────────

def _detect_hook_moments(words: list[dict], clip_start: float) -> list[float]:
    """
    Scan transcript for hook keywords.
    Returns a deduplicated list of timestamps (relative to clip start).
    """
    raw = []
    for w in words:
        clean = w["word"].lower().strip(".,!?;:\"'—–")
        if clean in HOOK_WORDS:
            t = w["start"] - clip_start
            if t > 0.8:   # leave room for the intro whoosh
                raw.append(t)

    # Enforce minimum gap between impacts
    filtered, last_t = [], -MIN_IMPACT_GAP
    for t in sorted(raw):
        if t - last_t >= MIN_IMPACT_GAP:
            filtered.append(round(t, 3))
            last_t = t

    return filtered[:MAX_IMPACTS]


# ─── Sound generators ─────────────────────────────────────────────────────────

def _gen_whoosh(duration: float = 1.8) -> np.ndarray:
    """
    Premium cinematic whoosh/riser. 
    Rich harmonic sweep + wind noise envelope + stereo auto-pan.
    """
    n  = int(SR * duration)
    t  = np.linspace(0, duration, n, endpoint=False)

    # 1. Filtered wind noise
    noise = np.random.default_rng().standard_normal(n).astype(np.float32)
    # Amplitude envelope: rise to 70%, then fall quickly
    peak_idx = int(0.7 * n)
    env = np.zeros_like(t)
    env[:peak_idx] = np.linspace(0, 1, peak_idx) ** 2
    env[peak_idx:] = np.linspace(1, 0, n - peak_idx) ** 2
    wind = noise * env * 0.35

    # 2. Rich harmonic sweep
    phase  = 2 * np.pi * (120 * t + 800 * t ** 2) # pitch rising
    sweep  = np.sin(phase) + 0.5 * np.sin(phase * 2) + 0.25 * np.sin(phase * 3)
    sweep *= env * 0.35

    mono = np.clip(sweep + wind, -1.0, 1.0)

    # 3. Auto-pan left to right for spatial feel
    pan   = np.linspace(0.2, 0.8, n)
    left  = mono * np.sqrt(1 - pan)
    right = mono * np.sqrt(pan)

    return np.stack([left, right], axis=1).astype(np.float32)


def _gen_impact(duration: float = 2.5) -> np.ndarray:
    """
    Premium cinematic impact (boom).
    Exponential pitch drop sub-bass + transient hit + metallic rumble + wide reverb tail.
    """
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    # 1. Sub-bass with exponential pitch drop (150Hz -> 40Hz)
    # phase = integral of frequency = 40*t - (110/15)*exp(-t*15)
    phase = 2 * np.pi * (40 * t - (110 / 15.0) * np.exp(-t * 15.0))
    sub = np.sin(phase) * np.exp(-t * 2.5) * 0.95
    sub = np.tanh(sub * 1.5) # saturation

    # 2. Transient hit/crack
    noise = np.random.default_rng().standard_normal(n).astype(np.float32)
    hit   = noise * np.exp(-t * 50.0) * 0.6

    # 3. Metallic/cinematic rumble (detuned oscillators)
    rumble1 = np.sin(2 * np.pi * 65 * t) * np.exp(-t * 4.0)
    rumble2 = np.sin(2 * np.pi * 68 * t) * np.exp(-t * 4.0)
    rumble  = np.tanh((rumble1 + rumble2) * 1.5) * 0.25

    # 4. Long noise tail for "reverb" feel
    tail = noise * np.exp(-t * 1.5) * 0.15

    mono = np.clip(sub + hit + rumble + tail, -1.0, 1.0)

    # 5. Stereo widening (Haas effect) on the tail/rumble
    left  = mono.copy()
    right = mono.copy()
    delay = int(SR * 0.015) # 15ms delay
    right[delay:] = right[:-delay]

    return np.stack([left, right], axis=1).astype(np.float32)


def _gen_bg_music(duration: float) -> np.ndarray:
    """
    Premium ambient cinematic pad.
    Lush detuned chords with very slow modulation.
    """
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    # Suspended cinematic chord (A minor 9)
    voices = [
        (55.00,  0.40), # A1 (sub)
        (110.00, 0.35), # A2
        (164.81, 0.25), # E3
        (246.94, 0.20), # B3 (9th adds tension/premium feel)
        (261.63, 0.15), # C4
    ]
    rng = np.random.default_rng(42)
    sig = np.zeros(n, dtype=np.float64)

    for freq, vol in voices:
        # 3 oscillators per voice for rich chorus effect
        sig += np.sin(2 * np.pi * freq * t) * vol * 0.4
        sig += np.sin(2 * np.pi * (freq + rng.uniform(0.2, 0.5)) * t) * vol * 0.3
        sig += np.sin(2 * np.pi * (freq - rng.uniform(0.2, 0.5)) * t) * vol * 0.3

    # Breathing LFO (0.1 Hz)
    lfo  = 0.70 + 0.30 * np.sin(2 * np.pi * 0.10 * t)
    sig *= lfo * 0.065

    # Fade in / out
    fade = min(int(SR * 3.0), n // 4)
    sig[:fade]  *= np.linspace(0, 1, fade)
    sig[-fade:] *= np.linspace(1, 0, fade)

    # Stereo widening
    left  = sig
    right = np.roll(sig, int(SR * 0.02)) # 20ms phase shift
    
    return np.stack([left, right], axis=1).astype(np.float32)


# ─── Utility ──────────────────────────────────────────────────────────────────

def _stereo(mono: np.ndarray) -> np.ndarray:
    """Convert 1-D mono array to (N, 2) stereo."""
    return np.stack([mono, mono], axis=1).astype(np.float32)


def _mix_at(bus: np.ndarray, sound: np.ndarray, start_sample: int):
    """Add `sound` into `bus` beginning at `start_sample`, clamp to bus length."""
    end    = min(start_sample + len(sound), len(bus))
    length = end - start_sample
    if length > 0:
        bus[start_sample:end] += sound[:length]


def _ffmpeg_mix(video: str, effects_wav: str, out_path: str):
    """
    Merge video + effects_wav using FFmpeg amix.
    Original speech is kept at full volume; effects sit alongside it.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video,
        "-i", effects_wav,
        "-filter_complex",
        (
            "[0:a]volume=1.0[orig];"
            "[1:a]volume=1.0[fx];"
            "[orig][fx]amix=inputs=2:duration=first:normalize=0[aout]"
        ),
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg SFX mix failed:\n{result.stderr[-600:]}")
