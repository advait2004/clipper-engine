"""
Audio Energy Analyzer
=====================
Detects volume spikes, speech rate changes, and dramatic pauses
in video audio — entirely local, no API calls needed.

These signals are fed as hints to the AI scorer to improve
viral moment detection.
"""

import os
import subprocess
import tempfile

import numpy as np


def analyze_audio_energy(
    video_path: str,
    window_sec: float = 5.0,
    peak_threshold: float = 1.5,
) -> dict:
    """
    Analyze audio energy of a video file.

    Returns a dict with:
      - peaks: list of {time, energy} dicts where energy exceeds threshold
      - pauses: list of {time, duration} dicts for silence gaps > 1s
      - speech_rate_spikes: list of timestamps with unusually fast speech
      - avg_energy: baseline average energy
    """
    temp_wav = tempfile.mktemp(suffix=".wav")
    try:
        # Extract mono 16kHz WAV
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                temp_wav,
            ],
            capture_output=True,
            check=True,
        )
        return _analyze_wav(temp_wav, window_sec, peak_threshold)
    except Exception as e:
        print(f"  ⚠ Audio analysis failed: {e}")
        return {"peaks": [], "pauses": [], "speech_rate_spikes": [], "avg_energy": 0}
    finally:
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass


def _analyze_wav(
    wav_path: str,
    window_sec: float,
    peak_threshold: float,
) -> dict:
    """Core analysis on a WAV file."""
    import soundfile as sf

    data, sr = sf.read(wav_path)

    # Ensure mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    window_samples = int(sr * window_sec)
    total_samples = len(data)

    # ── 1. Compute RMS energy per window ──────────────────────────────────────
    energies = []
    for i in range(0, total_samples, window_samples):
        chunk = data[i : i + window_samples]
        if len(chunk) < window_samples // 2:
            break
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        energies.append({"time": round(i / sr, 2), "energy": round(rms, 6)})

    if not energies:
        return {"peaks": [], "pauses": [], "speech_rate_spikes": [], "avg_energy": 0}

    energy_values = [e["energy"] for e in energies]
    avg_energy = float(np.mean(energy_values))
    median_energy = float(np.median(energy_values))

    # Peaks = windows with energy > threshold × median
    peaks = [
        e for e in energies
        if e["energy"] > peak_threshold * median_energy
    ]

    # ── 2. Detect silence / dramatic pauses ───────────────────────────────────
    silence_threshold = avg_energy * 0.1  # Below 10% of average = silence
    min_pause_sec = 0.8  # Minimum pause duration to count

    pauses = []
    silence_start = None
    small_window = int(sr * 0.25)  # 250ms windows for pause detection

    for i in range(0, total_samples, small_window):
        chunk = data[i : i + small_window]
        if len(chunk) < small_window // 2:
            break
        rms = float(np.sqrt(np.mean(chunk ** 2)))

        if rms < silence_threshold:
            if silence_start is None:
                silence_start = i / sr
        else:
            if silence_start is not None:
                duration = (i / sr) - silence_start
                if duration >= min_pause_sec:
                    pauses.append({
                        "time": round(silence_start, 2),
                        "duration": round(duration, 2),
                    })
                silence_start = None

    # ── 3. Detect speech rate spikes (via zero-crossing rate as proxy) ────────
    # High zero-crossing rate + high energy = fast/excited speech
    zcr_window = int(sr * 2.0)  # 2-second windows
    zcr_values = []

    for i in range(0, total_samples, zcr_window):
        chunk = data[i : i + zcr_window]
        if len(chunk) < zcr_window // 2:
            break
        # Zero-crossing rate
        zcr = float(np.sum(np.abs(np.diff(np.sign(chunk)))) / (2 * len(chunk)))
        chunk_rms = float(np.sqrt(np.mean(chunk ** 2)))
        zcr_values.append({
            "time": round(i / sr, 2),
            "zcr": round(zcr, 6),
            "energy": round(chunk_rms, 6),
        })

    if zcr_values:
        median_zcr = float(np.median([v["zcr"] for v in zcr_values]))
        # Speech rate spikes: high ZCR AND high energy (not just noise)
        speech_rate_spikes = [
            v["time"] for v in zcr_values
            if v["zcr"] > median_zcr * 1.4 and v["energy"] > median_energy
        ]
    else:
        speech_rate_spikes = []

    print(
        f"  Audio analysis: {len(peaks)} energy peaks, "
        f"{len(pauses)} dramatic pauses, "
        f"{len(speech_rate_spikes)} speech-rate spikes"
    )

    return {
        "peaks": peaks,
        "pauses": pauses,
        "speech_rate_spikes": speech_rate_spikes,
        "avg_energy": round(avg_energy, 6),
    }


def format_audio_hints(audio_data: dict, top_n: int = 15) -> str:
    """
    Format audio analysis results into a text hint block
    that can be appended to the AI scoring prompt.
    """
    if not audio_data or not any([
        audio_data.get("peaks"),
        audio_data.get("pauses"),
        audio_data.get("speech_rate_spikes"),
    ]):
        return ""

    lines = ["\n--- AUDIO ENERGY SIGNALS (detected locally) ---"]
    lines.append("Use these as HINTS — moments with high audio energy often")
    lines.append("indicate laughter, excitement, raised voices, or emphasis.\n")

    # Top energy peaks by magnitude
    peaks = sorted(audio_data.get("peaks", []), key=lambda p: p["energy"], reverse=True)
    if peaks:
        top_peaks = peaks[:top_n]
        times = ", ".join(f"{p['time']:.0f}s" for p in sorted(top_peaks, key=lambda p: p["time"]))
        lines.append(f"🔊 Energy peaks (loud/excited moments): {times}")

    # Dramatic pauses (sorted by duration, longest first)
    pauses = sorted(audio_data.get("pauses", []), key=lambda p: p["duration"], reverse=True)
    if pauses:
        top_pauses = pauses[:10]
        pause_strs = [f"{p['time']:.0f}s ({p['duration']:.1f}s pause)" for p in top_pauses]
        lines.append(f"⏸️  Dramatic pauses: {', '.join(pause_strs)}")

    # Speech rate spikes
    spikes = audio_data.get("speech_rate_spikes", [])
    if spikes:
        top_spikes = sorted(spikes)[:top_n]
        times = ", ".join(f"{t:.0f}s" for t in top_spikes)
        lines.append(f"⚡ Fast/excited speech: {times}")

    lines.append("")
    return "\n".join(lines)
