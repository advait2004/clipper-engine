"""
Speaker Diarizer — Audio-Based Speaker Identification
======================================================
Uses Pyannote for speaker diarization: segments the audio track into
labelled speaker turns (Speaker A: 4.2–7.8s, Speaker B: 8.1–12.4s, etc.)

This provides a second independent signal for active speaker identification,
complementing lip movement detection from face_analyzer.py.

Pyannote is OPTIONAL — if not installed, this module gracefully returns
empty results and the engine falls back to lip-only detection.

Setup (one-time):
  pip install pyannote.audio torch
  # Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1
  # Set HF token: export HF_TOKEN=hf_xxxxx
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class SpeakerSegment:
    """A time-labelled speaker segment from audio diarization."""
    start: float          # Start time in seconds
    end: float            # End time in seconds
    speaker_label: str    # e.g., "SPEAKER_00", "SPEAKER_01"


def diarize(video_path: str, hf_token: str = "") -> list[SpeakerSegment]:
    """
    Run speaker diarization on the audio track of a video file.

    Args:
        video_path: Path to the video file.
        hf_token: HuggingFace auth token (required for Pyannote model download).
                  Can also be set via HF_TOKEN environment variable.

    Returns:
        List of SpeakerSegment with start, end, and speaker_label.
        Returns empty list if Pyannote is not available.
    """
    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError:
        print("    Diarizer: pyannote.audio not installed — skipping audio diarization")
        print("    Diarizer: Install with: pip install pyannote.audio torch")
        return []

    # Resolve HuggingFace token
    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        print("    Diarizer: No HuggingFace token found — skipping diarization")
        print("    Diarizer: Set via config or: export HF_TOKEN=hf_xxxxx")
        return []

    # Extract audio to temp WAV
    temp_wav = tempfile.mktemp(suffix="_diarize.wav")
    try:
        print("    Diarizer: Extracting audio...")
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                temp_wav,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"    Diarizer: Audio extraction failed: {result.stderr[-200:]}")
            return []

        # Run Pyannote pipeline
        print("    Diarizer: Running speaker diarization (this may take a moment)...")
        try:
            try:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=token,
                )
            except TypeError:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=token,
                )

            # Use GPU if available
            import torch
            import soundfile as sf
            import numpy as np

            if torch.cuda.is_available():
                pipeline = pipeline.to(torch.device("cuda"))
                print("    Diarizer: Using GPU acceleration")

            # Load audio waveform into memory via soundfile to bypass broken torchcodec DLLs on Windows
            data, sr = sf.read(temp_wav, dtype="float32")
            tensor = torch.from_numpy(data).float()
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            diarization = pipeline({"waveform": tensor, "sample_rate": sr})

        except Exception as e:
            print(f"    Diarizer: Pipeline failed: {e}")
            print("    Diarizer: Ensure you've accepted the model licenses at all 3 required links:")
            print("    1. https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("    2. https://huggingface.co/pyannote/segmentation-3.0")
            print("    3. https://huggingface.co/pyannote/speaker-diarization-community-1")
            return []

        # Convert results to SpeakerSegments
        segments = []
        annotation = getattr(diarization, "speaker_diarization", diarization)
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                start=round(turn.start, 3),
                end=round(turn.end, 3),
                speaker_label=speaker,
            ))

        # Sort by start time
        segments.sort(key=lambda s: s.start)

        # Print summary
        speakers = set(s.speaker_label for s in segments)
        total_dur = sum(s.end - s.start for s in segments)
        print(f"    Diarizer: Found {len(speakers)} speakers, "
              f"{len(segments)} segments, {total_dur:.1f}s of speech")

        return segments

    except Exception as e:
        print(f"    Diarizer: Unexpected error: {e}")
        return []

    finally:
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass


def get_active_speaker(
    time: float,
    segments: list[SpeakerSegment],
) -> str | None:
    """
    Get the active speaker label at a specific timestamp.

    Returns the speaker_label if someone is speaking at that time,
    or None if no one is speaking (silence/gap).
    """
    for seg in segments:
        if seg.start <= time <= seg.end:
            return seg.speaker_label
    return None


def fuse_signals(
    lip_speaker_id: int | None,
    audio_speaker_label: str | None,
    face_id_to_audio_label: dict[int, str],
    confidence_threshold: float = 0.5,
) -> tuple[int | None, float]:
    """
    Fuse lip movement and audio diarization signals to determine
    the active speaker with higher confidence.

    Args:
        lip_speaker_id: Face ID of the lip-detected speaker (or None).
        audio_speaker_label: Audio speaker label at this timestamp (or None).
        face_id_to_audio_label: Mapping from face IDs to audio speaker labels
                                (built up over time by correlating lip activity
                                with audio segments).
        confidence_threshold: Minimum confidence to accept a match.

    Returns:
        (best_face_id, confidence) — the face ID of the active speaker
        and the confidence level (0.0–1.0).
    """
    # Both signals agree
    if lip_speaker_id is not None and audio_speaker_label is not None:
        expected_label = face_id_to_audio_label.get(lip_speaker_id)
        if expected_label == audio_speaker_label:
            return lip_speaker_id, 1.0  # High confidence: both agree
        elif expected_label is None:
            # First time seeing this face — associate it with the audio label
            face_id_to_audio_label[lip_speaker_id] = audio_speaker_label
            return lip_speaker_id, 0.8  # Moderate confidence: new association
        else:
            # Disagreement — hold position (return None to indicate uncertainty)
            return None, 0.2

    # Only lip signal
    if lip_speaker_id is not None:
        return lip_speaker_id, 0.6

    # Only audio signal — try to find matching face
    if audio_speaker_label is not None:
        for face_id, label in face_id_to_audio_label.items():
            if label == audio_speaker_label:
                return face_id, 0.5
        return None, 0.3

    # Neither signal
    return None, 0.0


def build_speaker_timeline(
    segments: list[SpeakerSegment],
    duration: float,
    resolution: float = 0.25,
) -> list[dict]:
    """
    Convert diarization segments into a regular timeline at the given resolution.

    Returns a list of {time, speaker_label} dicts at each timestep.
    speaker_label is None during silence.
    """
    timeline = []
    t = 0.0
    while t < duration:
        speaker = get_active_speaker(t, segments)
        timeline.append({"time": round(t, 3), "speaker_label": speaker})
        t += resolution
    return timeline
