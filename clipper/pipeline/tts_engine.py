"""
TTS Engine — Generate AI voiceover with precise word-level timestamps
======================================================================
Uses edge-tts (Microsoft Edge's free TTS API) for realistic AI voices.

Word timestamp accuracy:
  1. edge-tts generates VTT subtitles (line-level timing)
  2. If TTS_WORD_LEVEL is enabled, we re-transcribe the generated audio
     with faster-whisper to get *true* word-level timestamps
  3. Falls back to approximate VTT parsing if Whisper fails
"""

import os
import re
import subprocess
from config import Config


def parse_vtt_time(time_str: str) -> float:
    """Converts VTT timestamp (HH:MM:SS.mmm) to seconds."""
    parts = time_str.split(':')
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def parse_vtt_to_words(vtt_path: str) -> list[dict]:
    """
    Parses an edge-tts VTT file and returns word-level timestamps.
    
    edge-tts VTT contains word-boundary cues — each cue is typically
    a single word or short phrase. For multi-word cues, we distribute
    duration evenly across words.
    
    Returns: [{"word": "hello", "start": 0.1, "end": 0.5}, ...]
    """
    words = []
    with open(vtt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_start = 0.0
    current_end = 0.0

    for line in lines:
        line = line.strip()
        if not line or line == "WEBVTT":
            continue

        # Match timestamp line: "00:00:00.100 --> 00:00:00.500"
        time_match = re.match(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s-->\s(\d{2}:\d{2}:\d{2}\.\d{3})", line)
        if time_match:
            current_start = parse_vtt_time(time_match.group(1))
            current_end = parse_vtt_time(time_match.group(2))
            continue

        # If it's a text line (not a number block or timestamp)
        if not line.isdigit() and "-->" not in line:
            text_words = line.split()
            if not text_words:
                continue

            duration = current_end - current_start
            word_dur = duration / len(text_words)

            for i, w in enumerate(text_words):
                w_clean = w.strip()
                if w_clean:
                    words.append({
                        "word": w_clean,
                        "start": round(current_start + (i * word_dur), 3),
                        "end": round(current_start + ((i + 1) * word_dur), 3),
                    })

    return words


def _whisper_retranscribe(audio_path: str, cfg: Config) -> list[dict] | None:
    """
    Re-transcribe the TTS audio with faster-whisper to get true word-level
    timestamps. This is much more accurate than approximating from VTT.
    
    Uses a small Whisper model since TTS audio is crystal clear.
    Returns word list or None if Whisper fails.
    """
    try:
        from pipeline.transcriber import transcribe
        
        # Use 'base' model — TTS audio is perfectly clean, no need for large-v3
        whisper_model = "base"
        print(f"  [TTS Engine] Re-transcribing with Whisper '{whisper_model}' for accurate word timestamps...")
        
        words, duration = transcribe(audio_path, model_size=whisper_model)
        
        if words:
            print(f"  [TTS Engine] Whisper extracted {len(words)} precise word timestamps.")
            return words
        else:
            print("  [TTS Engine] Whisper returned no words, falling back to VTT parsing.")
            return None
            
    except Exception as e:
        print(f"  [TTS Engine] Whisper re-transcription failed: {e}")
        print("  [TTS Engine] Falling back to approximate VTT timestamps.")
        return None


def generate_tts(text: str, output_audio: str, output_vtt: str, cfg: Config) -> list[dict]:
    """
    Generates TTS audio and word-level timestamps using edge-tts.
    
    If cfg.TTS_WORD_LEVEL is True, re-transcribes the audio with Whisper
    for precise karaoke timing. Otherwise, parses the VTT approximation.
    
    Returns the word-level timestamps:
        [{"word": "hello", "start": 0.1, "end": 0.5}, ...]
    """
    voice = getattr(cfg, "TTS_VOICE", "en-US-ChristopherNeural")
    print(f"  [TTS Engine] Generating voiceover (Voice: {voice})...")

    cmd = [
        "edge-tts",
        "--voice", voice,
        "--text", text,
        "--write-media", output_audio,
        "--write-subtitles", output_vtt,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError(
            "edge-tts is not installed. Run: pip install edge-tts"
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode('utf-8') if e.stderr else str(e)
        print(f"  [TTS Engine] Error running edge-tts: {stderr}")
        raise

    if not os.path.exists(output_audio):
        raise FileNotFoundError("edge-tts failed to generate audio file.")

    # ── Get word timestamps ───────────────────────────────────────────────────
    words = None

    # Option 1: Re-transcribe with Whisper for precise word-level timing
    if getattr(cfg, "TTS_WORD_LEVEL", True):
        words = _whisper_retranscribe(output_audio, cfg)

    # Option 2: Parse VTT (approximate timing)
    if words is None:
        if os.path.exists(output_vtt):
            words = parse_vtt_to_words(output_vtt)
            print(f"  [TTS Engine] Parsed {len(words)} words from VTT (approximate timing).")
        else:
            print("  [TTS Engine] Warning: No VTT file generated.")
            words = []

    print(f"  [TTS Engine] Generated {len(words)} words of audio.")
    return words
