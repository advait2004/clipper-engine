import os
from faster_whisper import WhisperModel

# Singleton model — loaded once, reused across calls
_model: WhisperModel | None = None
_model_size: str = ""


def _get_model(model_size: str = "base") -> WhisperModel:
    global _model, _model_size

    if _model is None or _model_size != model_size:
        print(f"  Loading Whisper model '{model_size}' ...")
        import time
        for attempt in range(3):
            try:
                _model = WhisperModel(
                    model_size,
                    device="cuda",       # Change to "cuda" if you have a GPU
                    compute_type="float16" # int8 = fast on CPU, good accuracy
                )
                _model_size = model_size
                break
            except Exception as e:
                print(f"  ⚠ Failed to load/download Whisper model (Attempt {attempt+1}/3): {e}")
                if attempt == 2:
                    raise RuntimeError("Failed to download Whisper model. Please check your internet connection or HuggingFace status.") from e
                time.sleep(3)

    return _model


def transcribe(
    video_path: str,
    model_size: str = "base"
) -> tuple[list[dict], float]:
    """
    Transcribe a video file and return word-level timestamps.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cache_file = f"{video_path}.transcript.{model_size}.json"
    if os.path.exists(cache_file):
        import json
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                print(f"  Loaded cached transcript: {len(data['words'])} words over {data['duration']:.1f}s")
                return data["words"], data["duration"]
        except Exception as e:
            print(f"  Failed to load transcript cache: {e}")

    model = _get_model(model_size)
    
    import tempfile
    import subprocess
    import json
    
    # Extract audio to 16kHz WAV
    temp_wav = tempfile.mktemp(suffix=".wav")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, 
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
            temp_wav
        ], capture_output=True, check=True)
        
        segments, info = model.transcribe(
            temp_wav,
            word_timestamps=True,
            beam_size=5,
            condition_on_previous_text=False, # Reduces hallucinations on overlapping speech
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 2000
            }
        )
    finally:
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass

    words = []
    for segment in segments:
        if not segment.words:
            continue
        for w in segment.words:
            word = w.word.strip()
            if word:
                words.append({
                    "word":  word,
                    "start": round(w.start, 3),
                    "end":   round(w.end,   3)
                })

    duration = info.duration
    print(
        f"  Transcribed: {len(words)} words "
        f"over {duration:.1f}s "
        f"({info.language}, "
        f"confidence {info.language_probability:.0%})"
    )

    try:
        with open(cache_file, "w") as f:
            json.dump({"words": words, "duration": duration}, f)
    except Exception as e:
        print(f"  Failed to save transcript cache: {e}")

    return words, duration
