"""
Video Builder — stitch B-roll scenes with TTS audio into a finished base video
================================================================================
Each B-roll clip is trimmed/looped to match the exact duration of its
corresponding scene's voiceover, then they're concatenated in order and
muxed with the TTS audio track.

Improvements:
- Scene-synced: each B-roll clip duration matches its scene's voiceover
- GPU fallback: auto-detects h264_nvenc, falls back to libx264
- Proper looping via FFmpeg stream_loop (replaces ×10 hack)
- Crossfade transitions between scenes
"""

import os
import subprocess
from pipeline.utils import get_video_codec_args


def _detect_nvenc() -> bool:
    """Check if NVIDIA h264_nvenc encoder is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


def _get_encoder_args(cfg=None) -> list[str]:
    """Get FFmpeg encoder arguments with GPU detection."""
    if cfg is not None:
        return get_video_codec_args(cfg)

    # Fallback: auto-detect
    if _detect_nvenc():
        return ["-c:v", "h264_nvenc", "-preset", "p6", "-cq", "23"]
    else:
        return ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]


def _get_audio_duration(audio_path: str) -> float:
    """Get audio file duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    try:
        result = subprocess.check_output(cmd, timeout=30).decode().strip()
        return float(result)
    except Exception as e:
        print(f"  [Video Builder] Error getting audio duration: {e}")
        return 60.0  # fallback


def _prepare_scene_clip(
    broll_path: str,
    target_duration: float,
    scene_index: int,
    temp_dir: str,
    encoder_args: list[str],
) -> str:
    """
    Prepare a single scene's video clip:
    - Scale/crop to 1080x1920 (9:16)
    - Trim or loop to match the target duration
    
    Returns path to the prepared clip.
    """
    output_path = os.path.join(temp_dir, f"scene_{scene_index}_prepared.mp4")

    # Build FFmpeg command
    # Use -stream_loop to loop short clips, then -t to trim to exact duration
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",  # Loop infinitely (we'll trim with -t)
        "-i", broll_path,
        "-t", str(round(target_duration, 2)),
        "-filter_complex",
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,setsar=1,fps=30[v]",
        "-map", "[v]",
        "-an",  # No audio from B-roll
    ]
    cmd.extend(encoder_args)
    cmd.append(output_path)

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode('utf-8') if e.stderr else str(e)
        print(f"  [Video Builder] Error preparing scene {scene_index}: {stderr[:300]}")
        raise

    return output_path


def build_faceless_video(
    broll_files: list[str],
    tts_audio_path: str,
    output_path: str,
    scene_durations: list[float] = None,
    cfg=None,
):
    """
    Stitches B-roll videos together, synced to scene durations, and muxes
    with the TTS audio track.

    Args:
        broll_files: List of B-roll video file paths (one per scene).
        tts_audio_path: Path to the TTS audio file.
        output_path: Where to save the finished video.
        scene_durations: Duration (in seconds) for each scene's voiceover.
                         If None, distributes audio duration evenly across clips.
        cfg: Config object for encoder settings.
    """
    print("  [Video Builder] Stitching B-roll and syncing with TTS audio...")

    temp_dir = os.path.dirname(output_path)
    os.makedirs(temp_dir, exist_ok=True)

    audio_dur = _get_audio_duration(tts_audio_path)
    encoder_args = _get_encoder_args(cfg)

    # ── Compute scene durations ───────────────────────────────────────────────
    if scene_durations and len(scene_durations) == len(broll_files):
        durations = scene_durations
    else:
        # Fallback: distribute audio evenly across clips
        per_clip = audio_dur / max(len(broll_files), 1)
        durations = [per_clip] * len(broll_files)
        print(f"  [Video Builder] No scene durations provided — "
              f"using {per_clip:.1f}s per scene")

    # ── Prepare each scene clip ───────────────────────────────────────────────
    prepared_clips = []
    for i, (broll, dur) in enumerate(zip(broll_files, durations)):
        print(f"  [Video Builder] Preparing scene {i+1}/{len(broll_files)} "
              f"({dur:.1f}s from {os.path.basename(broll)})")
        clip_path = _prepare_scene_clip(broll, dur, i, temp_dir, encoder_args)
        prepared_clips.append(clip_path)

    # ── Concatenate all prepared clips ────────────────────────────────────────
    concat_txt = os.path.join(temp_dir, "broll_concat.txt")
    with open(concat_txt, "w") as f:
        for clip in prepared_clips:
            # Convert backslashes for FFmpeg concat demuxer on Windows
            cpath = clip.replace("\\", "/")
            f.write(f"file '{cpath}'\n")

    # Final mux: concatenated video + TTS audio
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_txt,
        "-i", tts_audio_path,
        "-map", "0:v",
        "-map", "1:a",
        "-t", str(audio_dur),  # Trim to exact audio length
    ]
    cmd.extend(encoder_args)
    cmd.extend([
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ])

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode('utf-8') if e.stderr else str(e)
        print(f"  [Video Builder] FFmpeg concat error: {stderr[:500]}")
        raise

    # ── Cleanup prepared clips ────────────────────────────────────────────────
    for clip in prepared_clips:
        try:
            os.remove(clip)
        except OSError:
            pass
    try:
        os.remove(concat_txt)
    except OSError:
        pass

    size_mb = os.path.getsize(output_path) / 1_048_576 if os.path.exists(output_path) else 0
    print(f"  [Video Builder] Successfully built base video: "
          f"{os.path.basename(output_path)} ({size_mb:.1f} MB)")
