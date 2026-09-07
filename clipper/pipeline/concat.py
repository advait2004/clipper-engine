"""
Concatenate multiple video files into a single .mp4 using FFmpeg's concat demuxer.
"""
import os
import subprocess
import tempfile

from config import Config
from pipeline.utils import get_video_codec_args

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".ts"}


def find_videos_in_folder(folder_path: str) -> list[str]:
    """
    Scan a folder recursively for video files and return sorted list of absolute paths.
    Files are sorted alphabetically by path for predictable merge order.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    videos = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                videos.append(os.path.join(root, file))

    return sorted(videos)


def concatenate_videos(video_paths: list[str], output_path: str) -> str:
    """
    Concatenate a list of video files into a single .mp4.
    Uses FFmpeg concat demuxer (fast, no re-encoding when codecs match).
    Falls back to re-encoding if concat demuxer fails.
    Returns the output file path.
    """
    if not video_paths:
        raise ValueError("No video files to concatenate")

    if len(video_paths) == 1:
        # Single file — just copy it
        import shutil
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        shutil.copy2(video_paths[0], output_path)
        print(f"  Concat: single file, copied as-is")
        return output_path

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Build concat list file
    concat_list = tempfile.mktemp(suffix=".txt")
    try:
        with open(concat_list, "w", encoding="utf-8") as f:
            for path in video_paths:
                # FFmpeg concat demuxer requires forward slashes and escaped quotes
                safe_path = path.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        print(f"  Concatenating {len(video_paths)} videos...")
        for i, p in enumerate(video_paths, 1):
            print(f"    {i}. {os.path.basename(p)}")

        # Try fast concat demuxer first (no re-encoding)
        success = _try_concat_demuxer(concat_list, output_path)

        if not success:
            # Fallback: re-encode all inputs
            print("  Concat demuxer failed, falling back to re-encode...")
            _concat_reencode(video_paths, output_path)

    finally:
        if os.path.exists(concat_list):
            os.remove(concat_list)

    if not os.path.exists(output_path):
        raise RuntimeError("Concatenation failed: output file not created")

    size_mb = os.path.getsize(output_path) / 1_048_576
    print(f"  Concatenated: {os.path.basename(output_path)} ({size_mb:.1f} MB)")
    return output_path


def _try_concat_demuxer(concat_list: str, output_path: str) -> bool:
    """Try the fast concat demuxer (no re-encoding). Returns True on success."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list,
        "-c", "copy",
        "-movflags", "+faststart",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def _concat_reencode(video_paths: list[str], output_path: str):
    """Fallback: re-encode all inputs to ensure compatible codec/resolution."""
    from pipeline.utils import get_video_info

    # Gather info for all input videos to determine target dimensions & audio status
    video_infos = []
    for path in video_paths:
        try:
            info = get_video_info(path)
        except Exception:
            info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0, "has_audio": True}
        video_infos.append(info)

    # Use first video's resolution and fps as target
    target_w = video_infos[0].get("width", 1920)
    target_h = video_infos[0].get("height", 1080)
    target_fps = video_infos[0].get("fps", 30.0)

    # Ensure even dimensions for libx264
    target_w = target_w if target_w % 2 == 0 else target_w - 1
    target_h = target_h if target_h % 2 == 0 else target_h - 1

    # Check if ANY video has audio. If none of the videos have audio, do video-only concat.
    any_audio = any(info.get("has_audio", True) for info in video_infos)

    inputs = []
    video_filters = []
    audio_filters = []

    for i, path in enumerate(video_paths):
        inputs.extend(["-i", path])
        info = video_infos[i]
        duration = info.get("duration", 10.0)
        has_aud = info.get("has_audio", True)

        # Normalize video stream
        video_filters.append(
            f"[{i}:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,fps={target_fps},format=yuv420p,settb=AVTB[normv{i}]"
        )

        # Normalize audio stream if we are outputting audio
        if any_audio:
            if has_aud:
                audio_filters.append(
                    f"[{i}:a]aformat=sample_rates=44100:channel_layouts=stereo[norma{i}]"
                )
            else:
                # Generate silent audio matching this video's duration
                audio_filters.append(
                    f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={duration}[norma{i}]"
                )

    n = len(video_paths)
    filter_parts_v = [f"[normv{i}]" for i in range(n)]

    if any_audio:
        filter_parts_a = [f"[norma{i}]" for i in range(n)]
        filter_str = (
            ";".join(video_filters + audio_filters) + ";" +
            "".join(filter_parts_v) + "".join(filter_parts_a) + f"concat=n={n}:v=1:a=1[outv][outa]"
        )
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_str,
            "-map", "[outv]", "-map", "[outa]",
            *get_video_codec_args(Config()),
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output_path
        ]
    else:
        filter_str = (
            ";".join(video_filters) + ";" +
            "".join(filter_parts_v) + f"concat=n={n}:v=1:a=0[outv]"
        )
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_str,
            "-map", "[outv]",
            *get_video_codec_args(Config()),
            "-movflags", "+faststart",
            output_path
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Concatenation re-encode failed:\n{result.stderr[-500:]}")
