import os
import subprocess

from config import Config
from pipeline.utils import get_video_codec_args


def cut(
    src: str,
    start: float,
    end: float,
    out_path: str,
    use_intro_zoom: bool = False,
    flash_forward_start: float = None,
    flash_forward_dur: float = 3.0,
) -> str:
    """
    Cut a clip from src video between start and end seconds.
    Always re-encodes to ensure frame-accurate cuts so subtitles sync perfectly.
    
    If flash_forward_start is set, extracts a segment starting from that
    timestamp (with duration flash_forward_dur) and concatenates it to the very beginning of the clip as a cold open.
    
    If use_intro_zoom is True, applies a subtle 10% zoom-in over the first 0.5s
    for a visual "pattern interrupt" hook effect (applied after flash-forward if both exist).
    
    Returns out_path on success.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if flash_forward_start is not None:
        # Complex concatenation (intro zoom is handled inside this if needed, 
        # but to keep it simple, we just apply the concat first. 
        # For this version, we will just do a hard cut concat without zoom for the flash forward).
        success = _cut_reencode_flash_forward(src, start, end, flash_forward_start, flash_forward_dur, out_path)
    elif use_intro_zoom:
        success = _cut_reencode_with_zoom(src, start, end, out_path)
    else:
        # Always re-encode for frame-accurate cuts. 
        # Stream copy (-c copy) often cuts at the wrong keyframe, causing audio sync issues.
        success = _cut_reencode(src, start, end, out_path)

    if not success:
        raise RuntimeError(f"Failed to cut clip: {src} [{start}s – {end}s]")

    size_mb = os.path.getsize(out_path) / 1_048_576
    print(f"    Cut: {os.path.basename(out_path)} ({size_mb:.1f} MB)")
    return out_path


def _cut_copy(src, start, end, out_path) -> bool:
    """Stream-copy cut — very fast, may have small keyframe offset."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss",  str(start),
        "-i",   src,
        "-t",   str(duration),
        "-c",   "copy",
        "-avoid_negative_ts", "make_zero",
        out_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and os.path.exists(out_path)


def _cut_reencode(src, start, end, out_path) -> bool:
    """Re-encode cut — frame-accurate, slightly slower."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-i",   src,
        "-ss",  str(start),
        "-t",   str(duration),
        *get_video_codec_args(Config()),
        "-c:a", "aac",
        "-b:a", "128k",
        "-af",  "aresample=async=1",
        "-movflags", "+faststart",
        out_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [FFMPEG ERROR in _cut_reencode]: {result.stderr[-2000:]}")
    return result.returncode == 0 and os.path.exists(out_path)


def _cut_reencode_flash_forward(src, start, end, flash_start, flash_dur, out_path) -> bool:
    """Extract a clip at flash_start and prepend it to [start, end] via filter_complex."""
    import tempfile
    
    # Create two temporary files
    fd_hook, temp_hook = tempfile.mkstemp(suffix=".mp4")
    fd_main, temp_main = tempfile.mkstemp(suffix=".mp4")
    os.close(fd_hook)
    os.close(fd_main)
    
    try:
        # Step 1: Extract the flash-forward hook (frame accurate)
        if not _cut_reencode(src, flash_start, flash_start + flash_dur, temp_hook):
            return _cut_reencode(src, start, end, out_path)
            
        # Step 2: Extract the main clip (frame accurate)
        if not _cut_reencode(src, start, end, temp_main):
            return _cut_reencode(src, start, end, out_path)

        # Step 3: Concat them with effects. Since they are already cut, no -ss needed.
        # [0:v] is the hook, [1:v] is the main clip
        # Adding setsar=1, format=yuv420p, and aformat ensures the two clips perfectly match 
        # so the concat filter doesn't fail due to minor resolution/audio discrepancies.
        filter_complex = (
            "[0:v]setsar=1,format=yuv420p,setpts=PTS-STARTPTS[v0];"
            "[0:a]aformat=sample_rates=44100:channel_layouts=stereo,asetpts=PTS-STARTPTS[a0];"
            "[1:v]fade=t=in:st=0:d=0.3:color=black,setsar=1,format=yuv420p,setpts=PTS-STARTPTS[v1];"
            "[1:a]aformat=sample_rates=44100:channel_layouts=stereo,asetpts=PTS-STARTPTS[a1];"
            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[vout][aout]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", temp_hook,
            "-i", temp_main,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "[aout]",
            *get_video_codec_args(Config()),
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            out_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = f"Flash-forward concat failed:\n{result.stderr[-1000:]}"
            print(f"    ⚠ {error_msg}")
            # Do NOT silently fallback, as that would cause massive subtitle desync in main.py
            raise RuntimeError(error_msg)
            
        return os.path.exists(out_path)
    finally:
        # Cleanup temp files
        if os.path.exists(temp_hook): os.remove(temp_hook)
        if os.path.exists(temp_main): os.remove(temp_main)





def _cut_reencode_with_zoom(src, start, end, out_path) -> bool:
    """
    Re-encode cut with a subtle intro zoom effect.
    
    Zooms from 110% to 100% over the first 0.5 seconds, creating a visual
    "punch-in" that serves as the hook's pattern-interrupt first frame.
    Uses FFmpeg's zoompan filter with keyframe expressions.
    """
    duration = end - start
    fps = 30  # Assumed FPS for zoom calculation
    zoom_frames = int(0.5 * fps)  # 0.5 seconds worth of frames

    # zoompan expression:
    # - First 0.5s: zoom from 1.10 → 1.00 (ease-out)
    # - Remaining: zoom stays at 1.00
    # 'on' = current frame number, 'zoom' = current zoom level
    zoom_expr = (
        f"if(lt(on,{zoom_frames}),"
        f"1.10-0.10*(on/{zoom_frames}),"
        f"1.0)"
    )

    # x/y expressions to keep the zoom centered
    x_expr = "iw/2-(iw/zoom/2)"
    y_expr = "ih/2-(ih/zoom/2)"

    vf = (
        f"zoompan=z='{zoom_expr}'"
        f":x='{x_expr}':y='{y_expr}'"
        f":d=1:s=iw*2:fps={fps},"
        f"scale=-2:ih"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i",   src,
        "-ss",  str(start),
        "-t",   str(duration),
        "-vf",  vf,
        *get_video_codec_args(Config()),
        "-c:a", "aac",
        "-b:a", "128k",
        "-af",  "aresample=async=1",
        "-movflags", "+faststart",
        out_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Fallback: if zoompan fails (codec issues), try without zoom
    if result.returncode != 0:
        print("    ⚠ Intro zoom failed, cutting without zoom...")
        return _cut_reencode(src, start, end, out_path)

    return os.path.exists(out_path)


def _file_too_small(path: str, min_bytes: int = 50_000) -> bool:
    """Return True if file is suspiciously small (bad cut)."""
    try:
        return os.path.getsize(path) < min_bytes
    except OSError:
        return True

