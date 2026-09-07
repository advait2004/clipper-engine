"""
Smart Assembler — execute the AI's edit plan
=============================================
Takes an Edit Decision List (EDL) and:
  1. Cuts each segment from its source file
  2. Applies transitions between segments (FFmpeg xfade)
  3. Concatenates into a single assembled video
  4. Tracks word timestamps across the assembled timeline
"""

import os
import subprocess
import tempfile

from pipeline.clipper import cut
from pipeline.transitions import get_transition, TRANSITIONS
from pipeline.utils import get_video_codec_args


def assemble(
    edl: list[dict],
    profiles: list[dict],
    output_path: str,
    temp_dir: str,
    cfg=None,
) -> tuple[str, list[dict]]:
    """
    Assemble segments per the edit plan.
    
    Args:
        edl: Edit decision list from story_planner
        profiles: Content profiles with words and paths
        output_path: Where to write the assembled video
        temp_dir: Directory for intermediate files
        cfg: Config object (used for reframing segments)
    
    Returns:
        (output_path, merged_words) where merged_words is the
        word list with timestamps adjusted for the assembled timeline.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    if not edl:
        raise ValueError("Empty edit decision list")

    # Build file lookup
    file_lookup = {p["file"]: p for p in profiles}

    # ── 1. Cut and Reframe each segment ───────────────────────────────────────
    print(f"\n  ── Assembling {len(edl)} segments ──")
    segment_paths = []
    segment_durations = []

    # Local import to avoid circular dependencies
    from pipeline import reframer

    for i, seg in enumerate(edl):
        profile = file_lookup.get(seg["source_file"])
        if not profile:
            print(f"    ⚠ Source not found: {seg['source_file']}, skipping")
            continue

        seg_path = os.path.join(temp_dir, f"seg_{i:03d}.mp4")
        rf_path = os.path.join(temp_dir, f"rf_{i:03d}.mp4")
        try:
            cut(profile["path"], seg["start"], seg["end"], seg_path)
            # Reframe individual segment BEFORE assembly to preserve face tracking
            reframer.reframe(seg_path, rf_path, cfg=cfg)
            duration = seg["end"] - seg["start"]
            segment_paths.append(rf_path)
            segment_durations.append(duration)
        except Exception as e:
            print(f"    ⚠ Failed to cut segment {i+1}: {e}")

    if not segment_paths:
        raise RuntimeError("No segments could be cut")

    # ── 2. Apply transitions and concatenate ──────────────────────────────────
    transition_types = [
        seg.get("transition", "cut") for seg in edl
    ][1:]  # First segment has no transition

    if len(segment_paths) == 1:
        # Single segment — just copy
        import shutil
        shutil.copy2(segment_paths[0], output_path)
    elif _all_hard_cuts(transition_types):
        # All hard cuts — use fast concat demuxer
        _concat_segments(segment_paths, output_path)
    else:
        # Has transitions — use xfade filter_complex
        _xfade_segments(segment_paths, segment_durations, transition_types, output_path)

    # ── 3. Build merged word timestamps ───────────────────────────────────────
    merged_words = _merge_word_timestamps(edl, profiles, segment_durations, transition_types)

    # ── 4. Cleanup segment files ──────────────────────────────────────────────
    for path in segment_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / 1_048_576
        print(f"  ✅ Assembled: {os.path.basename(output_path)} ({size_mb:.1f} MB)")
    else:
        raise RuntimeError("Assembly failed: output not created")

    return output_path, merged_words


def _all_hard_cuts(transition_types: list[str]) -> bool:
    """Check if all transitions are hard cuts."""
    return all(t in ("cut", "none") for t in transition_types)


def _concat_segments(segment_paths: list[str], output_path: str):
    """Fast concat using concat demuxer (no re-encoding)."""
    concat_list = tempfile.mktemp(suffix=".txt")
    try:
        with open(concat_list, "w", encoding="utf-8") as f:
            for path in segment_paths:
                safe = path.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{safe}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Fallback to re-encode concat
            _reencode_concat(segment_paths, output_path)
    finally:
        if os.path.exists(concat_list):
            os.remove(concat_list)


def _reencode_concat(segment_paths: list[str], output_path: str):
    """Re-encode concat fallback with robust input normalization."""
    from pipeline.utils import get_video_info

    # Query first segment to check if audio is present
    try:
        first_info = get_video_info(segment_paths[0])
        has_audio = first_info.get("has_audio", True)
    except Exception:
        has_audio = True

    inputs = []
    video_filters = []
    audio_filters = []

    for i, path in enumerate(segment_paths):
        inputs.extend(["-i", path])
        # Normalize video: scale to 1080x1920, pad, unify to 30fps, format yuv420p, settb AVTB
        video_filters.append(
            f"[{i}:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
            f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2,fps=30,format=yuv420p,settb=AVTB[normv{i}]"
        )
        if has_audio:
            # Normalize audio: sample rate 44100Hz, stereo layout
            audio_filters.append(
                f"[{i}:a]aformat=sample_rates=44100:channel_layouts=stereo[norma{i}]"
            )

    n = len(segment_paths)
    filter_parts_v = [f"[normv{i}]" for i in range(n)]

    if has_audio:
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
            *get_video_codec_args(cfg),
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
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
            *get_video_codec_args(cfg),
            "-movflags", "+faststart",
            output_path,
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Re-encode concat failed:\n{result.stderr[-500:]}")


def _xfade_segments(
    segment_paths: list[str],
    segment_durations: list[float],
    transition_types: list[str],
    output_path: str,
):
    """
    Assemble segments with xfade transitions using FFmpeg filter_complex.
    
    Uses a sequential chaining approach:
      [0:v][1:v]xfade=...[xf0]; [xf0][2:v]xfade=...[xf1]; ...
    """
    n = len(segment_paths)
    if n < 2:
        raise ValueError("Need at least 2 segments for xfade")

    # Build inputs
    inputs = []
    video_filters = []
    audio_filters = []
    for i, path in enumerate(segment_paths):
        inputs.extend(["-i", path])
        # Normalize video: 1080x1920 (padded), 30fps, yuv420p, common timebase
        video_filters.append(
            f"[{i}:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
            f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2,fps=30,format=yuv420p,settb=AVTB[normv{i}]"
        )
        # Normalize audio: 44100Hz, stereo
        audio_filters.append(
            f"[{i}:a]aformat=sample_rates=44100:channel_layouts=stereo[norma{i}]"
        )

    cumulative_offset = 0.0

    for i in range(n - 1):
        trans_type = transition_types[i] if i < len(transition_types) else "crossfade"
        trans_cfg = get_transition(trans_type)
        trans_dur = trans_cfg["duration"]

        if trans_cfg["xfade_type"] is None:
            trans_dur = 0.0

        # Use normalized inputs for the first segment
        v_in_a = f"[normv0]" if i == 0 else f"[xfv{i-1}]"
        v_in_b = f"[normv{i+1}]"
        v_out = f"[xfv{i}]"

        a_in_a = f"[norma0]" if i == 0 else f"[xfa{i-1}]"
        a_in_b = f"[norma{i+1}]"
        a_out = f"[xfa{i}]"

        offset = cumulative_offset + segment_durations[i] - trans_dur
        offset = max(0, offset)

        if trans_cfg["xfade_type"] is not None:
            # xfade transition
            video_filters.append(
                f"{v_in_a}{v_in_b}xfade=transition={trans_cfg['xfade_type']}"
                f":duration={trans_dur}:offset={offset:.3f}{v_out}"
            )
            audio_filters.append(
                f"{a_in_a}{a_in_b}acrossfade=d={trans_dur}:c1=tri:c2=tri{a_out}"
            )
        else:
            # Hard cut (concat)
            video_filters.append(
                f"{v_in_a}{v_in_b}concat=n=2:v=1:a=0{v_out}"
            )
            audio_filters.append(
                f"{a_in_a}{a_in_b}concat=n=2:v=0:a=1{a_out}"
            )

        cumulative_offset += segment_durations[i] - trans_dur

    # Final labels
    final_v = f"[xfv{n-2}]"
    final_a = f"[xfa{n-2}]"

    filter_complex = ";".join(video_filters + audio_filters)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", final_v,
        "-map", final_a,
        *get_video_codec_args(cfg),
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]

    print(f"    Applying transitions: {', '.join(transition_types)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ⚠ xfade failed, falling back to concat: {result.stderr[-300:]}")
        _reencode_concat(segment_paths, output_path)


def _merge_word_timestamps(
    edl: list[dict],
    profiles: list[dict],
    segment_durations: list[float],
    transition_types: list[str],
) -> list[dict]:
    """
    Map word timestamps from individual clips into the assembled timeline.
    
    For each segment in the EDL, find words that fall within [start, end]
    of the source clip, then offset them to the assembled timeline position.
    """
    file_lookup = {p["file"]: p for p in profiles}
    merged = []
    timeline_offset = 0.0

    for i, seg in enumerate(edl):
        profile = file_lookup.get(seg["source_file"])
        if not profile:
            continue

        seg_start = seg["start"]
        seg_end = seg["end"]

        # Find words within this segment's range
        for w in profile.get("words", []):
            if w["start"] >= seg_start and w["end"] <= seg_end:
                merged.append({
                    "word": w["word"],
                    "start": round(timeline_offset + (w["start"] - seg_start), 3),
                    "end": round(timeline_offset + (w["end"] - seg_start), 3),
                })

        # Advance timeline offset
        seg_dur = segment_durations[i] if i < len(segment_durations) else (seg_end - seg_start)
        trans_dur = 0.0
        if i < len(transition_types):
            trans_cfg = get_transition(transition_types[i])
            trans_dur = trans_cfg["duration"]

        timeline_offset += seg_dur - trans_dur

    print(f"  Merged {len(merged)} words into assembled timeline")
    return merged
