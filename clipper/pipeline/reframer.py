import os
import subprocess
import shutil
import tempfile

import cv2
import numpy as np

from pipeline.utils import get_video_info, get_video_codec_args
from config import Config

# Advanced reframing imports (lazy-loaded when features are enabled)
_face_analyzer_mod = None
_face_scorer_mod = None
_virtual_camera_mod = None
_scene_classifier_mod = None
_speaker_diarizer_mod = None


def _lazy_import_advanced():
    """Lazy-import advanced modules to avoid loading them when disabled."""
    global _face_analyzer_mod, _face_scorer_mod, _virtual_camera_mod
    global _scene_classifier_mod, _speaker_diarizer_mod
    if _face_analyzer_mod is None:
        from pipeline import face_analyzer as _fa
        from pipeline import face_scorer as _fs
        from pipeline import virtual_camera as _vc
        from pipeline import scene_classifier as _sc
        from pipeline import speaker_diarizer as _sd
        _face_analyzer_mod = _fa
        _face_scorer_mod = _fs
        _virtual_camera_mod = _vc
        _scene_classifier_mod = _sc
        _speaker_diarizer_mod = _sd
    return (_face_analyzer_mod, _face_scorer_mod, _virtual_camera_mod,
            _scene_classifier_mod, _speaker_diarizer_mod)


# ─── Sampling & threshold constants ──────────────────────────────────────────

# Sample interval in seconds (0.01 = 100 FPS → ±10ms precision)
_SAMPLE_INTERVAL = 0.04  # seconds (25 FPS) per user request

# Minimum horizontal separation (normalized 0–1) for split-screen
_MIN_SPLIT_SEPARATION = 0.12

# Minimum duration (seconds) a layout must hold to survive temporal smoothing
_MIN_SEGMENT_SECONDS = 1.5
_MIN_SEGMENT_ENTRIES = int(_MIN_SEGMENT_SECONDS / _SAMPLE_INTERVAL)

# EMA smoothing factor (lower = smoother/laggier, higher = more responsive/jittery)
_EMA_ALPHA = 0.3

# Keyframes per second in FFmpeg crop expressions (1 = smooth enough, keeps expr small)
_KEYFRAMES_PER_SECOND = 1

# Minimum face movement (fraction of frame width) to use animated vs static crop
_MIN_MOVEMENT_FRAC = 0.03


def reframe(src: str, out_path: str, cfg=None) -> str:
    """
    Reframe video to 9:16 portrait aspect ratio.

    When advanced features are enabled (via cfg), uses:
    - Face Mesh 478-landmark analysis for lip/gaze/frontality
    - Speaker tracking via lip movement detection
    - Audio speaker diarization (if pyannote installed)
    - Cinematic ease-in-out camera panning
    - Dynamic split screen with speaker emphasis
    - Scene type classification
    - Ken Burns effect for b-roll segments

    Falls back to the original EMA-smoothed face detection pipeline
    when advanced features are disabled or unavailable.

    Returns out_path on success.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    info     = get_video_info(src)
    duration = info.get("duration", 0.0)

    cap    = cv2.VideoCapture(src)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if width <= 0 or height <= 0:
        width  = info.get("width", 1920)
        height = info.get("height", 1080)

    # Already portrait (or close to 9:16)?  Just copy.
    current_ratio = width / height if height else 1.0
    target_ratio  = 9 / 16
    if current_ratio <= target_ratio * 1.1:
        shutil.copy2(src, out_path)
        print("    Reframe: already portrait, copied as-is")
        return out_path

    target_width  = int(height * 9 / 16)
    target_width  = target_width  if target_width  % 2 == 0 else target_width  - 1
    target_height = height if height % 2 == 0 else height - 1

    if duration <= 0:
        duration = _get_duration_cv2(src)

    if duration <= 0:
        print("    Reframe: Could not determine duration, using centre crop")
        crop_x = int(width / 2.0 - target_width / 2)
        crop_x = max(0, min(crop_x, width - target_width))
        _ffmpeg_crop(src, out_path, crop_x, target_width, target_height)
        return out_path

    # ── Decide whether to use advanced or legacy pipeline ─────────────────
    use_advanced = (
        cfg is not None
        and getattr(cfg, "FACE_MESH_ENABLED", False)
    )

    if use_advanced:
        try:
            return _reframe_advanced(
                src, out_path, cfg, duration,
                width, height, target_width, target_height,
            )
        except Exception as e:
            print(f"    Reframe: Advanced pipeline failed ({e}), falling back to legacy")
            import traceback; traceback.print_exc()

    # ── Legacy pipeline (original behavior) ───────────────────────────────
    print(f"    Reframe: Analyzing layout at {1/_SAMPLE_INTERVAL:.0f} FPS "
          f"for {duration:.1f}s clip ...")
    timeline = _analyze_layout_timeline(src, duration)
    segments = _chunk_segments(timeline)

    if not segments:
        print("    Reframe: No valid segments, using centre crop")
        crop_x = int(width / 2.0 - target_width / 2)
        crop_x = max(0, min(crop_x, width - target_width))
        _ffmpeg_crop(src, out_path, crop_x, target_width, target_height)
        return out_path

    print(f"    Reframe: {len(segments)} segment(s) after chunking")

    if len(segments) == 1:
        _process_segment(src, out_path, segments[0],
                         width, height, target_width, target_height)
    else:
        temp_dir = tempfile.mkdtemp(prefix="reframe_")
        try:
            sub_clips = []
            for i, seg in enumerate(segments):
                sub_out = os.path.join(temp_dir, f"seg_{i}.ts")
                print(f"    Reframe: Segment {i+1}/{len(segments)} "
                      f"({seg['start']:.3f}s - {seg['end']:.3f}s) -> {seg['type']}")
                _process_segment(
                    src, sub_out, seg, width, height,
                    target_width, target_height,
                    start_time=seg["start"],
                    duration=seg["end"] - seg["start"],
                )
                sub_clips.append(sub_out)

            print(f"    Reframe: Concatenating {len(sub_clips)} segments ...")
            concat_list = os.path.join(temp_dir, "concat.txt")
            with open(concat_list, "w") as f:
                for clip in sub_clips:
                    f.write(f"file '{clip.replace(chr(92), '/')}'\n")

            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                "-bsf:a", "aac_adtstoasc",
                out_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg concat failed:\n{result.stderr[-500:]}"
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return out_path


# ─── Advanced reframing pipeline ─────────────────────────────────────────────

def _reframe_advanced(
    src: str,
    out_path: str,
    cfg,
    duration: float,
    width: int,
    height: int,
    target_width: int,
    target_height: int,
) -> str:
    """
    Advanced reframing pipeline using Face Mesh, speaker tracking,
    cinematic panning, scene classification, and dynamic split screen.
    """
    fa_mod, fs_mod, vc_mod, sc_mod, sd_mod = _lazy_import_advanced()

    print(f"    Reframe [Advanced]: Analyzing with Face Mesh at "
          f"{1/_SAMPLE_INTERVAL:.0f} FPS for {duration:.1f}s clip ...")

    # ── 1. Face Mesh analysis ─────────────────────────────────────────────
    analyzer = fa_mod.FaceAnalyzer(max_faces=4)
    face_timeline = []

    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    current_time = 0.0
    frame_count = 0
    total_frames = int(duration * fps) if duration > 0 else float("inf")

    while True:
        ret, frame = cap.read()
        if not ret or frame_count > total_frames:
            break

        t_sec = frame_count / fps
        if t_sec >= current_time:
            faces = analyzer.analyze_frame(frame)

            # Score each face
            if faces:
                fs_mod.rank_faces(faces)

            # Determine primary speaker
            primary_speaker_id = -1
            for face in faces:
                if face.is_speaking:
                    primary_speaker_id = face.face_id
                    break
            # If no one is speaking, use highest-scoring face
            if primary_speaker_id == -1 and faces:
                primary_speaker_id = faces[0].face_id

            face_timeline.append({
                "time": round(t_sec, 3),
                "faces": faces,
                "primary_speaker_id": primary_speaker_id,
            })

            current_time = t_sec + _SAMPLE_INTERVAL
            if current_time >= duration and duration > 0:
                break

        frame_count += 1

    cap.release()
    analyzer.close()

    if not face_timeline:
        print("    Reframe [Advanced]: No frames analyzed, centre crop")
        crop_x = int(width / 2.0 - target_width / 2)
        _ffmpeg_crop(src, out_path, crop_x, target_width, target_height)
        return out_path

    # ── 2. Audio diarization (optional) ───────────────────────────────────
    diarization_segments = []
    if getattr(cfg, "SPEAKER_DIARIZATION", False):
        print("    Reframe [Advanced]: Running audio speaker diarization...")
        hf_token = getattr(cfg, "HF_TOKEN", "")
        diarization_segments = sd_mod.diarize(src, hf_token)

    # ── 3. Scene classification ───────────────────────────────────────────
    if getattr(cfg, "SCENE_CLASSIFICATION", True):
        scene_type = sc_mod.classify_scene(face_timeline)
    else:
        scene_type = "talking_head"

    strategy_desc = sc_mod.get_strategy_description(scene_type)
    print(f"    Reframe [Advanced]: Scene type: {scene_type} → {strategy_desc}")

    # ── 4. Route to the appropriate strategy ──────────────────────────────
    if scene_type == sc_mod.SCENE_BROLL:
        return _process_broll(
            src, out_path, cfg, duration,
            width, height, target_width, target_height,
        )

    if scene_type == sc_mod.SCENE_INTERVIEW:
        mode = getattr(cfg, "INTERVIEW_LAYOUT_MODE", "snap_split")
        if mode == "snap_split":
            print("    Reframe [Advanced]: Score-Split Mode: Dynamic split with reaction cuts")
            segments = []
            current_type = None
            seg_start = 0.0

            # Apply EMA (Exponential Moving Average) to scores for smoothing (Hysteresis)
            smoothed_scores: dict[int, float] = {}
            EMA_ALPHA = 0.2  # 2-second rolling window equivalent at 100ms intervals

            for entry in face_timeline:
                t = entry["time"]
                faces = entry.get("faces", [])
                
                highest_score = 0.0
                second_highest = 0.0
                
                for f in faces:
                    fid = f.face_id
                    raw_score = getattr(f, "subject_score", 0.0)
                    
                    # Update EMA
                    prev = smoothed_scores.get(fid, raw_score)
                    smoothed = EMA_ALPHA * raw_score + (1.0 - EMA_ALPHA) * prev
                    smoothed_scores[fid] = smoothed
                    
                    if smoothed > highest_score:
                        second_highest = highest_score
                        highest_score = smoothed
                    elif smoothed > second_highest:
                        second_highest = smoothed

                # Reaction Cut Logic: 
                # If the highest score is massive (>0.85) AND significantly beats the second person, cut to single.
                # Otherwise, maintain the dynamic split.
                if highest_score > 0.85 and highest_score > (second_highest * 1.5):
                    layout_type = "single"
                else:
                    layout_type = "split"
                
                if current_type is None:
                    current_type = layout_type
                
                if layout_type != current_type:
                    # Prevent erratic micro-cuts (must hold for at least 1.5 seconds)
                    if t - seg_start > 1.5:
                        segments.append({"start": seg_start, "end": t, "type": current_type})
                        seg_start = t
                        current_type = layout_type

            if current_type is not None:
                segments.append({"start": seg_start, "end": duration, "type": current_type})
                
            if not segments:
                segments = [{"start": 0.0, "end": duration, "type": "split"}]

            # Render segments and concatenate
            if len(segments) == 1:
                if segments[0]["type"] == "single":
                    return _process_cinematic_tracking(src, out_path, cfg, face_timeline, diarization_segments, duration, width, height, target_width, target_height)
                else:
                    return _process_dynamic_split(src, out_path, cfg, face_timeline, diarization_segments, duration, width, height, target_width, target_height)
            
            import tempfile
            import shutil
            temp_dir = tempfile.mkdtemp(prefix="snap_split_")
            try:
                sub_clips = []
                for i, seg in enumerate(segments):
                    sub_out = os.path.join(temp_dir, f"seg_{i}.ts")
                    seg_dur = seg["end"] - seg["start"]
                    print(f"    Reframe [Advanced]: Snap-Split Segment {i+1}/{len(segments)} ({seg['start']:.3f}s - {seg['end']:.3f}s) -> {seg['type']}")
                    if seg["type"] == "single":
                        _process_cinematic_tracking(src, sub_out, cfg, face_timeline, diarization_segments, seg_dur, width, height, target_width, target_height, start_time=seg["start"])
                    else:
                        _process_dynamic_split(src, sub_out, cfg, face_timeline, diarization_segments, seg_dur, width, height, target_width, target_height, start_time=seg["start"])
                    sub_clips.append(sub_out)

                print(f"    Reframe [Advanced]: Concatenating {len(sub_clips)} snap-split segments ...")
                concat_list = os.path.join(temp_dir, "concat.txt")
                with open(concat_list, "w") as f:
                    for clip in sub_clips:
                        f.write(f"file '{clip.replace(chr(92), '/')}'\n")

                cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", concat_list,
                    "-c", "copy",
                    "-bsf:a", "aac_adtstoasc",
                    out_path,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg concat failed:\n{result.stderr[-500:]}")
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

            return out_path
        elif mode == "single":
            print("    Reframe [Advanced]: Interview mode: single-camera cuts")
            return _process_cinematic_tracking(
                src, out_path, cfg, face_timeline, diarization_segments,
                duration, width, height, target_width, target_height,
            )
        elif mode == "hybrid":
            # Hybrid Broadcast Mode: analyze speech dominance and banter
            speaker_counts = {}
            switches = 0
            last_spk = -1
            for entry in face_timeline:
                spk = entry.get("primary_speaker_id", -1)
                if spk != -1:
                    speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
                    if last_spk != -1 and spk != last_spk:
                        switches += 1
                    last_spk = spk
            
            total_speech = sum(speaker_counts.values())
            max_share = (max(speaker_counts.values()) / total_speech) if total_speech > 0 else 0
            
            if total_speech > 0 and (max_share > 0.68 or switches <= 1):
                print(f"    Reframe [Advanced]: Hybrid Mode: Monologue ({max_share*100:.0f}% dominance) → 100% Full-screen cut")
                return _process_cinematic_tracking(
                    src, out_path, cfg, face_timeline, diarization_segments,
                    duration, width, height, target_width, target_height,
                )
            else:
                print(f"    Reframe [Advanced]: Hybrid Mode: Banter ({switches} switches) → Dynamic split screen")
        
        if getattr(cfg, "DYNAMIC_SPLIT_SCREEN", True):
            return _process_dynamic_split(
                src, out_path, cfg, face_timeline, diarization_segments,
                duration, width, height, target_width, target_height,
            )

    if scene_type == sc_mod.SCENE_PANEL:
        # Panel: use letterbox to show all faces
        print("    Reframe [Advanced]: Panel scene → letterbox")
        _ffmpeg_letterbox(src, out_path, width, height,
                          target_width, target_height)
        return out_path

    # talking_head / presentation → cinematic single-face tracking
    return _process_cinematic_tracking(
        src, out_path, cfg, face_timeline, diarization_segments,
        duration, width, height, target_width, target_height,
    )


def _enforce_bbox_enclosure(cx_pixel: float, t: float, face_timeline: list, width: int, target_width: int) -> float:
    """
    Ensure that the crop window centered at cx_pixel completely encloses
    the primary speaker's face bounding box with a 4% safety margin.
    """
    if not face_timeline:
        return cx_pixel
    entry = min(face_timeline, key=lambda e: abs(e.get("time", 0.0) - t))
    faces = entry.get("faces", [])
    if not faces:
        return cx_pixel
    spk_id = entry.get("primary_speaker_id", -1)
    target_face = None
    for f in faces:
        if f.face_id == spk_id:
            target_face = f
            break
    if target_face is None:
        target_face = faces[0]

    margin = width * 0.04
    left_edge = target_face.bbox_x * width
    right_edge = (target_face.bbox_x + target_face.bbox_w) * width

    crop_x = cx_pixel - target_width / 2.0
    if right_edge - left_edge + 2 * margin <= target_width:
        if crop_x > left_edge - margin:
            crop_x = left_edge - margin
        if crop_x + target_width < right_edge + margin:
            crop_x = right_edge + margin - target_width

    crop_x = max(0.0, min(crop_x, float(width - target_width)))
    return crop_x + target_width / 2.0


def _process_cinematic_tracking(
    src: str,
    out_path: str,
    cfg,
    face_timeline: list,
    diarization_segments: list,
    duration: float,
    width: int,
    height: int,
    target_width: int,
    target_height: int,
    start_time: float = None,
) -> str:
    """
    Single-face tracking with cinematic ease-in-out panning.
    Used for talking_head and presentation scenes.
    """
    _, _, vc_mod, _, sd_mod = _lazy_import_advanced()

    # Build speaker timeline for the virtual camera
    speaker_tl = vc_mod.build_speaker_timeline_from_faces(face_timeline)

    # Fuse with audio diarization if available
    if diarization_segments and getattr(cfg, "SPEAKER_DIARIZATION", False):
        _fuse_audio_into_timeline(speaker_tl, diarization_segments, face_timeline, sd_mod)

    # Create virtual camera with config parameters
    if getattr(cfg, "CINEMATIC_PANNING", True):
        camera = vc_mod.VirtualCamera(
            hold_delay=getattr(cfg, "PAN_HOLD_DELAY", 0.3),
            max_velocity=getattr(cfg, "PAN_MAX_VELOCITY", 0.30),
            min_pan_duration=getattr(cfg, "PAN_MIN_DURATION", 0.4),
            max_pan_duration=getattr(cfg, "PAN_MAX_DURATION", 1.5),
        )
        keyframes = camera.compute_keyframes(speaker_tl, _SAMPLE_INTERVAL)

        # Convert to pixel keyframes for FFmpeg with bounding box enclosure clamping
        pixel_kf = [
            (kf.time, _enforce_bbox_enclosure(kf.crop_cx * width, kf.time, face_timeline, width, target_width))
            for kf in keyframes
        ]

        print(f"    Reframe [Advanced]: Cinematic tracking with "
              f"{len(pixel_kf)} keyframes")

        if len(pixel_kf) > 1:
            _ffmpeg_crop_tracked(
                src, out_path, pixel_kf, width,
                target_width, target_height,
                start_time=start_time, duration=duration,
            )
        else:
            cx = pixel_kf[0][1] if pixel_kf else width / 2
            crop_x = int(cx - target_width / 2)
            crop_x = max(0, min(crop_x, width - target_width))
            _ffmpeg_crop(src, out_path, crop_x, target_width, target_height,
                         start_time=start_time, duration=duration)
    else:
        # No cinematic panning — use simple EMA tracking with bounding box enclosure
        avg_cx = np.mean([
            entry["faces"][0].norm_cx
            for entry in face_timeline
            if entry.get("faces")
        ]) if any(e.get("faces") for e in face_timeline) else 0.5

        cx_pixel = _enforce_bbox_enclosure(avg_cx * width, 0.0, face_timeline, width, target_width)
        crop_x = int(cx_pixel - target_width / 2)
        crop_x = max(0, min(crop_x, width - target_width))
        _ffmpeg_crop(src, out_path, crop_x, target_width, target_height)

    return out_path


def _process_dynamic_split(
    src: str,
    out_path: str,
    cfg,
    face_timeline: list,
    diarization_segments: list,
    duration: float,
    width: int,
    height: int,
    target_width: int,
    target_height: int,
    start_time: float = None,
) -> str:
    """
    Dynamic split screen where the active speaker gets more screen space.
    The split boundary smoothly shifts based on who is talking.
    """
    print("    Reframe [Advanced]: Dynamic split screen")

    active_ratio = getattr(cfg, "SPLIT_ACTIVE_RATIO", 0.60)
    divider_h = 4

    # Identify the two primary faces
    face_ids_count: dict[int, int] = {}
    face_avg_cx: dict[int, list[float]] = {}
    for entry in face_timeline:
        for face in entry.get("faces", []):
            fid = face.face_id
            face_ids_count[fid] = face_ids_count.get(fid, 0) + 1
            if fid not in face_avg_cx:
                face_avg_cx[fid] = []
            face_avg_cx[fid].append(face.norm_cx)

    # Pick the two most-seen faces
    top_faces = sorted(face_ids_count, key=face_ids_count.get, reverse=True)[:2]
    if len(top_faces) < 2:
        # Fall back to letterbox if we can't find 2 faces
        print("    Reframe [Advanced]: Can't find 2 faces for split, using letterbox")
        _ffmpeg_letterbox(src, out_path, width, height,
                          target_width, target_height)
        return out_path

    face_a, face_b = top_faces[0], top_faces[1]
    cx_a = float(np.mean(face_avg_cx[face_a]))
    cx_b = float(np.mean(face_avg_cx[face_b]))

    # Determine which face goes on top (leftmost face on top)
    if cx_a <= cx_b:
        top_id, bot_id = face_a, face_b
        top_cx, bot_cx = cx_a, cx_b
    else:
        top_id, bot_id = face_b, face_a
        top_cx, bot_cx = cx_b, cx_a

    # Build speaker-emphasis keyframes
    # Each keyframe: (time, top_ratio) where top_ratio is the fraction of
    # vertical space allocated to the top face
    split_keyframes = []
    prev_ratio = 0.5
    ema_ratio = 0.5
    ema_alpha = 0.15  # Slow blend for smooth split transitions

    for entry in face_timeline:
        t = entry["time"]
        speaker_id = entry.get("primary_speaker_id", -1)

        # Determine target ratio
        if speaker_id == top_id:
            target_ratio = active_ratio
        elif speaker_id == bot_id:
            target_ratio = 1.0 - active_ratio
        else:
            target_ratio = 0.5  # Neither or both speaking → equal

        # EMA smooth the ratio transition
        ema_ratio = ema_alpha * target_ratio + (1.0 - ema_alpha) * ema_ratio

        split_keyframes.append((t, ema_ratio))

    # Downsample keyframes for FFmpeg
    kf_interval = 1.0 / _KEYFRAMES_PER_SECOND
    downsampled = []
    for i, (t, ratio) in enumerate(split_keyframes):
        if i == 0 or i == len(split_keyframes) - 1:
            downsampled.append((t, ratio))
        elif i % max(1, int(kf_interval / _SAMPLE_INTERVAL)) == 0:
            downsampled.append((t, ratio))

    # Build FFmpeg filter with animated split
    _ffmpeg_dynamic_split(
        src, out_path,
        top_cx * width, bot_cx * width,
        downsampled,
        width, height, target_width, target_height,
        divider_h, start_time=start_time, duration=duration,
    )

    return out_path


def _process_broll(
    src: str,
    out_path: str,
    cfg,
    duration: float,
    width: int,
    height: int,
    target_width: int,
    target_height: int,
) -> str:
    """Apply Ken Burns effect (slow zoom + pan) for b-roll segments."""
    if getattr(cfg, "BROLL_KEN_BURNS", True):
        print("    Reframe [Advanced]: B-roll → Ken Burns effect")
        _ffmpeg_ken_burns(
            src, out_path, width, height,
            target_width, target_height, duration,
        )
    else:
        print("    Reframe [Advanced]: B-roll → centre crop")
        crop_x = int(width / 2.0 - target_width / 2)
        crop_x = max(0, min(crop_x, width - target_width))
        _ffmpeg_crop(src, out_path, crop_x, target_width, target_height)

    return out_path


def _fuse_audio_into_timeline(
    speaker_tl: list[dict],
    diarization_segments: list,
    face_timeline: list,
    sd_mod,
):
    """
    Enhance the speaker timeline with audio diarization data.
    When lip detection and audio agree, confidence is high.
    When they disagree, hold the current position.
    """
    face_id_to_audio: dict[int, str] = {}

    for i, entry in enumerate(speaker_tl):
        t = entry["time"]
        lip_speaker = entry["speaker_id"] if entry["speaker_id"] >= 0 else None
        audio_speaker = sd_mod.get_active_speaker(t, diarization_segments)

        fused_id, confidence = sd_mod.fuse_signals(
            lip_speaker, audio_speaker, face_id_to_audio,
        )

        if fused_id is not None and confidence >= 0.5:
            entry["speaker_id"] = fused_id
            # Find the face position for this ID
            if i < len(face_timeline):
                for face in face_timeline[i].get("faces", []):
                    if face.face_id == fused_id:
                        entry["target_cx"] = face.norm_cx
                        break
        elif confidence < 0.3:
            # Low confidence — hold position (don't change speaker_id)
            pass


def _get_duration_cv2(video_path: str) -> float:
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0 and frames > 0:
        return frames / fps
    return 0.0


# ─── Segment processing ──────────────────────────────────────────────────────

def _process_segment(src, out_path, seg, width, height,
                     target_width, target_height,
                     start_time=None, duration=None):
    """
    Process one layout segment.
    Uses keyframed (animated) crop when faces move significantly,
    otherwise falls back to a cheaper static crop.
    """
    keyframes = seg.get("keyframes", [])

    if seg["type"] == "split" and len(seg.get("cxs", [])) >= 2:
        # ── Multiple faces: letterbox the entire frame into 9:16 ──────
        _ffmpeg_letterbox(src, out_path, width, height,
                          target_width, target_height,
                          start_time, duration)

    else:
        # ── Single-face crop ──────────────────────────────────────────
        if keyframes and _has_significant_movement(keyframes, min_cxs=1):
            kf = [(t, cxs[0] * width) for t, cxs in keyframes]
            _ffmpeg_crop_tracked(src, out_path, kf, width,
                                target_width, target_height,
                                start_time, duration)
        else:
            cx_pixel = seg["cxs"][0] * width
            crop_x   = int(cx_pixel - target_width / 2)
            crop_x   = max(0, min(crop_x, width - target_width))
            _ffmpeg_crop(src, out_path, crop_x, target_width, target_height,
                         start_time, duration)


def _has_significant_movement(keyframes, min_cxs=1):
    """True if the primary face moves more than _MIN_MOVEMENT_FRAC of the frame."""
    if len(keyframes) <= 1:
        return False
    positions = [cxs[0] for _, cxs in keyframes if len(cxs) >= min_cxs]
    if len(positions) < 2:
        return False
    return (max(positions) - min(positions)) > _MIN_MOVEMENT_FRAC


# ─── Face detection ───────────────────────────────────────────────────────────

def _create_face_detector():
    """
    Create a face detector using the best available backend.
    Returns (detector_callable, cleanup_callable).

    detector_callable(frame_bgr) -> list[(norm_cx, norm_area)]
    cleanup_callable() -> None
    """
    project_root = os.path.dirname(os.path.dirname(__file__))

    # ── 1. Try MediaPipe Tasks API with BOTH models for best coverage ─────
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions

        short_model = os.path.join(project_root, "blaze_face_short_range.tflite")
        full_model = os.path.join(project_root, "blaze_face_full_range.tflite")

        detectors = []
        labels = []

        # Try to load full-range model (better for distant faces in interviews)
        if os.path.isfile(full_model):
            opts = FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=full_model),
                min_detection_confidence=0.35,
            )
            detectors.append(FaceDetector.create_from_options(opts))
            labels.append("full-range")

        # Also load short-range model (better for close-up faces)
        if os.path.isfile(short_model):
            opts = FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=short_model),
                min_detection_confidence=0.35,
            )
            detectors.append(FaceDetector.create_from_options(opts))
            labels.append("short-range")

        if not detectors:
            raise FileNotFoundError("No MediaPipe Tasks model files found")

        def _detect_tasks(frame_bgr):
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_h, frame_w = frame_bgr.shape[:2]

            all_faces = []
            for det in detectors:
                result = det.detect(mp_image)
                for d in result.detections:
                    bb = d.bounding_box
                    norm_cx = (bb.origin_x + bb.width / 2) / frame_w
                    norm_area = (bb.width * bb.height) / (frame_w * frame_h)
                    all_faces.append((norm_cx, norm_area))

            # Deduplicate: merge faces whose centres are within 10% of frame width
            all_faces.sort(key=lambda f: f[0])
            merged = []
            for face in all_faces:
                if merged and abs(face[0] - merged[-1][0]) < 0.10:
                    # Keep the one with larger area (more confident detection)
                    if face[1] > merged[-1][1]:
                        merged[-1] = face
                else:
                    merged.append(face)
            return merged

        def _cleanup():
            for det in detectors:
                det.close()

        print(f"    Reframe: Using MediaPipe Tasks face detector ({' + '.join(labels)})")
        return _detect_tasks, _cleanup

    except Exception:
        pass

    # ── 2. Try legacy MediaPipe solutions API ─────────────────────────────
    try:
        import mediapipe as mp
        fd = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.35
        )

        def _detect_solutions(frame_bgr):
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = fd.process(rgb)
            faces = []
            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    norm_cx = bbox.xmin + bbox.width / 2.0
                    norm_area = bbox.width * bbox.height
                    faces.append((norm_cx, norm_area))
            return faces

        print("    Reframe: Using MediaPipe Solutions face detector")
        return _detect_solutions, lambda: fd.close()

    except Exception:
        pass

    # ── 3. Fallback: OpenCV Haar cascade ──────────────────────────────────
    cascade_path = os.path.join(project_root, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    def _detect_haar(frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = gray.shape[:2]
        min_size = (int(frame_w * 0.05), int(frame_h * 0.05))
        rects = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=min_size)
        faces = []
        for (x, y, w, h) in rects:
            norm_cx = (x + w / 2.0) / frame_w
            norm_area = (w * h) / (frame_w * frame_h)
            faces.append((norm_cx, norm_area))
        return faces

    print("    Reframe: Using OpenCV Haar cascade face detector (less accurate)")
    return _detect_haar, lambda: None


# ─── Layout timeline analysis (4 FPS) ────────────────────────────────────────

def _analyze_layout_timeline(video_path: str, duration: float) -> list:
    """
    Step through clip at _SAMPLE_INTERVAL (250 ms = 4 FPS).
    Returns a timeline of per-sample layout decisions with sub-second precision.
    """
    detect, cleanup = _create_face_detector()

    timeline     = []
    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    current_time = 0.0
    frame_count  = 0
    total_frames = int(duration * fps) if duration > 0 else float("inf")

    while True:
        ret, frame = cap.read()
        if not ret or frame_count > total_frames:
            break

        t_sec = frame_count / fps
        if t_sec >= current_time:
            face_info        = detect(frame)
            cluster_summaries = _cluster_faces(face_info)

            # ── Decide layout for this sample ─────────────────────────
            valid_cxs = []
            if cluster_summaries:
                largest_area = cluster_summaries[0][1]
                for cx, area in cluster_summaries:
                    if area >= largest_area * 0.15:
                        valid_cxs.append(cx)

            layout_type = "single"
            cxs         = [0.5]

            if len(valid_cxs) >= 2:
                pair = sorted(valid_cxs[:2])
                if abs(pair[1] - pair[0]) >= _MIN_SPLIT_SEPARATION:
                    layout_type = "split"
                    cxs = pair
                else:
                    layout_type = "single"
                    cxs = [(pair[0] + pair[1]) / 2.0]
            elif len(valid_cxs) == 1:
                cxs = [valid_cxs[0]]

            timeline.append({
                "start": round(current_time, 3),
                "end":   round(current_time + _SAMPLE_INTERVAL, 3),
                "type":  layout_type,
                "cxs":   cxs,
            })

            current_time += _SAMPLE_INTERVAL
            if current_time >= duration and duration > 0:
                break

        frame_count += 1

    cap.release()
    cleanup()

    # ── Fix up last interval ──────────────────────────────────────────────
    if timeline and timeline[-1]["end"] < duration:
        timeline[-1]["end"] = round(duration, 3)
    elif not timeline:
        timeline.append({
            "start": 0.0,
            "end":   round(duration, 3),
            "type":  "single",
            "cxs":   [0.5],
        })
    elif timeline[-1]["end"] > duration:
        timeline[-1]["end"] = round(duration, 3)

    # ── Temporal smoothing ────────────────────────────────────────────────
    timeline = _smooth_timeline(timeline)
    return timeline


# ─── Temporal smoothing ──────────────────────────────────────────────────────

def _smooth_timeline(timeline: list) -> list:
    """
    Remove short layout flips.  If a run of 'split' (or 'single') is shorter
    than _MIN_SEGMENT_ENTRIES samples, convert it to match its neighbours.
    """
    if len(timeline) <= _MIN_SEGMENT_ENTRIES:
        return timeline

    types = [t["type"] for t in timeline]

    # Identify runs
    runs = []  # (start_idx, end_idx_exclusive, type)
    i = 0
    while i < len(types):
        j = i
        while j < len(types) and types[j] == types[i]:
            j += 1
        runs.append((i, j, types[i]))
        i = j

    # Flip short runs to their dominant neighbour
    for ri, (start, end, rtype) in enumerate(runs):
        run_len = end - start
        if run_len < _MIN_SEGMENT_ENTRIES:
            prev_type = runs[ri - 1][2] if ri > 0 else None
            next_type = runs[ri + 1][2] if ri < len(runs) - 1 else None
            flip_to   = prev_type or next_type or "single"
            if prev_type and next_type and prev_type != next_type:
                flip_to = "single"  # ambiguous — default to single
            for idx in range(start, end):
                timeline[idx]["type"] = flip_to
                # When flipping split→single, keep only the first cx (largest face)
                if flip_to == "single" and len(timeline[idx]["cxs"]) > 1:
                    timeline[idx]["cxs"] = [timeline[idx]["cxs"][0]]
                # When flipping single→split, we can't fabricate a second face,
                # so keep the single cx — _chunk_segments will handle mismatch
                # by not merging with real split neighbours.

    return timeline


# ─── Face clustering ─────────────────────────────────────────────────────────

def _cluster_faces(
    face_info: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Cluster face detections and return centres sorted by area (largest first)."""
    if not face_info:
        return []

    face_info_sorted = sorted(face_info, key=lambda f: f[0])
    clusters         = []
    current_cluster  = [face_info_sorted[0]]
    threshold        = 0.15

    for f in face_info_sorted[1:]:
        if f[0] - current_cluster[-1][0] < threshold:
            current_cluster.append(f)
        else:
            clusters.append(current_cluster)
            current_cluster = [f]
    clusters.append(current_cluster)

    cluster_summaries = []
    for c in clusters:
        median_x = float(np.median([f[0] for f in c]))
        max_area = max(f[1] for f in c)
        cluster_summaries.append((median_x, max_area))

    cluster_summaries.sort(key=lambda c: c[1], reverse=True)
    return cluster_summaries


# ─── Segment chunking with EMA-smoothed keyframes ────────────────────────────

def _chunk_segments(timeline: list) -> list:
    """
    Group contiguous intervals that share the same layout type.
    Preserves per-sample face positions and produces EMA-smoothed keyframes
    for animated FFmpeg crop expressions.
    """
    if not timeline:
        return []

    segments    = []
    current_seg = timeline[0].copy()
    current_seg["cxs_list"]   = [current_seg["cxs"]]
    current_seg["times_list"] = [current_seg["start"]]

    move_threshold = 0.15
    for t in timeline[1:]:
        same_type      = t["type"] == current_seg["type"]
        moved_too_much = False

        if same_type:
            last_cxs = current_seg["cxs_list"][-1]
            if len(t["cxs"]) != len(last_cxs):
                same_type = False
            else:
                for cx_new, cx_old in zip(t["cxs"], last_cxs):
                    if abs(cx_new - cx_old) > move_threshold:
                        moved_too_much = True
                        break

        if same_type and not moved_too_much:
            current_seg["end"] = t["end"]
            current_seg["cxs_list"].append(t["cxs"])
            current_seg["times_list"].append(t["start"])
        else:
            _finalize_segment(current_seg)
            segments.append(current_seg)

            current_seg = t.copy()
            current_seg["cxs_list"]   = [current_seg["cxs"]]
            current_seg["times_list"] = [current_seg["start"]]

    _finalize_segment(current_seg)
    segments.append(current_seg)

    return segments


def _finalize_segment(seg: dict):
    """
    Compute average cxs (for static-crop fallback) and produce
    EMA-smoothed, downsampled keyframes for animated crop expressions.
    """
    cxs_list   = seg["cxs_list"]
    times_list = seg["times_list"]

    # Average cxs (used by static-crop fallback)
    seg["cxs"] = np.mean(cxs_list, axis=0).tolist()

    # EMA-smooth the raw per-sample positions
    smoothed = _ema_smooth(cxs_list)

    # Downsample to _KEYFRAMES_PER_SECOND for the FFmpeg expression
    seg_start      = seg["start"]
    keyframes      = []
    kf_interval    = 1.0 / _KEYFRAMES_PER_SECOND
    samples_per_kf = max(1, int(kf_interval / _SAMPLE_INTERVAL))

    for i in range(0, len(smoothed), samples_per_kf):
        abs_time = times_list[i]
        keyframes.append((round(abs_time, 3), smoothed[i]))

    # Always include the last sample as a keyframe
    if len(smoothed) > 1:
        abs_time = times_list[-1]
        if not keyframes or keyframes[-1][0] != round(abs_time, 3):
            keyframes.append((round(abs_time, 3), smoothed[-1]))

    seg["keyframes"] = keyframes

    # Cleanup temporary lists
    del seg["cxs_list"]
    del seg["times_list"]


def _ema_smooth(cxs_list: list) -> list:
    """
    Apply Exponential Moving Average to face positions.
    Removes detection jitter while preserving intentional face movement.
    """
    if not cxs_list:
        return []

    smoothed = [list(cxs_list[0])]
    for i in range(1, len(cxs_list)):
        prev = smoothed[-1]
        curr = cxs_list[i]
        new_pos = [
            _EMA_ALPHA * c + (1 - _EMA_ALPHA) * p
            for p, c in zip(prev, curr)
        ]
        smoothed.append(new_pos)

    return smoothed


# ─── FFmpeg expression builder ───────────────────────────────────────────────

def _build_crop_expr(keyframes: list, width: int, target_width: int) -> str:
    """
    Convert (time, face_centre_px) keyframes to a piecewise-linear FFmpeg
    expression for the crop x position.
    """
    if not keyframes:
        return str(int(width / 2 - target_width / 2))

    # Convert centre → left-edge, clamped to valid pixel range
    clamped = []
    for t, cx in keyframes:
        x = cx - target_width / 2
        x = max(0, min(x, width - target_width))
        clamped.append((t, x))

    return _build_expr_raw(clamped)


def _build_expr_raw(keyframes: list) -> str:
    """
    Build a piecewise-linear FFmpeg expression from (time, value) keyframes.

    Output: nested if(lt(t,T), lerp, …) expression that linearly interpolates
    between consecutive keyframe values.
    """
    if not keyframes:
        return "0"
    if len(keyframes) == 1:
        return f"{keyframes[0][1]:.0f}"

    # Build from the end backwards:
    # default = last value
    expr = f"{keyframes[-1][1]:.0f}"

    for i in range(len(keyframes) - 2, -1, -1):
        t0, x0 = keyframes[i]
        t1, x1 = keyframes[i + 1]
        dt = t1 - t0

        if abs(x1 - x0) < 1.0 or dt <= 0:
            # No meaningful movement — constant value
            lerp = f"{x0:.0f}"
        else:
            # Linear interpolation: x0 + (x1-x0)*(t-t0)/(t1-t0)
            lerp = f"{x0:.1f}+{x1 - x0:.1f}*(t-{t0:.3f})/{dt:.3f}"

        expr = f"if(lt(t,{t1:.3f}),{lerp},{expr})"

    return expr


# ─── FFmpeg letterbox — full frame with black bars ───────────────────────────

def _ffmpeg_letterbox(
    src: str,
    out_path: str,
    width: int,
    height: int,
    target_width: int,
    target_height: int,
    start_time: float = None,
    duration: float = None,
):
    """
    Fit the entire source frame into a 9:16 canvas with black bars
    on top and bottom.  Used when multiple faces are detected so that
    no subject is cropped out.
    """
    # Scale source to fit within target_width x target_height, preserving AR
    vf = (
        f"scale={target_width}:{target_height}"
        f":force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:black"
    )

    cmd = ["ffmpeg", "-y", "-i", src]
    if start_time is not None and start_time > 0:
        cmd.extend(["-ss", f"{start_time:.3f}"])
    if duration is not None:
        cmd.extend(["-t", f"{duration:.3f}"])

    cmd.extend([
        "-vf", vf,
        *get_video_codec_args(Config()),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-map", "0:v",
        "-map", "0:a?",
        out_path,
    ])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg letterbox failed:\n{result.stderr[-500:]}")


# ─── FFmpeg crop — static (fallback) ─────────────────────────────────────────

def _ffmpeg_crop(
    src: str,
    out_path: str,
    crop_x: int,
    target_width: int,
    target_height: int,
    start_time: float = None,
    duration: float = None,
):
    """Static crop — fast, single position for the whole segment."""
    cmd = ["ffmpeg", "-y", "-i", src]
    if start_time is not None and start_time > 0:
        cmd.extend(["-ss", f"{start_time:.3f}"])
    if duration is not None:
        cmd.extend(["-t", f"{duration:.3f}"])

    cmd.extend([
        "-vf", f"crop={target_width}:{target_height}:{crop_x}:0",
        *get_video_codec_args(Config()),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-map", "0:v",
        "-map", "0:a?",
        out_path,
    ])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg crop failed:\n{result.stderr[-500:]}")


# ─── FFmpeg crop — animated (face tracking) ──────────────────────────────────

def _ffmpeg_crop_tracked(
    src: str,
    out_path: str,
    keyframes: list,
    width: int,
    target_width: int,
    target_height: int,
    start_time: float = None,
    duration: float = None,
):
    """
    Animated crop that smoothly follows the face using a piecewise-linear
    FFmpeg expression.  Falls back to static crop on failure.
    """
    x_expr = _build_crop_expr(keyframes, width, target_width)

    cmd = ["ffmpeg", "-y", "-i", src]
    if start_time is not None and start_time > 0:
        cmd.extend(["-ss", f"{start_time:.3f}"])
    if duration is not None:
        cmd.extend(["-t", f"{duration:.3f}"])

    # Single-quote the expression so FFmpeg's filter parser treats commas
    # inside it as part of the expression, not as filter-graph separators.
    vf = f"crop={target_width}:{target_height}:'{x_expr}':0"

    cmd.extend([
        "-vf", vf,
        *get_video_codec_args(Config()),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-map", "0:v",
        "-map", "0:a?",
        out_path,
    ])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ⚠ Animated crop failed, falling back to static crop")
        print(f"      stderr: …{result.stderr[-200:]}")
        avg_cx = np.mean([cx for _, cx in keyframes])
        crop_x = int(avg_cx - target_width / 2)
        crop_x = max(0, min(crop_x, width - target_width))
        _ffmpeg_crop(src, out_path, crop_x, target_width, target_height,
                     start_time, duration)


# ─── FFmpeg split crop — static (fallback) ───────────────────────────────────

def _ffmpeg_split_crop(
    src: str,
    out_path: str,
    cx1: float,
    cx2: float,
    width: int,
    height: int,
    target_width: int,
    target_height: int,
    start_time: float = None,
    duration: float = None,
):
    """
    Static split-screen 9:16 output with two face crops stacked vertically.

    Polish details:
    - 4 px black divider line between the two halves
    - 70% source-height crop for generous framing
    - Vertical crop centred on the middle of the frame
    """
    divider_h = 4
    half_h    = (target_height - divider_h) // 2
    half_h    = half_h if half_h % 2 == 0 else half_h - 1

    crop_h = int(height * 0.52)
    crop_w = int(crop_h * 9 / 8)
    if crop_w > width:
        crop_w = width
        crop_h = int(crop_w * 8 / 9)
    crop_h = crop_h if crop_h % 2 == 0 else crop_h - 1
    crop_w = crop_w if crop_w % 2 == 0 else crop_w - 1

    top_x = int(cx1 - crop_w / 2)
    top_x = max(0, min(top_x, width - crop_w))

    bot_x = int(cx2 - crop_w / 2)
    bot_x = max(0, min(bot_x, width - crop_w))

    crop_y = int((height - crop_h) * 0.35)
    crop_y = max(0, min(crop_y, height - crop_h))

    filter_complex = (
        f"[0:v]crop={crop_w}:{crop_h}:{top_x}:{crop_y},"
        f"scale={target_width}:{half_h}:flags=lanczos[top];"

        f"[0:v]crop={crop_w}:{crop_h}:{bot_x}:{crop_y},"
        f"scale={target_width}:{half_h}:flags=lanczos[bottom];"

        f"color=black:{target_width}x{divider_h}:d=1[div];"

        f"[top][div][bottom]vstack=inputs=3[v]"
    )

    cmd = ["ffmpeg", "-y", "-i", src]
    if start_time is not None and start_time > 0:
        cmd.extend(["-ss", f"{start_time:.3f}"])
    if duration is not None:
        cmd.extend(["-t", f"{duration:.3f}"])

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a?",
        *get_video_codec_args(Config()),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        out_path,
    ])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg split crop failed:\n{result.stderr[-500:]}")


# ─── FFmpeg split crop — animated (face tracking) ────────────────────────────

def _ffmpeg_split_crop_tracked(
    src: str,
    out_path: str,
    kf_top: list,
    kf_bot: list,
    width: int,
    height: int,
    target_width: int,
    target_height: int,
    start_time: float = None,
    duration: float = None,
):
    """
    Animated split-screen where each panel's crop independently tracks
    its face.  Falls back to static split on failure.
    """
    divider_h = 4
    half_h    = (target_height - divider_h) // 2
    half_h    = half_h if half_h % 2 == 0 else half_h - 1

    crop_h = int(height * 0.70)
    crop_w = int(crop_h * 9 / 8)
    if crop_w > width:
        crop_w = width
        crop_h = int(crop_w * 8 / 9)
    crop_h = crop_h if crop_h % 2 == 0 else crop_h - 1
    crop_w = crop_w if crop_w % 2 == 0 else crop_w - 1

    crop_y = int((height - crop_h) * 0.35)
    crop_y = max(0, min(crop_y, height - crop_h))

    # Build tracking expressions — convert face-centre to crop-left, clamped
    kf_top_clamped = [
        (t, max(0, min(cx - crop_w / 2, width - crop_w)))
        for t, cx in kf_top
    ]
    kf_bot_clamped = [
        (t, max(0, min(cx - crop_w / 2, width - crop_w)))
        for t, cx in kf_bot
    ]

    expr_top = _build_expr_raw(kf_top_clamped)
    expr_bot = _build_expr_raw(kf_bot_clamped)

    filter_complex = (
        f"[0:v]crop={crop_w}:{crop_h}:'{expr_top}':{crop_y},"
        f"scale={target_width}:{half_h}:flags=lanczos[top];"

        f"[0:v]crop={crop_w}:{crop_h}:'{expr_bot}':{crop_y},"
        f"scale={target_width}:{half_h}:flags=lanczos[bottom];"

        f"color=black:{target_width}x{divider_h}:d=1[div];"

        f"[top][div][bottom]vstack=inputs=3[v]"
    )

    cmd = ["ffmpeg", "-y", "-i", src]
    if start_time is not None and start_time > 0:
        cmd.extend(["-ss", f"{start_time:.3f}"])
    if duration is not None:
        cmd.extend(["-t", f"{duration:.3f}"])

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a?",
        *get_video_codec_args(Config()),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        out_path,
    ])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ⚠ Tracked split crop failed, falling back to static split")
        print(f"      stderr: …{result.stderr[-200:]}")
        avg_top = np.mean([cx for _, cx in kf_top])
        avg_bot = np.mean([cx for _, cx in kf_bot])
        _ffmpeg_split_crop(
            src, out_path, avg_top, avg_bot,
            width, height, target_width, target_height,
            start_time, duration,
        )


# ─── FFmpeg Ken Burns — gentle zoom + pan for b-roll ─────────────────────────

def _ffmpeg_ken_burns(
    src: str,
    out_path: str,
    width: int,
    height: int,
    target_width: int,
    target_height: int,
    duration: float,
    start_time: float = None,
):
    """
    Apply a slow Ken Burns effect (gentle zoom-in + pan) for b-roll
    or faceless segments. Creates cinematic motion on otherwise static shots.

    The zoom starts at 1.0x and slowly increases to ~1.15x over the
    segment duration, while panning gently from center-left to center-right.
    """
    # Calculate zoom rate based on duration (longer = slower zoom)
    # Target: 15% zoom over the full duration
    max_zoom = 1.15
    fps_out = 30  # Output framerate for zoompan

    # zoompan filter works on single frames, so we need to compute
    # zoom and position per frame
    total_frames = int(duration * fps_out)
    if total_frames <= 0:
        total_frames = 1

    # Zoom expression: linear zoom from 1.0 to max_zoom
    zoom_per_frame = (max_zoom - 1.0) / total_frames
    zoom_expr = f"min(zoom+{zoom_per_frame:.8f},{max_zoom})"

    # Pan expression: gentle horizontal drift
    # Start slightly left of center, drift right
    # x range: from 10% to 40% of (zoomed_width - target_width)
    # Using iw/ih for input width/height
    x_expr = f"iw/2-(iw/zoom/2)+((iw/zoom-iw/zoom/{max_zoom})*on/{total_frames}*0.3)"
    y_expr = f"ih/2-(ih/zoom/2)"

    # First scale to ensure we have enough resolution for zooming
    scale_w = int(target_width * max_zoom * 1.1)
    scale_h = int(target_height * max_zoom * 1.1)
    # Ensure even dimensions
    scale_w = scale_w if scale_w % 2 == 0 else scale_w + 1
    scale_h = scale_h if scale_h % 2 == 0 else scale_h + 1

    vf = (
        f"scale={scale_w}:{scale_h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={scale_w}:{scale_h},"
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}'"
        f":d=1:s={target_width}x{target_height}:fps={fps_out}"
    )

    cmd = ["ffmpeg", "-y"]
    if start_time is not None and start_time > 0:
        cmd.extend(["-ss", f"{start_time:.3f}"])
    cmd.extend(["-i", src])
    if duration is not None:
        cmd.extend(["-t", f"{duration:.3f}"])

    cmd.extend([
        "-vf", vf,
        "-map", "0:v:0",
        "-map", "0:a?",
        *get_video_codec_args(Config()),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        out_path,
    ])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ⚠ Ken Burns effect failed, falling back to centre crop")
        print(f"      stderr: …{result.stderr[-200:]}")
        crop_x = int(width / 2 - target_width / 2)
        crop_x = max(0, min(crop_x, width - target_width))
        _ffmpeg_crop(src, out_path, crop_x, target_width, target_height,
                     start_time, duration)


# ─── FFmpeg dynamic split — conversation-responsive split screen ─────────────

def _ffmpeg_dynamic_split(
    src: str,
    out_path: str,
    top_cx: float,
    bot_cx: float,
    split_keyframes: list[tuple[float, float]],
    width: int,
    height: int,
    target_width: int,
    target_height: int,
    divider_h: int = 4,
    start_time: float = None,
    duration: float = None,
):
    """
    Dynamic split screen where the vertical split ratio changes over time
    based on who is speaking. The active speaker gets more screen space.

    split_keyframes: list of (time, top_ratio) where top_ratio is the
    fraction of vertical space for the top panel (0.4 to 0.6 typically).
    """
    crop_h = int(height * 0.52)
    crop_w = int(crop_h * 9 / 8)
    if crop_w > width:
        crop_w = width
        crop_h = int(crop_w * 8 / 9)
    crop_h = crop_h if crop_h % 2 == 0 else crop_h - 1
    crop_w = crop_w if crop_w % 2 == 0 else crop_w - 1

    crop_y = int((height - crop_h) * 0.35)
    crop_y = max(0, min(crop_y, height - crop_h))

    top_x = int(top_cx - crop_w / 2)
    top_x = max(0, min(top_x, width - crop_w))

    bot_x = int(bot_cx - crop_w / 2)
    bot_x = max(0, min(bot_x, width - crop_w))

    # For dynamic split, we use a fixed average ratio since time-varying
    # crop height in FFmpeg filter graphs is extremely complex.
    # Instead, we use the average ratio for a smooth-looking result,
    # and rely on the slight emphasis difference to be perceptible.
    if split_keyframes:
        avg_ratio = float(np.mean([r for _, r in split_keyframes]))
    else:
        avg_ratio = 0.5

    usable_height = target_height - divider_h
    top_h = int(usable_height * avg_ratio)
    top_h = top_h if top_h % 2 == 0 else top_h - 1
    bot_h = usable_height - top_h
    bot_h = bot_h if bot_h % 2 == 0 else bot_h - 1

    # Adjust divider to absorb any rounding error
    actual_divider = target_height - top_h - bot_h

    filter_complex = (
        f"[0:v]crop={crop_w}:{crop_h}:{top_x}:{crop_y},"
        f"scale={target_width}:{top_h}:flags=lanczos[top];"

        f"[0:v]crop={crop_w}:{crop_h}:{bot_x}:{crop_y},"
        f"scale={target_width}:{bot_h}:flags=lanczos[bottom];"

        f"color=black:{target_width}x{actual_divider}:d=1[div];"

        f"[top][div][bottom]vstack=inputs=3[v]"
    )

    cmd = ["ffmpeg", "-y", "-i", src]
    if start_time is not None and start_time > 0:
        cmd.extend(["-ss", f"{start_time:.3f}"])
    if duration is not None:
        cmd.extend(["-t", f"{duration:.3f}"])

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a?",
        *get_video_codec_args(Config()),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        out_path,
    ])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ⚠ Dynamic split failed, falling back to static split")
        print(f"      stderr: …{result.stderr[-200:]}")
        _ffmpeg_split_crop(
            src, out_path, top_cx, bot_cx,
            width, height, target_width, target_height,
            start_time, duration,
        )

