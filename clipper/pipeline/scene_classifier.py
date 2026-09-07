"""
Scene Classifier — Automatic Scene Type Detection
===================================================
Classifies video scenes into types based on face analysis data,
so the right reframing strategy is applied automatically:

  - talking_head: 1 face, consistent position, high frontality
  - interview: 2 faces, alternating activity, stable positions
  - panel: 3+ faces, varying activity patterns
  - presentation: 1 face with significant movement
  - b_roll: <20% of frames have faces (cutaway footage)

Classification uses heuristics on face timeline data — no extra model needed.
"""

from collections import Counter
from pipeline.face_analyzer import FaceData


# ─── Scene types ──────────────────────────────────────────────────────────────

SCENE_TALKING_HEAD = "talking_head"
SCENE_INTERVIEW = "interview"
SCENE_PANEL = "panel"
SCENE_PRESENTATION = "presentation"
SCENE_BROLL = "b_roll"


def classify_scene(face_timeline: list[dict]) -> str:
    """
    Classify the scene type based on face analysis timeline data.

    Args:
        face_timeline: List of dicts with keys:
            - "time": float
            - "faces": list[FaceData]

    Returns:
        One of: "talking_head", "interview", "panel", "presentation", "b_roll"
    """
    if not face_timeline:
        return SCENE_BROLL

    total_frames = len(face_timeline)

    # ── Count face presence ───────────────────────────────────────────────
    frames_with_faces = sum(1 for entry in face_timeline if entry.get("faces"))
    face_presence_ratio = frames_with_faces / total_frames if total_frames > 0 else 0.0

    # B-roll: very few faces
    if face_presence_ratio < 0.20:
        return SCENE_BROLL

    # ── Collect face statistics ───────────────────────────────────────────
    face_counts = []
    unique_face_ids = set()
    face_positions: dict[int, list[float]] = {}  # face_id → list of norm_cx
    face_frontalities: dict[int, list[float]] = {}
    face_scores: dict[int, list[float]] = {}     # face_id → list of subject_score

    for entry in face_timeline:
        faces = entry.get("faces", [])
        face_counts.append(len(faces))

        for face in faces:
            fid = face.face_id
            unique_face_ids.add(fid)

            if fid not in face_positions:
                face_positions[fid] = []
                face_frontalities[fid] = []
                face_scores[fid] = []

            face_positions[fid].append(face.norm_cx)
            face_frontalities[fid].append(face.frontality)
            face_scores[fid].append(getattr(face, "subject_score", 0.0))

    # Number of unique persistent faces
    num_unique_faces = len(unique_face_ids)

    # Typical (mode) face count per frame
    if face_counts:
        mode_count = Counter(face_counts).most_common(1)[0][0]
        median_count = sorted(face_counts)[len(face_counts) // 2]
    else:
        mode_count = 0
        median_count = 0

    # ── Panel: 3+ faces consistently ──────────────────────────────────────
    if median_count >= 3 or (num_unique_faces >= 3 and mode_count >= 3):
        return SCENE_PANEL

    # ── Interview vs talking head (1–2 faces) ─────────────────────────────
    if median_count >= 2 or num_unique_faces >= 2:
        # Two faces detected — is it an interview layout?
        # Check if they have stable, separated positions
        if _is_interview_layout(face_positions, face_scores, face_timeline):
            return SCENE_INTERVIEW

    # ── Single face — talking head or presentation? ───────────────────────
    if num_unique_faces == 1 or (num_unique_faces <= 2 and mode_count <= 1):
        # Check for significant movement → presentation
        for fid, positions in face_positions.items():
            if len(positions) >= 5:
                import numpy as np
                pos_range = max(positions) - min(positions)
                if pos_range > 0.20:
                    return SCENE_PRESENTATION

        # Check frontality — high frontality = talking head
        avg_frontality = 0.0
        count = 0
        for fid, fronts in face_frontalities.items():
            avg_frontality += sum(fronts)
            count += len(fronts)
        if count > 0:
            avg_frontality /= count

        if avg_frontality > 0.6:
            return SCENE_TALKING_HEAD
        else:
            return SCENE_PRESENTATION  # Low frontality + single face = presenter moving

    return SCENE_TALKING_HEAD  # Default


def _is_interview_layout(
    face_positions: dict[int, list[float]],
    face_scores: dict[int, list[float]],
    face_timeline: list[dict],
) -> bool:
    """
    Check if the face layout looks like a 2-person interview:
    - Two faces with stable, horizontally separated positions.
    - Both faces are prominent (appear for a significant portion of the clip).
    """
    import numpy as np

    if len(face_positions) < 2:
        return False

    # Get the two most frequently appearing face IDs
    face_appearance_counts = {fid: len(positions) for fid, positions in face_positions.items()}
    top_faces = sorted(face_appearance_counts, key=face_appearance_counts.get, reverse=True)[:2]

    if len(top_faces) < 2:
        return False

    positions_a = face_positions[top_faces[0]]
    positions_b = face_positions[top_faces[1]]

    # Prominence Check: Both faces must appear in at least 25% of the video
    total_frames = max(1, len(face_timeline))
    if len(positions_a) / total_frames < 0.25 or len(positions_b) / total_frames < 0.25:
        return False

    # Score Check: Average subject_score must be > 0.25 for both
    scores_a = face_scores.get(top_faces[0], [])
    scores_b = face_scores.get(top_faces[1], [])
    score_a = sum(scores_a) / len(scores_a) if scores_a else 0
    score_b = sum(scores_b) / len(scores_b) if scores_b else 0

    if score_a < 0.25 or score_b < 0.25:
        return False

    mean_a = np.mean(positions_a)
    mean_b = np.mean(positions_b)

    # Separation Check: faces should be well apart horizontally
    separation = abs(mean_a - mean_b)
    if separation < 0.12:
        return False

    return True


def get_strategy_description(scene_type: str) -> str:
    """Human-readable description of the reframing strategy for a scene type."""
    descriptions = {
        SCENE_TALKING_HEAD: "Single-face tracking with cinematic panning",
        SCENE_INTERVIEW: "Dynamic split screen with speaker emphasis (60/40)",
        SCENE_PANEL: "Wide crop with group context priority",
        SCENE_PRESENTATION: "Following crop balancing face and visual content",
        SCENE_BROLL: "Ken Burns effect (gentle zoom + pan)",
    }
    return descriptions.get(scene_type, "Default center crop")
