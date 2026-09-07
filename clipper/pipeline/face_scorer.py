"""
Face Quality Scorer
====================
Combines multiple face signals into a single composite "subject score"
to determine which face in a multi-person scene deserves primary attention.

Scoring weights:
  - Size (30%): Larger faces are closer/more important
  - Frontality (20%): Frontal faces are subjects, profiles are listeners
  - Confidence (15%): Higher detection confidence = more reliable
  - Lip activity (15%): Speaking faces are more important
  - Eye openness (10%): Open eyes = active subject
  - Gaze (10%): Looking at camera = primary subject
"""

from pipeline.face_analyzer import FaceData


# ─── Scoring weights ─────────────────────────────────────────────────────────

_WEIGHT_SIZE = 0.25
_WEIGHT_FRONTALITY = 0.20
_WEIGHT_CONFIDENCE = 0.10
_WEIGHT_LIP_ACTIVITY = 0.25
_WEIGHT_EYE_OPENNESS = 0.10
_WEIGHT_GAZE = 0.10


def score_face(face: FaceData, max_area: float = 0.0) -> float:
    """
    Compute a composite subject score for a single face.

    Args:
        face: FaceData with all signals populated.
        max_area: The largest face area in the current frame (for normalization).
                  If 0, size_score uses raw area.

    Returns:
        Score from 0.0 (low priority) to 1.0 (high priority).
    """
    # ── Size score ────────────────────────────────────────────────────────
    # Normalize area relative to the largest face in frame
    if max_area > 0:
        size_score = min(face.norm_area / max_area, 1.0)
    else:
        # Absolute sizing: a face taking 5%+ of frame is large
        size_score = min(face.norm_area / 0.05, 1.0)

    # ── Confidence score ──────────────────────────────────────────────────
    confidence_score = max(0.0, min(face.confidence, 1.0))

    # ── Frontality score ──────────────────────────────────────────────────
    frontality_score = max(0.0, min(face.frontality, 1.0))

    # ── Eye openness score ────────────────────────────────────────────────
    eye_score = max(0.0, min(face.eye_openness, 1.0))

    # ── Gaze score ────────────────────────────────────────────────────────
    gaze_score = max(0.0, min(face.gaze_at_camera, 1.0))

    # ── Lip activity score ────────────────────────────────────────────────
    # Binary boost: speaking faces get full score, silent faces get partial
    lip_score = 1.0 if face.is_speaking else 0.3

    # ── Weighted composite ────────────────────────────────────────────────
    total = (
        _WEIGHT_SIZE * size_score +
        _WEIGHT_CONFIDENCE * confidence_score +
        _WEIGHT_FRONTALITY * frontality_score +
        _WEIGHT_EYE_OPENNESS * eye_score +
        _WEIGHT_GAZE * gaze_score +
        _WEIGHT_LIP_ACTIVITY * lip_score
    )

    return float(max(0.0, min(1.0, total)))


def rank_faces(faces: list[FaceData]) -> list[FaceData]:
    """
    Score and rank faces by subject importance.
    Modifies each face's subject_score in-place and returns
    the list sorted by score (highest first).
    """
    if not faces:
        return faces

    max_area = max(f.norm_area for f in faces) if faces else 0.0

    for face in faces:
        face.subject_score = score_face(face, max_area)

    faces.sort(key=lambda f: f.subject_score, reverse=True)
    return faces
