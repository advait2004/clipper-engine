"""
Face Analyzer — MediaPipe Face Mesh 478-Landmark Analysis
==========================================================
Replaces basic face detection with deep landmark analysis:
- Lip aperture measurement (upper/lower lip distance)
- Frontality scoring (face angle relative to camera)
- Eye openness detection
- Gaze direction (iris tracking)
- Persistent face ID tracking across frames
- Active speaker detection via lip movement variance

Uses MediaPipe Face Mesh with refine_landmarks=True for iris tracking.
Falls back gracefully to basic face detection if Face Mesh is unavailable.
"""

from dataclasses import dataclass, field
from typing import Optional
import math
import os

import cv2
import numpy as np


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class FaceData:
    """Rich per-face data extracted from Face Mesh landmarks."""
    # Position (normalized 0–1)
    norm_cx: float = 0.0
    norm_cy: float = 0.0
    norm_area: float = 0.0

    # Bounding box (normalized)
    bbox_x: float = 0.0
    bbox_y: float = 0.0
    bbox_w: float = 0.0
    bbox_h: float = 0.0

    # Lip analysis
    lip_aperture: float = 0.0          # Normalized lip opening (0 = closed, 1 = wide open)
    lip_aperture_raw: float = 0.0      # Raw pixel distance between lips

    # Quality signals
    confidence: float = 0.0            # MediaPipe detection confidence
    frontality: float = 0.0            # 0 = profile, 1 = frontal
    eye_openness: float = 0.0          # 0 = closed, 1 = fully open
    gaze_at_camera: float = 0.0        # 0 = looking away, 1 = looking at camera

    # Tracking
    face_id: int = -1                  # Persistent ID across frames
    is_speaking: bool = False          # Whether this face is currently speaking

    # Composite score (computed by face_scorer)
    subject_score: float = 0.0


# ─── Landmark indices (MediaPipe Face Mesh 478-point model) ───────────────────

# Inner lip landmarks for aperture measurement
_LIP_TOP_INNER = 13       # Upper lip inner center
_LIP_BOTTOM_INNER = 14    # Lower lip inner center

# Outer lip landmarks (for wider measurement)
_LIP_TOP_OUTER = 0        # Upper lip outer top
_LIP_BOTTOM_OUTER = 17    # Lower lip outer bottom (chin area)

# Additional lip points for robust aperture
_LIP_LEFT = 78             # Left corner of mouth
_LIP_RIGHT = 308           # Right corner of mouth
_LIP_UPPER_INNER_L = 82   # Upper inner lip left
_LIP_UPPER_INNER_R = 312  # Upper inner lip right
_LIP_LOWER_INNER_L = 87   # Lower inner lip left
_LIP_LOWER_INNER_R = 317  # Lower inner lip right

# Face height reference points
_FOREHEAD = 10             # Top of forehead
_CHIN = 152                # Bottom of chin

# Nose tip (for frontality)
_NOSE_TIP = 1
_NOSE_BRIDGE = 6

# Face outline for frontality (left and right edges)
_FACE_LEFT = 234           # Left side of face
_FACE_RIGHT = 454          # Right side of face

# Eye landmarks
_LEFT_EYE_TOP = 159
_LEFT_EYE_BOTTOM = 145
_LEFT_EYE_LEFT = 33
_LEFT_EYE_RIGHT = 133

_RIGHT_EYE_TOP = 386
_RIGHT_EYE_BOTTOM = 374
_RIGHT_EYE_LEFT = 362
_RIGHT_EYE_RIGHT = 263

# Iris landmarks (available with refine_landmarks=True, indices 468–477)
_LEFT_IRIS_CENTER = 468
_RIGHT_IRIS_CENTER = 473

# Eye socket centers (for gaze reference)
_LEFT_EYE_CENTER_APPROX = 159   # We compute center from corners instead
_RIGHT_EYE_CENTER_APPROX = 386


# ─── FaceAnalyzer ─────────────────────────────────────────────────────────────

class FaceAnalyzer:
    """
    Analyzes video frames using MediaPipe Face Mesh to extract
    rich per-face data including lip movement for speaker detection.
    """

    def __init__(self, max_faces: int = 4, min_confidence: float = 0.35):
        self._mesh = None
        self._max_faces = max_faces
        self._min_confidence = min_confidence
        self._backend = "none"

        # Face ID tracking state
        self._next_face_id = 0
        self._prev_faces: list[FaceData] = []
        self._id_match_threshold = 0.12  # Max normalized distance to match same face

        # Lip history for speaker detection
        # face_id → list of recent lip apertures
        self._lip_history: dict[int, list[float]] = {}
        self._lip_history_max = 12  # ~3 seconds at 4 FPS

        self._init_detector()

    def _init_detector(self):
        """Initialize the best available face analysis backend."""
        project_root = os.path.dirname(os.path.dirname(__file__))

        # ── 1. Try MediaPipe Tasks FaceLandmarker (latest API) ────────────────
        try:
            import urllib.request
            
            # Only skip on Windows
            if os.name == 'nt':
                raise Exception("Skipping Tasks API to prevent clearcut telemetry hang on Windows")
            
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import RunningMode

            model_path = os.path.join(project_root, "face_landmarker.task")
            if not os.path.isfile(model_path):
                print("    FaceAnalyzer: Downloading face_landmarker.task (this only happens once)...")
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                urllib.request.urlretrieve(url, model_path)

            if os.path.isfile(model_path):
                self._mp = mp
                
                # Attempt GPU first
                try:
                    base_opts = BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.GPU)
                    opts = FaceLandmarkerOptions(
                        base_options=base_opts,
                        running_mode=RunningMode.IMAGE,
                        num_faces=self._max_faces,
                        min_face_detection_confidence=self._min_confidence,
                        min_tracking_confidence=0.35,
                    )
                    self._landmarker = FaceLandmarker.create_from_options(opts)
                    print("    FaceAnalyzer: Using MediaPipe Tasks FaceLandmarker (478 landmarks) on GPU")
                except Exception as gpu_err:
                    print(f"    FaceAnalyzer: GPU delegate failed ({gpu_err}), falling back to CPU...")
                    base_opts = BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU)
                    opts = FaceLandmarkerOptions(
                        base_options=base_opts,
                        running_mode=RunningMode.IMAGE,
                        num_faces=self._max_faces,
                        min_face_detection_confidence=self._min_confidence,
                        min_tracking_confidence=0.35,
                    )
                    self._landmarker = FaceLandmarker.create_from_options(opts)
                    print("    FaceAnalyzer: Using MediaPipe Tasks FaceLandmarker (478 landmarks) on CPU")
                    
                self._backend = "face_landmarker"
                return
        except Exception as e:
            print(f"    FaceAnalyzer: FaceLandmarker unavailable ({e}), trying solutions...")

        # ── 2. Try legacy MediaPipe solutions FaceMesh ────────────────────────
        try:
            import mediapipe as mp
            self._mp = mp
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self._max_faces,
                refine_landmarks=True,
                min_detection_confidence=self._min_confidence,
                min_tracking_confidence=0.35,
            )
            self._backend = "face_mesh"
            print("    FaceAnalyzer: Using MediaPipe solutions FaceMesh (478 landmarks)")
            return
        except Exception as e:
            print(f"    FaceAnalyzer: legacy FaceMesh unavailable ({e})")

        # ── 3. Try MediaPipe Tasks FaceDetector (basic bounding box) ──────────
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions

            short_model = os.path.join(project_root, "blaze_face_short_range.tflite")
            full_model = os.path.join(project_root, "blaze_face_full_range.tflite")
            model_path = full_model if os.path.isfile(full_model) else (short_model if os.path.isfile(short_model) else None)

            if model_path:
                self._mp = mp
                opts = FaceDetectorOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    min_detection_confidence=self._min_confidence,
                )
                self._basic_detector = FaceDetector.create_from_options(opts)
                self._backend = "face_detector_tasks"
                print("    FaceAnalyzer: Using MediaPipe Tasks FaceDetector (basic, no lip data)")
                return
        except Exception as e:
            print(f"    FaceAnalyzer: Tasks FaceDetector unavailable ({e})")

        # ── 4. Fallback: legacy MediaPipe solutions FaceDetection ─────────────
        try:
            import mediapipe as mp
            self._mp = mp
            self._basic_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=self._min_confidence,
            )
            self._backend = "face_detection"
            print("    FaceAnalyzer: Using MediaPipe solutions FaceDetection (basic)")
            return
        except Exception:
            pass

        self._backend = "none"
        print("    FaceAnalyzer: No face analysis backend available!")

    def analyze_frame(self, frame_bgr: np.ndarray) -> list[FaceData]:
        """
        Analyze a single video frame. Returns a list of FaceData objects,
        one per detected face, with all available signals populated.
        """
        if self._backend == "face_landmarker":
            faces = self._analyze_landmarker(frame_bgr)
        elif self._backend == "face_mesh":
            faces = self._analyze_mesh(frame_bgr)
        elif self._backend == "face_detector_tasks":
            faces = self._analyze_detector_tasks(frame_bgr)
        elif self._backend == "face_detection":
            faces = self._analyze_basic(frame_bgr)
        else:
            return []

        # Assign persistent face IDs by matching positions to previous frame
        self._assign_face_ids(faces)

        # Update lip history and detect speakers
        self._update_lip_history(faces)
        self._detect_speakers(faces)

        self._prev_faces = faces
        return faces

    def _analyze_mesh(self, frame_bgr: np.ndarray) -> list[FaceData]:
        """Full Face Mesh analysis with 478 landmarks."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)

        faces = []
        if not results.multi_face_landmarks:
            return faces

        for face_lm in results.multi_face_landmarks:
            lm = face_lm.landmark
            face = FaceData()

            # ── Bounding box from landmarks ──────────────────────────────
            xs = [p.x for p in lm]
            ys = [p.y for p in lm]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            face.bbox_x = x_min
            face.bbox_y = y_min
            face.bbox_w = x_max - x_min
            face.bbox_h = y_max - y_min
            face.norm_cx = (x_min + x_max) / 2.0
            face.norm_cy = (y_min + y_max) / 2.0
            face.norm_area = face.bbox_w * face.bbox_h

            # ── Lip aperture ─────────────────────────────────────────────
            face.lip_aperture, face.lip_aperture_raw = self._compute_lip_aperture(lm, h)

            # ── Confidence (Face Mesh doesn't give per-detection scores,
            #    so we estimate from landmark visibility) ──────────────────
            key_landmarks = [_NOSE_TIP, _CHIN, _FOREHEAD, _FACE_LEFT, _FACE_RIGHT]
            visibilities = []
            for idx in key_landmarks:
                if hasattr(lm[idx], 'visibility'):
                    visibilities.append(lm[idx].visibility)
            face.confidence = float(np.mean(visibilities)) if visibilities else 0.8

            # ── Frontality ───────────────────────────────────────────────
            face.frontality = self._compute_frontality(lm)

            # ── Eye openness ─────────────────────────────────────────────
            face.eye_openness = self._compute_eye_openness(lm)

            # ── Gaze direction ───────────────────────────────────────────
            face.gaze_at_camera = self._compute_gaze(lm)

            faces.append(face)

        # Sort by area (largest first)
        faces.sort(key=lambda f: f.norm_area, reverse=True)
        return faces

    def _analyze_landmarker(self, frame_bgr: np.ndarray) -> list[FaceData]:
        """Full Face Mesh analysis via MediaPipe Tasks API."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(mp_image)

        faces = []
        if not results.face_landmarks:
            return faces

        for lm in results.face_landmarks:
            face = FaceData()
            xs = [p.x for p in lm]
            ys = [p.y for p in lm]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            face.bbox_x = x_min
            face.bbox_y = y_min
            face.bbox_w = x_max - x_min
            face.bbox_h = y_max - y_min
            face.norm_cx = (x_min + x_max) / 2.0
            face.norm_cy = (y_min + y_max) / 2.0
            face.norm_area = face.bbox_w * face.bbox_h

            face.lip_aperture, face.lip_aperture_raw = self._compute_lip_aperture(lm, h)

            key_landmarks = [_NOSE_TIP, _CHIN, _FOREHEAD, _FACE_LEFT, _FACE_RIGHT]
            visibilities = []
            for idx in key_landmarks:
                if hasattr(lm[idx], 'visibility') and lm[idx].visibility is not None:
                    visibilities.append(lm[idx].visibility)
            face.confidence = float(np.mean(visibilities)) if visibilities else 0.8

            face.frontality = self._compute_frontality(lm)
            face.eye_openness = self._compute_eye_openness(lm)
            face.gaze_at_camera = self._compute_gaze(lm)
            faces.append(face)

        faces.sort(key=lambda f: f.norm_area, reverse=True)
        return faces

    def _analyze_detector_tasks(self, frame_bgr: np.ndarray) -> list[FaceData]:
        """Basic bounding box detection via MediaPipe Tasks API."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        results = self._basic_detector.detect(mp_image)

        faces = []
        if not results.detections:
            return faces

        for det in results.detections:
            bb = det.bounding_box
            norm_x = bb.origin_x / w
            norm_y = bb.origin_y / h
            norm_w = bb.width / w
            norm_h = bb.height / h
            face = FaceData(
                norm_cx=norm_x + norm_w / 2.0,
                norm_cy=norm_y + norm_h / 2.0,
                norm_area=norm_w * norm_h,
                bbox_x=norm_x,
                bbox_y=norm_y,
                bbox_w=norm_w,
                bbox_h=norm_h,
                confidence=det.categories[0].score if det.categories else 0.5,
                frontality=0.7,
                eye_openness=0.8,
                gaze_at_camera=0.5,
                lip_aperture=0.0,
            )
            faces.append(face)

        faces.sort(key=lambda f: f.norm_area, reverse=True)
        return faces

    def _analyze_basic(self, frame_bgr: np.ndarray) -> list[FaceData]:
        """Fallback: basic face detection without landmarks."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._basic_detector.process(rgb)

        faces = []
        if not results.detections:
            return faces

        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            face = FaceData(
                norm_cx=bbox.xmin + bbox.width / 2.0,
                norm_cy=bbox.ymin + bbox.height / 2.0,
                norm_area=bbox.width * bbox.height,
                bbox_x=bbox.xmin,
                bbox_y=bbox.ymin,
                bbox_w=bbox.width,
                bbox_h=bbox.height,
                confidence=det.score[0] if det.score else 0.5,
                # Without landmarks, default to neutral values
                frontality=0.7,
                eye_openness=0.8,
                gaze_at_camera=0.5,
                lip_aperture=0.0,
            )
            faces.append(face)

        faces.sort(key=lambda f: f.norm_area, reverse=True)
        return faces

    # ─── Landmark computations ────────────────────────────────────────────────

    def _compute_lip_aperture(self, landmarks, frame_height: int) -> tuple[float, float]:
        """
        Measure how open the mouth is by computing the distance between
        inner upper and lower lip landmarks, normalized by face height.

        Returns (normalized_aperture, raw_pixel_distance).
        """
        # Inner lip vertical distance (most reliable for speech detection)
        top = landmarks[_LIP_TOP_INNER]
        bottom = landmarks[_LIP_BOTTOM_INNER]
        inner_dist = abs(bottom.y - top.y)

        # Also measure at the left and right inner lip points for robustness
        upper_l = landmarks[_LIP_UPPER_INNER_L]
        lower_l = landmarks[_LIP_LOWER_INNER_L]
        upper_r = landmarks[_LIP_UPPER_INNER_R]
        lower_r = landmarks[_LIP_LOWER_INNER_R]

        left_dist = abs(lower_l.y - upper_l.y)
        right_dist = abs(lower_r.y - upper_r.y)

        # Average the three measurements
        avg_dist = (inner_dist + left_dist + right_dist) / 3.0

        # Normalize by face height (forehead to chin)
        forehead = landmarks[_FOREHEAD]
        chin = landmarks[_CHIN]
        face_height = abs(chin.y - forehead.y)

        if face_height < 0.001:
            return 0.0, 0.0

        normalized = avg_dist / face_height
        raw_px = avg_dist * frame_height

        return float(normalized), float(raw_px)

    def _compute_frontality(self, landmarks) -> float:
        """
        Estimate how frontal the face is by comparing the nose tip's
        horizontal position relative to the face edges.

        A perfectly frontal face has the nose tip centered between
        the left and right face edges. Returns 0.0 (profile) to 1.0 (frontal).
        """
        nose = landmarks[_NOSE_TIP]
        left = landmarks[_FACE_LEFT]
        right = landmarks[_FACE_RIGHT]

        face_width = abs(right.x - left.x)
        if face_width < 0.001:
            return 0.5

        # Distance from nose to face center
        face_center_x = (left.x + right.x) / 2.0
        nose_offset = abs(nose.x - face_center_x)

        # Normalize: 0 offset = frontal (1.0), max offset = profile (0.0)
        max_offset = face_width / 2.0
        frontality = 1.0 - min(nose_offset / max_offset, 1.0)

        # Also check vertical frontality (nose bridge to nose tip angle)
        bridge = landmarks[_NOSE_BRIDGE]
        vertical_offset = abs(nose.x - bridge.x) / max(face_width, 0.001)
        vertical_frontality = 1.0 - min(vertical_offset * 3.0, 1.0)

        # Combine horizontal and vertical
        return float(min(frontality * 0.7 + vertical_frontality * 0.3, 1.0))

    def _compute_eye_openness(self, landmarks) -> float:
        """
        Measure eye openness as the ratio of eye height to eye width.
        Average of both eyes. Returns 0.0 (closed) to 1.0 (fully open).
        """
        def _eye_ratio(top_idx, bottom_idx, left_idx, right_idx):
            top = landmarks[top_idx]
            bottom = landmarks[bottom_idx]
            left = landmarks[left_idx]
            right = landmarks[right_idx]

            eye_h = abs(top.y - bottom.y)
            eye_w = abs(right.x - left.x)
            if eye_w < 0.001:
                return 0.5
            return eye_h / eye_w

        left_ratio = _eye_ratio(
            _LEFT_EYE_TOP, _LEFT_EYE_BOTTOM,
            _LEFT_EYE_LEFT, _LEFT_EYE_RIGHT,
        )
        right_ratio = _eye_ratio(
            _RIGHT_EYE_TOP, _RIGHT_EYE_BOTTOM,
            _RIGHT_EYE_LEFT, _RIGHT_EYE_RIGHT,
        )

        avg_ratio = (left_ratio + right_ratio) / 2.0

        # Typical eye aspect ratio: ~0.2 (closed) to ~0.4 (open)
        # Normalize to 0–1 range
        normalized = (avg_ratio - 0.15) / 0.25
        return float(max(0.0, min(1.0, normalized)))

    def _compute_gaze(self, landmarks) -> float:
        """
        Estimate whether the person is looking at the camera by checking
        iris position relative to the eye socket center.

        Uses refined iris landmarks (468–477) when available.
        Returns 0.0 (looking away) to 1.0 (looking directly at camera).
        """
        try:
            # Iris center landmarks (only available with refine_landmarks=True)
            left_iris = landmarks[_LEFT_IRIS_CENTER]
            right_iris = landmarks[_RIGHT_IRIS_CENTER]

            # Compute eye socket centers from corner landmarks
            left_eye_cx = (landmarks[_LEFT_EYE_LEFT].x + landmarks[_LEFT_EYE_RIGHT].x) / 2.0
            left_eye_cy = (landmarks[_LEFT_EYE_TOP].y + landmarks[_LEFT_EYE_BOTTOM].y) / 2.0

            right_eye_cx = (landmarks[_RIGHT_EYE_LEFT].x + landmarks[_RIGHT_EYE_RIGHT].x) / 2.0
            right_eye_cy = (landmarks[_RIGHT_EYE_TOP].y + landmarks[_RIGHT_EYE_BOTTOM].y) / 2.0

            # Distance from iris to eye center (lower = looking at camera)
            left_offset = math.sqrt(
                (left_iris.x - left_eye_cx) ** 2 +
                (left_iris.y - left_eye_cy) ** 2
            )
            right_offset = math.sqrt(
                (right_iris.x - right_eye_cx) ** 2 +
                (right_iris.y - right_eye_cy) ** 2
            )

            avg_offset = (left_offset + right_offset) / 2.0

            # Normalize: typical eye width is ~0.05 in normalized coords
            eye_w = abs(landmarks[_LEFT_EYE_RIGHT].x - landmarks[_LEFT_EYE_LEFT].x)
            if eye_w < 0.001:
                return 0.5

            normalized_offset = avg_offset / eye_w
            gaze = 1.0 - min(normalized_offset * 4.0, 1.0)
            return float(max(0.0, gaze))

        except (IndexError, AttributeError):
            # Iris landmarks not available
            return 0.5

    # ─── Face ID tracking ─────────────────────────────────────────────────────

    def _assign_face_ids(self, faces: list[FaceData]):
        """
        Assign persistent face IDs by matching current faces to previous
        frame faces based on spatial proximity.

        Uses a greedy nearest-neighbor matching with a distance threshold.
        """
        if not self._prev_faces:
            # First frame — assign new IDs to all faces
            for face in faces:
                face.face_id = self._next_face_id
                self._next_face_id += 1
            return

        # Compute distance matrix (current × previous)
        used_prev = set()
        matches = []

        for curr in faces:
            best_dist = float('inf')
            best_prev = None

            for i, prev in enumerate(self._prev_faces):
                if i in used_prev:
                    continue
                dist = math.sqrt(
                    (curr.norm_cx - prev.norm_cx) ** 2 +
                    (curr.norm_cy - prev.norm_cy) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_prev = i

            if best_prev is not None and best_dist < self._id_match_threshold:
                curr.face_id = self._prev_faces[best_prev].face_id
                used_prev.add(best_prev)
            else:
                curr.face_id = self._next_face_id
                self._next_face_id += 1

    # ─── Lip history & speaker detection ──────────────────────────────────────

    def _update_lip_history(self, faces: list[FaceData]):
        """Track lip aperture over time for each face ID."""
        active_ids = set()
        for face in faces:
            if face.face_id not in self._lip_history:
                self._lip_history[face.face_id] = []

            self._lip_history[face.face_id].append(face.lip_aperture)

            # Trim to max length
            if len(self._lip_history[face.face_id]) > self._lip_history_max:
                self._lip_history[face.face_id] = \
                    self._lip_history[face.face_id][-self._lip_history_max:]

            active_ids.add(face.face_id)

        # Clean up histories for faces that have disappeared for too long
        stale_ids = [fid for fid in self._lip_history if fid not in active_ids]
        for fid in stale_ids:
            # Keep history for a while in case face reappears
            pass

    def _detect_speakers(self, faces: list[FaceData]):
        """
        Determine which face(s) are currently speaking by analyzing
        lip aperture variance over recent history.

        High variance = lips moving = speaking.
        Low variance = lips still = not speaking.
        """
        if not faces:
            return

        # Compute lip activity score for each face
        activities = []
        for face in faces:
            history = self._lip_history.get(face.face_id, [])
            if len(history) < 3:
                # Not enough history — can't determine
                activities.append((face, 0.0))
                continue

            # Variance of lip aperture = how much the mouth is moving
            arr = np.array(history)
            variance = float(np.var(arr))

            # Also consider mean aperture — a consistently open mouth
            # (like laughing) has high mean AND high variance
            mean_aperture = float(np.mean(arr))

            # Combined activity: variance is primary, mean is secondary
            activity = variance * 100.0 + mean_aperture * 0.5
            activities.append((face, activity))

        if not activities:
            return

        # Find the face with highest lip activity
        max_activity = max(a for _, a in activities)

        # Threshold: must have meaningful activity to be considered speaking
        speaking_threshold = 0.025

        for face, activity in activities:
            if max_activity > speaking_threshold and activity >= max_activity * 0.5:
                face.is_speaking = True
            else:
                face.is_speaking = False

    # ─── Cleanup ──────────────────────────────────────────────────────────────

    def close(self):
        """Release resources."""
        if self._backend == "face_landmarker" and hasattr(self, '_landmarker'):
            self._landmarker.close()
        elif self._backend == "face_mesh" and self._mesh:
            self._mesh.close()
        elif self._backend in ("face_detection", "face_detector_tasks") and hasattr(self, '_basic_detector'):
            self._basic_detector.close()

    def reset_tracking(self):
        """Reset face ID tracking and lip history (e.g., between clips)."""
        self._next_face_id = 0
        self._prev_faces = []
        self._lip_history = {}
