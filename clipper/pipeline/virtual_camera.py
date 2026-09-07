"""
Virtual Camera — Cinematic Panning with Ease-In-Out Motion
============================================================
Treats the crop window as a virtual camera and applies professional
camera operator rules:

1. Hold delay: pause briefly before panning to a new speaker
2. Ease-in-out motion: slow start → fast middle → slow stop (Hermite smoothstep)
3. Velocity cap: maximum pan speed prevents frantic motion
4. Rapid alternation damping: if speakers switch too fast, hold on the
   longer speaker rather than chasing every exchange

Produces FFmpeg-ready keyframes with cinematic interpolation baked in.
"""

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SpeakerChange:
    """Records a speaker change event."""
    time: float             # When the change was detected
    new_speaker_id: int     # Face ID of the new speaker
    new_target_cx: float    # Normalized x position of the new speaker's face


@dataclass
class CameraKeyframe:
    """A single keyframe for the virtual camera position."""
    time: float             # Absolute time in seconds
    crop_cx: float          # Normalized crop center x (0–1)


class VirtualCamera:
    """
    Manages smooth, cinematic crop position transitions when the
    active speaker changes.
    """

    def __init__(
        self,
        hold_delay: float = 0.3,
        max_velocity: float = 0.30,
        min_pan_duration: float = 0.4,
        max_pan_duration: float = 1.5,
        rapid_switch_window: float = 2.0,
        rapid_switch_max: int = 3,
    ):
        """
        Args:
            hold_delay: Seconds to hold on current speaker before panning.
            max_velocity: Max pan speed (fraction of frame width per second).
            min_pan_duration: Minimum duration of a pan (seconds).
            max_pan_duration: Maximum duration of a pan (seconds).
            rapid_switch_window: Window (seconds) to detect rapid alternation.
            rapid_switch_max: Max switches in window before damping kicks in.
        """
        self.hold_delay = hold_delay
        self.max_velocity = max_velocity
        self.min_pan_duration = min_pan_duration
        self.max_pan_duration = max_pan_duration
        self.rapid_switch_window = rapid_switch_window
        self.rapid_switch_max = rapid_switch_max

        # State
        self._current_cx = 0.5
        self._current_speaker_id = -1
        self._recent_switches: list[float] = []  # timestamps of recent switches
        self._speaker_durations: dict[int, float] = {}  # cumulative time per speaker

    def compute_keyframes(
        self,
        speaker_timeline: list[dict],
        sample_interval: float = 0.25,
    ) -> list[CameraKeyframe]:
        """
        Convert a speaker timeline into smooth camera keyframes.

        Args:
            speaker_timeline: List of dicts with keys:
                - "time": float (seconds)
                - "speaker_id": int (face ID of active speaker, or -1 for none)
                - "target_cx": float (normalized x of the speaker's face)
            sample_interval: Time between samples in the timeline.

        Returns:
            List of CameraKeyframe with ease-in-out interpolation applied.
        """
        if not speaker_timeline:
            return [CameraKeyframe(time=0.0, crop_cx=0.5)]

        # Start at the first speaker's position
        first_entry = speaker_timeline[0]
        self._current_cx = first_entry.get("target_cx", 0.5)
        self._current_speaker_id = first_entry.get("speaker_id", -1)

        raw_keyframes = [CameraKeyframe(
            time=first_entry["time"],
            crop_cx=self._current_cx,
        )]

        # Track speaker changes
        pending_pan: dict | None = None  # {start_time, start_cx, end_cx, duration}

        for entry in speaker_timeline[1:]:
            t = entry["time"]
            speaker_id = entry.get("speaker_id", -1)
            target_cx = entry.get("target_cx", self._current_cx)

            # Update speaker durations
            if speaker_id >= 0:
                self._speaker_durations[speaker_id] = \
                    self._speaker_durations.get(speaker_id, 0) + sample_interval

            # ── Handle active pan ─────────────────────────────────────────
            if pending_pan is not None:
                pan = pending_pan
                pan_progress = (t - pan["pan_start"]) / pan["duration"]

                if pan_progress >= 1.0:
                    # Pan complete — snap to target
                    self._current_cx = pan["end_cx"]
                    raw_keyframes.append(CameraKeyframe(
                        time=pan["pan_start"] + pan["duration"],
                        crop_cx=self._current_cx,
                    ))
                    pending_pan = None
                else:
                    # Mid-pan — let it continue (easing is applied later)
                    continue

            # ── Detect speaker change ─────────────────────────────────────
            if speaker_id != self._current_speaker_id and speaker_id >= 0:
                # Check for rapid alternation
                if self._should_damp(t, speaker_id):
                    continue  # Ignore this switch

                # Record the switch
                self._recent_switches.append(t)
                self._recent_switches = [
                    s for s in self._recent_switches
                    if t - s <= self.rapid_switch_window
                ]

                # Calculate pan parameters
                distance = abs(target_cx - self._current_cx)
                if distance < 0.02:
                    # Negligible movement — just update speaker ID
                    self._current_speaker_id = speaker_id
                    continue

                pan_duration = self._compute_pan_duration(distance)
                pan_start = t + self.hold_delay

                # Add hold keyframe (camera stays put during delay)
                raw_keyframes.append(CameraKeyframe(
                    time=pan_start,
                    crop_cx=self._current_cx,
                ))

                # Set up the pending pan
                pending_pan = {
                    "pan_start": pan_start,
                    "start_cx": self._current_cx,
                    "end_cx": target_cx,
                    "duration": pan_duration,
                }

                self._current_speaker_id = speaker_id

            elif speaker_id == self._current_speaker_id and speaker_id >= 0:
                # Same speaker — track gentle position drift
                drift = target_cx - self._current_cx
                if abs(drift) > 0.03:
                    # Gentle follow (not a full pan, just subtle tracking)
                    self._current_cx += drift * 0.1
                    raw_keyframes.append(CameraKeyframe(
                        time=t,
                        crop_cx=self._current_cx,
                    ))

        # Finalize any pending pan
        if pending_pan is not None:
            pan = pending_pan
            raw_keyframes.append(CameraKeyframe(
                time=pan["pan_start"] + pan["duration"],
                crop_cx=pan["end_cx"],
            ))

        # Apply ease-in-out to all pans
        eased = self._apply_easing(raw_keyframes)

        # Deduplicate and sort
        eased.sort(key=lambda kf: kf.time)
        return eased

    def _should_damp(self, current_time: float, new_speaker_id: int) -> bool:
        """
        Determine if this speaker switch should be damped (ignored)
        due to rapid alternation.

        If switches happen faster than rapid_switch_max times within
        rapid_switch_window, prioritize the speaker with the most
        cumulative speaking time.
        """
        recent = [
            s for s in self._recent_switches
            if current_time - s <= self.rapid_switch_window
        ]

        if len(recent) >= self.rapid_switch_max:
            # Too many switches — should we follow this one?
            # Follow only if the new speaker has MORE cumulative time
            current_dur = self._speaker_durations.get(self._current_speaker_id, 0)
            new_dur = self._speaker_durations.get(new_speaker_id, 0)

            if new_dur <= current_dur:
                return True  # Damp: new speaker hasn't talked enough

        return False

    def _compute_pan_duration(self, distance: float) -> float:
        """
        Calculate pan duration based on distance and velocity constraints.

        Longer distances get longer pans, but capped by max_velocity.
        """
        # Duration = distance / velocity
        duration = distance / self.max_velocity

        # Clamp to min/max bounds
        duration = max(self.min_pan_duration, min(duration, self.max_pan_duration))

        return duration

    def _apply_easing(self, keyframes: list[CameraKeyframe]) -> list[CameraKeyframe]:
        """
        Apply cubic ease-in-out (Hermite smoothstep) to pan movements.

        For each pair of keyframes where the position changes, insert
        intermediate keyframes with eased positions.
        """
        if len(keyframes) <= 1:
            return keyframes

        eased = [keyframes[0]]
        EASE_STEPS = 8  # Intermediate points per pan

        for i in range(1, len(keyframes)):
            prev = keyframes[i - 1]
            curr = keyframes[i]

            distance = abs(curr.crop_cx - prev.crop_cx)
            dt = curr.time - prev.time

            if distance < 0.01 or dt <= 0:
                # No significant movement — just add the keyframe
                eased.append(curr)
                continue

            # Insert eased intermediate keyframes
            for step in range(1, EASE_STEPS + 1):
                t_frac = step / EASE_STEPS
                t_abs = prev.time + t_frac * dt

                # Hermite smoothstep: 3t² - 2t³
                eased_t = _smoothstep(t_frac)
                eased_cx = prev.crop_cx + (curr.crop_cx - prev.crop_cx) * eased_t

                eased.append(CameraKeyframe(time=round(t_abs, 3), crop_cx=eased_cx))

        return eased

    def reset(self):
        """Reset camera state (e.g., between clips)."""
        self._current_cx = 0.5
        self._current_speaker_id = -1
        self._recent_switches = []
        self._speaker_durations = {}


def _smoothstep(t: float) -> float:
    """
    Hermite smoothstep: f(t) = 3t² - 2t³

    Maps [0, 1] → [0, 1] with ease-in-out behavior:
    - Slow at t=0 (derivative = 0)
    - Fastest at t=0.5
    - Slow at t=1 (derivative = 0)
    """
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _smootherstep(t: float) -> float:
    """
    Ken Perlin's smootherstep: f(t) = 6t⁵ - 15t⁴ + 10t³

    Even smoother than smoothstep — second derivative is also 0 at endpoints.
    Use this for longer, more cinematic pans.
    """
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def build_speaker_timeline_from_faces(
    face_timeline: list[dict],
) -> list[dict]:
    """
    Convert a face analysis timeline into a speaker timeline suitable
    for VirtualCamera.compute_keyframes().

    Args:
        face_timeline: List of dicts with keys:
            - "time": float
            - "faces": list[FaceData]
            - "primary_speaker_id": int (face ID of active speaker)

    Returns:
        List of dicts with:
            - "time": float
            - "speaker_id": int
            - "target_cx": float
    """
    speaker_timeline = []
    last_valid_cx = None

    for entry in face_timeline:
        t = entry["time"]
        faces = entry.get("faces", [])
        speaker_id = entry.get("primary_speaker_id", -1)

        target_cx = None
        for face in faces:
            if face.face_id == speaker_id:
                target_cx = face.norm_cx
                break

        if target_cx is None and faces:
            # If no active speaker (or speaker ID not matched), hold on last valid cx if available,
            # otherwise lock onto the highest-ranked subject (faces[0])!
            target_cx = last_valid_cx if last_valid_cx is not None else faces[0].norm_cx
        elif target_cx is None:
            # If no faces detected at all in this frame, hold on last valid position
            target_cx = last_valid_cx if last_valid_cx is not None else 0.5

        last_valid_cx = target_cx

        speaker_timeline.append({
            "time": t,
            "speaker_id": speaker_id,
            "target_cx": target_cx,
        })

    return speaker_timeline
