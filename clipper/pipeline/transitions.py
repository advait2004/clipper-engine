"""
Transition Styles — FFmpeg xfade filter implementations
========================================================
Each transition returns the FFmpeg filter string for use in filter_complex chains.
Supports both video (xfade) and audio (acrossfade) transitions.
"""

# ─── Available Transitions ────────────────────────────────────────────────────

TRANSITIONS = {
    "cut": {
        "name": "Cut",
        "description": "Hard cut, no transition effect",
        "xfade_type": None,
        "duration": 0.0,
    },
    "crossfade": {
        "name": "Crossfade",
        "description": "Smooth dissolve between clips",
        "xfade_type": "fade",
        "duration": 0.5,
    },
    "zoom": {
        "name": "Zoom",
        "description": "Circle crop zoom transition",
        "xfade_type": "circlecrop",
        "duration": 0.4,
    },
    "whip_pan": {
        "name": "Whip Pan",
        "description": "Fast slide left like a camera whip",
        "xfade_type": "slideleft",
        "duration": 0.3,
    },
    "glitch": {
        "name": "Glitch",
        "description": "Pixelized glitch transition",
        "xfade_type": "pixelize",
        "duration": 0.3,
    },
    "fade_to_black": {
        "name": "Fade to Black",
        "description": "Fade out to black, then fade in",
        "xfade_type": "fadeblack",
        "duration": 0.8,
    },
}


def get_transition(name: str) -> dict:
    """Get a transition config by name. Falls back to 'crossfade'."""
    return TRANSITIONS.get(name, TRANSITIONS["crossfade"])


def list_transitions() -> list[dict]:
    """Return transition summaries for the API."""
    return [
        {
            "id": key,
            "name": t["name"],
            "description": t["description"],
            "duration": t["duration"],
        }
        for key, t in TRANSITIONS.items()
    ]


def build_xfade_filter(
    in_a: str,
    in_b: str,
    out_label: str,
    transition_type: str,
    offset: float,
) -> str | None:
    """
    Build an FFmpeg xfade filter string between two video streams.
    
    Args:
        in_a: Input label for first stream (e.g., "[v0]" or "[xf0]")
        in_b: Input label for second stream (e.g., "[v1]")
        out_label: Output label (e.g., "[xf1]")
        transition_type: One of the TRANSITIONS keys
        offset: Time offset (seconds) where the transition should start
    
    Returns:
        Filter string like "[v0][v1]xfade=transition=fade:duration=0.5:offset=5[xf1]"
        or None for hard cuts (no filter needed).
    """
    cfg = get_transition(transition_type)

    if cfg["xfade_type"] is None:
        # Hard cut — no xfade filter needed
        return None

    duration = cfg["duration"]
    xfade_type = cfg["xfade_type"]

    return (
        f"{in_a}{in_b}xfade=transition={xfade_type}"
        f":duration={duration}:offset={offset:.3f}{out_label}"
    )


def build_audio_crossfade_filter(
    in_a: str,
    in_b: str,
    out_label: str,
    transition_type: str,
) -> str | None:
    """
    Build an FFmpeg acrossfade filter for audio transition.
    Returns None for hard cuts.
    """
    cfg = get_transition(transition_type)

    if cfg["xfade_type"] is None:
        return None

    duration = cfg["duration"]

    return (
        f"{in_a}{in_b}acrossfade=d={duration}:c1=tri:c2=tri{out_label}"
    )


def build_transition_chain(
    segment_count: int,
    segment_durations: list[float],
    transition_types: list[str],
) -> tuple[str, str, str]:
    """
    Build the complete FFmpeg filter_complex string for chaining
    multiple segments with transitions.
    
    Args:
        segment_count: Number of video segments
        segment_durations: Duration of each segment in seconds
        transition_types: Transition type between each pair (length = segment_count - 1)
    
    Returns:
        Tuple of (filter_complex_string, final_video_label, final_audio_label)
    """
    if segment_count <= 0:
        raise ValueError("Need at least 1 segment")

    if segment_count == 1:
        return "", "[0:v]", "[0:a]"

    video_filters = []
    audio_filters = []

    # Track cumulative offset for xfade timing
    cumulative_duration = 0.0

    for i in range(segment_count - 1):
        trans_type = transition_types[i] if i < len(transition_types) else "crossfade"
        trans_cfg = get_transition(trans_type)
        trans_dur = trans_cfg["duration"]

        # Input labels
        if i == 0:
            v_in_a = f"[{i}:v]"
            a_in_a = f"[{i}:a]"
        else:
            v_in_a = f"[xfv{i-1}]"
            a_in_a = f"[xfa{i-1}]"

        v_in_b = f"[{i+1}:v]"
        a_in_b = f"[{i+1}:a]"

        # Output labels
        v_out = f"[xfv{i}]"
        a_out = f"[xfa{i}]"

        # Calculate offset: where in the output timeline the transition starts
        offset = cumulative_duration + segment_durations[i] - trans_dur
        offset = max(0, offset)

        # Video transition
        v_filter = build_xfade_filter(v_in_a, v_in_b, v_out, trans_type, offset)
        if v_filter:
            video_filters.append(v_filter)
        else:
            # Hard cut: just concat
            video_filters.append(
                f"{v_in_a}{v_in_b}concat=n=2:v=1:a=0{v_out}"
            )

        # Audio transition
        a_filter = build_audio_crossfade_filter(a_in_a, a_in_b, a_out, trans_type)
        if a_filter:
            audio_filters.append(a_filter)
        else:
            audio_filters.append(
                f"{a_in_a}{a_in_b}concat=n=2:v=0:a=1{a_out}"
            )

        # Update cumulative duration (subtract transition overlap)
        cumulative_duration += segment_durations[i] - trans_dur

    # Final labels
    final_v = f"[xfv{segment_count - 2}]"
    final_a = f"[xfa{segment_count - 2}]"

    all_filters = video_filters + audio_filters
    filter_complex = ";".join(all_filters)

    return filter_complex, final_v, final_a
