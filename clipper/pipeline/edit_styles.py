"""
Edit Style Presets — pacing, AI behavior, and default settings per edit type
============================================================================
Each preset controls:
  - Target duration (dynamic based on content)
  - Segment length range
  - Default transitions and captions
  - AI system prompt customization
"""


EDIT_STYLES = {
    # ─── Podcast ──────────────────────────────────────────────────────────────
    "podcast": {
        "name": "Podcast",
        "icon": "🎙️",
        "description": "Story-driven, relaxed pacing with narrative arcs",
        "duration_ratio": 0.25,           # Use ~25% of total source content
        "min_duration": 45,
        "max_duration": 120,
        "min_segment": 10,
        "max_segment": 40,
        "default_transition": "crossfade",
        "default_caption": "subtitle_bar",
        "preferred_transitions": ["crossfade", "fade_to_black", "cut"],
        "ai_prompt_additions": """
## Edit Style: PODCAST
- Build a narrative arc: hook → context → escalation → payoff
- Use LONG segments (15–40s) that let stories breathe
- Prioritize: storytelling moments, controversial opinions, hot takes, back-and-forth banter
- Avoid cutting mid-story or mid-argument
- Prefer crossfade transitions between different speakers/topics
- Fade to black between completely different topics
- Keep conversational flow natural — don't chop up dialogue unnaturally
""",
    },

    # ─── Fast / Hype ──────────────────────────────────────────────────────────
    "fast": {
        "name": "Fast / Hype",
        "icon": "⚡",
        "description": "Rapid-fire cuts, maximum energy, short punchy segments",
        "duration_ratio": 0.10,           # Use ~10% of total content (very selective)
        "min_duration": 15,
        "max_duration": 45,
        "min_segment": 3,
        "max_segment": 10,
        "default_transition": "whip_pan",
        "default_caption": "bold_pop",
        "preferred_transitions": ["whip_pan", "zoom", "glitch", "cut"],
        "ai_prompt_additions": """
## Edit Style: FAST / HYPE
- Maximum energy. Every second must hit hard.
- Use SHORT segments (3–10s). Cut RUTHLESSLY — only the peaks.
- Start with the MOST SHOCKING or HIGH-ENERGY moment from ANY clip
- Prioritize: energy peaks, fast speech, punchlines, shocking statements, emotional outbursts
- Use rapid transitions: whip pan, zoom, glitch
- Cut on beat — transition at moments of emphasis (key words, volume spikes)
- Leave NO dead air. If someone pauses, skip to the next peak moment.
- This should feel like a highlight reel / dopamine rush
""",
    },

    # ─── Cinematic ────────────────────────────────────────────────────────────
    "cinematic": {
        "name": "Cinematic",
        "icon": "🎬",
        "description": "Slow, dramatic pacing with emotional weight",
        "duration_ratio": 0.20,
        "min_duration": 30,
        "max_duration": 90,
        "min_segment": 8,
        "max_segment": 30,
        "default_transition": "crossfade",
        "default_caption": "minimal",
        "preferred_transitions": ["crossfade", "fade_to_black"],
        "ai_prompt_additions": """
## Edit Style: CINEMATIC
- Slow and dramatic. Let moments BREATHE.
- Use medium-to-long segments (8–30s) with emotional weight
- Prioritize: emotional revelations, dramatic pauses, vulnerable moments, visual beauty
- Let dramatic pauses stay in — silence is powerful
- Use slow crossfades and fade-to-black for gravitas
- Build tension: start quiet, escalate, hit the emotional climax
- Think of this as a movie trailer — every segment should feel significant
""",
    },

    # ─── Vlog ─────────────────────────────────────────────────────────────────
    "vlog": {
        "name": "Vlog",
        "icon": "📹",
        "description": "Natural, medium pacing with personality",
        "duration_ratio": 0.20,
        "min_duration": 30,
        "max_duration": 75,
        "min_segment": 5,
        "max_segment": 20,
        "default_transition": "crossfade",
        "default_caption": "karaoke",
        "preferred_transitions": ["cut", "crossfade", "zoom"],
        "ai_prompt_additions": """
## Edit Style: VLOG
- Natural and personal. Keep the speaker's personality front and center.
- Use medium segments (5–20s) with a casual, relatable feel
- Prioritize: personal stories, funny reactions, relatable moments, genuine emotion
- Keep jump cuts for comedic timing
- Use crossfade only between different topics or locations
- The viewer should feel like they're hanging out with the creator
- Include funny mistakes or candid moments if they exist
""",
    },

    # ─── Documentary ──────────────────────────────────────────────────────────
    "documentary": {
        "name": "Documentary",
        "icon": "📖",
        "description": "Measured, informative pacing with educational value",
        "duration_ratio": 0.30,
        "min_duration": 60,
        "max_duration": 180,
        "min_segment": 10,
        "max_segment": 45,
        "default_transition": "crossfade",
        "default_caption": "subtitle_bar",
        "preferred_transitions": ["crossfade", "cut", "fade_to_black"],
        "ai_prompt_additions": """
## Edit Style: DOCUMENTARY
- Measured and informative. Every segment should teach or reveal something.
- Use long segments (10–45s) that fully explain a point
- Prioritize: facts, revelations, expert explanations, historical context, "I didn't know that" moments
- Build a logical argument: premise → evidence → conclusion
- Use crossfade between different topics or perspectives
- Fade to black between major topic shifts
- Maintain credibility — don't chop quotes out of context
- Include counter-arguments or nuance if they exist in the content
""",
    },
}


def get_style(name: str) -> dict:
    """Get an edit style by name. Falls back to 'podcast'."""
    return EDIT_STYLES.get(name, EDIT_STYLES["podcast"])


def list_styles() -> list[dict]:
    """Return edit style summaries for the API."""
    return [
        {
            "id": key,
            "name": style["name"],
            "icon": style["icon"],
            "description": style["description"],
            "default_caption": style["default_caption"],
            "default_transition": style["default_transition"],
        }
        for key, style in EDIT_STYLES.items()
    ]


def compute_target_duration(
    edit_style_name: str,
    total_source_duration: float,
) -> tuple[float, float]:
    """
    Compute the target output duration range based on edit style and
    total source content duration.
    
    Returns (min_duration, max_duration) in seconds.
    
    The ratio approach means:
    - 5 clips × 60s = 300s source → Podcast 25% → 45–75s target
    - 10 clips × 60s = 600s source → Fast 10% → 30–60s target
    - 3 clips × 120s = 360s source → Documentary 30% → 60–108s target
    
    Always clamped to the style's absolute min/max.
    """
    style = get_style(edit_style_name)
    ratio = style["duration_ratio"]
    abs_min = style["min_duration"]
    abs_max = style["max_duration"]

    # Dynamic target based on content length
    dynamic_target = total_source_duration * ratio

    # Clamp to absolute bounds
    target_min = max(abs_min, dynamic_target * 0.7)
    target_max = min(abs_max, dynamic_target * 1.3)

    # Ensure min <= max
    if target_min > target_max:
        target_min = abs_min
        target_max = abs_max

    return round(target_min, 1), round(target_max, 1)
