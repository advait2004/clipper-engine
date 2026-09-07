"""
Caption Styles — Multiple ASS subtitle style presets
=====================================================
Defines caption rendering configurations for different visual styles.
Each style controls font, color, size, animation, positioning, and word grouping.
"""


CAPTION_STYLES = {
    # ─── 1. Karaoke — word-by-word highlight ─────────────────────────────────
    "karaoke": {
        "name": "Karaoke",
        "description": "Word-by-word highlight as spoken",
        "font_name": "Montserrat",
        "font_size": 85,
        "primary_color": "&H0000FFFF",      # Yellow (active word via \kf)
        "secondary_color": "&H00FFFFFF",     # White (inactive/swept)
        "outline_color": "&H00000000",       # Black outline
        "back_color": "&H66000000",          # Semi-transparent shadow
        "bold": True,
        "outline": 8,
        "shadow": 4,
        "alignment": 2,                      # Bottom center
        "margin_v": 350,
        "words_per_line": 2,
        "uppercase": True,
        "animation": "karaoke",
    },

    # ─── 2. Minimal — clean, modern, subtle ──────────────────────────────────
    "minimal": {
        "name": "Minimal",
        "description": "Clean, thin font with fade-in/out",
        "font_name": "Oswald",
        "font_size": 50,
        "primary_color": "&H00FFFFFF",       # White
        "secondary_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
        "back_color": "&H80000000",          # Semi-transparent background
        "bold": False,
        "outline": 0,
        "shadow": 0,
        "alignment": 1,                      # Bottom left
        "margin_v": 300,
        "words_per_line": 5,
        "uppercase": False,
        "animation": "fade",
        "border_style": 3,                   # Opaque box behind text
    },

    # ─── 3. Bold Pop — TikTok-style animated ────────────────────────────────
    "bold_pop": {
        "name": "Bold Pop",
        "description": "ALL CAPS scale-in animation, TikTok style",
        "font_name": "Anton",
        "font_size": 100,
        "primary_color": "&H0000FFFF",       # Yellow
        "secondary_color": "&H000080FF",     # Orange
        "outline_color": "&H00000000",       # Black
        "back_color": "&H00000000",
        "bold": True,
        "outline": 10,
        "shadow": 5,
        "alignment": 5,                      # Center of screen
        "margin_v": 100,
        "words_per_line": 2,
        "uppercase": True,
        "animation": "scale_pop",
    },

    # ─── 4. Subtitle Bar — professional with dark background ─────────────────
    "subtitle_bar": {
        "name": "Subtitle Bar",
        "description": "Traditional subtitles with dark background bar",
        "font_name": "Arial",
        "font_size": 55,
        "primary_color": "&H00FFFFFF",       # White
        "secondary_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
        "back_color": "&HC0000000",          # Dark semi-transparent
        "bold": False,
        "outline": 0,
        "shadow": 0,
        "alignment": 2,                      # Bottom center
        "margin_v": 250,
        "words_per_line": 7,
        "uppercase": False,
        "animation": "none",
        "border_style": 3,                   # Opaque box
    },

    # ─── 5. Typewriter — words appear one by one ─────────────────────────────
    "typewriter": {
        "name": "Typewriter",
        "description": "Words appear one by one, terminal aesthetic",
        "font_name": "Courier New",
        "font_size": 58,
        "primary_color": "&H0000FF00",       # Green
        "secondary_color": "&H0000FF00",
        "outline_color": "&H00000000",
        "back_color": "&HC0000000",
        "bold": True,
        "outline": 2,
        "shadow": 0,
        "alignment": 1,                      # Bottom left
        "margin_v": 300,
        "words_per_line": 4,
        "uppercase": False,
        "animation": "typewriter",
        "border_style": 3,
    },
}

# ─── Internal overlay styles (not user-selectable) ────────────────────────────

HOOK_OVERLAY_STYLE = {
    "name": "HookOverlay",
    "font_name": "Anton",
    "font_size": 110,
    "primary_color": "&H00FFFFFF",       # White
    "secondary_color": "&H00FFFFFF",
    "outline_color": "&H00000000",       # Black
    "back_color": "&H00000000",
    "bold": True,
    "outline": 10,
    "shadow": 6,
    "alignment": 8,                      # Top center
    "margin_v": 280,
    "words_per_line": 8,
    "uppercase": True,
    "animation": "none",
}

CTA_OVERLAY_STYLE = {
    "name": "CtaOverlay",
    "font_name": "Montserrat",
    "font_size": 48,
    "primary_color": "&H00FFFFFF",       # White
    "secondary_color": "&H00FFFFFF",
    "outline_color": "&H00000000",       # Black
    "back_color": "&H80000000",          # Semi-transparent
    "bold": False,
    "outline": 3,
    "shadow": 2,
    "alignment": 2,                      # Bottom center
    "margin_v": 120,
    "words_per_line": 10,
    "uppercase": False,
    "animation": "none",
    "border_style": 3,                   # Opaque box behind text
}


def get_style(name: str) -> dict:
    """Get a caption style by name. Falls back to 'karaoke' if not found."""
    return CAPTION_STYLES.get(name, CAPTION_STYLES["karaoke"])


def list_styles() -> list[dict]:
    """Return a list of style summaries for the API."""
    return [
        {
            "id": key,
            "name": style["name"],
            "description": style["description"],
        }
        for key, style in CAPTION_STYLES.items()
    ]


def build_ass_header(style: dict, play_res_x: int = 1080, play_res_y: int = 1920) -> str:
    """Build the ASS file header and style definition from a style config."""
    bold_flag = -1 if style.get("bold") else 0
    border_style = style.get("border_style", 1)

    return f"""\
[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style['font_name']},{style['font_size']},{style['primary_color']},{style['secondary_color']},{style['outline_color']},{style['back_color']},{bold_flag},0,0,0,100,100,1,0,{border_style},{style['outline']},{style['shadow']},{style['alignment']},60,60,{style['margin_v']},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
