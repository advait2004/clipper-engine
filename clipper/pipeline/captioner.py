"""
Captioner — burn styled captions into video
=============================================
Supports multiple caption styles: karaoke, minimal, bold_pop, subtitle_bar, typewriter.
Uses FFmpeg + ASS subtitles for rendering.
"""

import os
import subprocess
import tempfile
import shutil
import math

from config import Config
from pipeline.utils import get_video_codec_args
from pipeline.caption_styles import get_style, build_ass_header
from pipeline.caption_styles import HOOK_OVERLAY_STYLE, CTA_OVERLAY_STYLE


# ─── Public API ───────────────────────────────────────────────────────────────

def burn_all_captions(
    src: str,
    words: list[dict],
    clip_start: float,
    clip_duration: float,
    out_path: str,
    style_name: str      = "karaoke",
    words_per_line: int   = None,
    font_size: int        = None,
    uppercase: bool       = None,
    hook_text_overlay: str = "",
    cta_line: str          = "",
) -> str:
    """
    Burn all caption layers into video in a single FFmpeg pass:
      1. Speech captions (karaoke / typewriter / etc.)
      2. Hook text overlay (0-3s, big bold headline)
      3. CTA overlay (last 2-3s, call-to-action)

    This avoids double-encoding by combining all layers into one ASS file
    with multiple styles.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if not words and not hook_text_overlay and not cta_line:
        shutil.copy2(src, out_path)
        print("    Captions: no content, copied as-is")
        return out_path

    style = get_style(style_name)

    # Allow overrides
    wpl = words_per_line if words_per_line is not None else style["words_per_line"]
    fs = font_size if font_size is not None else style["font_size"]
    uc = uppercase if uppercase is not None else style.get("uppercase", True)

    style_copy = dict(style)
    style_copy["font_size"] = fs

    # ── Build multi-style ASS header ──────────────────────────────────────────
    ass_content = _build_multi_style_header(style_copy)

    # ── 1. Speech captions ────────────────────────────────────────────────────
    if words:
        animation = style.get("animation", "none")
        if animation == "karaoke":
            speech_lines = _build_ass_karaoke_lines(words, clip_start, wpl, uc)
        elif animation == "typewriter":
            speech_lines = _build_ass_typewriter_lines(words, clip_start, wpl, uc)
        elif animation == "scale_pop":
            speech_lines = _build_ass_scale_pop_lines(words, clip_start, wpl, uc)
        elif animation == "fade":
            speech_lines = _build_ass_fade_lines(words, clip_start, wpl, uc)
        else:
            speech_lines = _build_ass_basic_lines(words, clip_start, wpl, uc)
        ass_content += "\n".join(speech_lines) + "\n"

    # ── 2. Hook text overlay (0-3s) ───────────────────────────────────────────
    if hook_text_overlay:
        hook_text = hook_text_overlay.upper()
        s = _fmt_time(0.0)
        e = _fmt_time(3.0)
        # Scale-in (0→110%→100%) + fade-out in last 0.5s
        anim = (
            "{\\fscx0\\fscy0"
            "\\t(0,150,\\fscx110\\fscy110)"
            "\\t(150,250,\\fscx100\\fscy100)"
            "\\t(2500,3000,\\alpha&HFF&)}"
        )
        hook_text_escaped = _escape_ass(hook_text)
        ass_content += f"Dialogue: 1,{s},{e},HookOverlay,,0,0,0,,{anim}{hook_text_escaped}\n"

    # ── 3. CTA overlay (last 2-3s) ───────────────────────────────────────────
    if cta_line and clip_duration > 5.0:
        cta_start = max(0.0, clip_duration - 3.0)
        cta_end = clip_duration
        s = _fmt_time(cta_start)
        e = _fmt_time(cta_end)
        cta_anim = "{\\fad(400,200)}"
        cta_text_escaped = _escape_ass(cta_line)
        ass_content += f"Dialogue: 1,{s},{e},CtaOverlay,,0,0,0,,{cta_anim}{cta_text_escaped}\n"

    # Write ASS to a local temp file
    tmp_ass = os.path.basename(tempfile.mktemp(suffix=".ass", dir="."))
    with open(tmp_ass, "w", encoding="utf-8") as f:
        f.write(ass_content)

    try:
        _burn(src, tmp_ass, out_path)
    finally:
        if os.path.exists(tmp_ass):
            os.remove(tmp_ass)

    overlay_count = sum([bool(hook_text_overlay), bool(cta_line)])
    print(f"    Captions ({style['name']}): {len(words)} words + {overlay_count} overlay(s) burned in")
    return out_path


def burn_captions(
    src: str,
    words: list[dict],
    clip_start: float,
    out_path: str,
    style_name: str      = "karaoke",
    words_per_line: int   = None,
    font_size: int        = None,
    uppercase: bool       = None,
) -> str:
    """
    Burn captions into video using FFmpeg + ASS subtitles.
    Word timestamps are relative to the full video (clip_start offsets them).
    
    If words_per_line/font_size/uppercase are not specified, they come from the style.
    
    Returns out_path on success.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if not words:
        shutil.copy2(src, out_path)
        print("    Captions: no words found, copied as-is")
        return out_path

    style = get_style(style_name)

    # Allow overrides
    wpl = words_per_line if words_per_line is not None else style["words_per_line"]
    fs = font_size if font_size is not None else style["font_size"]
    uc = uppercase if uppercase is not None else style.get("uppercase", True)

    # Override font size in the style for the header
    style_copy = dict(style)
    style_copy["font_size"] = fs

    animation = style.get("animation", "none")

    if animation == "karaoke":
        ass_content = _build_ass_karaoke(words, clip_start, wpl, uc, style_copy)
    elif animation == "typewriter":
        ass_content = _build_ass_typewriter(words, clip_start, wpl, uc, style_copy)
    elif animation == "scale_pop":
        ass_content = _build_ass_scale_pop(words, clip_start, wpl, uc, style_copy)
    elif animation == "fade":
        ass_content = _build_ass_fade(words, clip_start, wpl, uc, style_copy)
    else:
        ass_content = _build_ass_basic(words, clip_start, wpl, uc, style_copy)

    # Write ASS to a local temp file
    tmp_ass = os.path.basename(tempfile.mktemp(suffix=".ass", dir="."))
    with open(tmp_ass, "w", encoding="utf-8") as f:
        f.write(ass_content)

    try:
        _burn(src, tmp_ass, out_path)
    finally:
        if os.path.exists(tmp_ass):
            os.remove(tmp_ass)

    print(f"    Captions ({style['name']}): {len(words)} words burned in")
    return out_path


# ─── ASS Builders ─────────────────────────────────────────────────────────────

def _build_ass_karaoke(
    words: list[dict],
    clip_start: float,
    words_per_line: int,
    uppercase: bool,
    style: dict,
) -> str:
    """
    Karaoke style: each word highlights as it's spoken using ASS \\kf timing.
    Words appear in groups, with the active word colored differently.
    """
    header = build_ass_header(style)
    lines = []

    chunks = _group_words(words, words_per_line)

    for chunk in chunks:
        if not chunk:
            continue

        t_start = max(0.0, chunk[0]["start"] - clip_start)
        t_end = chunk[-1]["end"] - clip_start

        s = _fmt_time(t_start)
        e = _fmt_time(t_end)

        # Build karaoke text with \kf per word
        karaoke_parts = []
        for w in chunk:
            word_text = w["word"].strip()
            if uppercase:
                word_text = word_text.upper()
            # \kf duration in centiseconds
            word_dur = max(0.1, w["end"] - w["start"])
            kf_cs = int(word_dur * 100)
            karaoke_parts.append(f"{{\\kf{kf_cs}}}{word_text}")

        text = " ".join(karaoke_parts)
        text = _escape_ass(text)

        lines.append(f"Dialogue: 0,{s},{e},Default,,0,0,0,,{text}")

    return header + "\n".join(lines) + "\n"


def _build_ass_scale_pop(
    words: list[dict],
    clip_start: float,
    words_per_line: int,
    uppercase: bool,
    style: dict,
) -> str:
    """
    Bold Pop: words scale in from 0% to 100% with a pop effect.
    """
    header = build_ass_header(style)
    lines = []

    chunks = _group_words(words, words_per_line)

    for chunk in chunks:
        if not chunk:
            continue

        t_start = max(0.0, chunk[0]["start"] - clip_start)
        t_end = chunk[-1]["end"] - clip_start

        s = _fmt_time(t_start)
        e = _fmt_time(t_end)

        text = " ".join(w["word"] for w in chunk)
        if uppercase:
            text = text.upper()
        text = _escape_ass(text)

        # Scale-in animation: 0% → 110% → 100% (overshoot bounce)
        pop_dur = 150  # ms
        bounce_dur = 80
        anim = (
            f"{{\\fscx0\\fscy0\\t(0,{pop_dur},\\fscx110\\fscy110)"
            f"\\t({pop_dur},{pop_dur + bounce_dur},\\fscx100\\fscy100)}}"
        )

        lines.append(f"Dialogue: 0,{s},{e},Default,,0,0,0,,{anim}{text}")

    return header + "\n".join(lines) + "\n"


def _build_ass_fade(
    words: list[dict],
    clip_start: float,
    words_per_line: int,
    uppercase: bool,
    style: dict,
) -> str:
    """
    Minimal style: clean text with fade-in/fade-out per line.
    """
    header = build_ass_header(style)
    lines = []

    chunks = _group_words(words, words_per_line)

    for chunk in chunks:
        if not chunk:
            continue

        t_start = max(0.0, chunk[0]["start"] - clip_start)
        t_end = chunk[-1]["end"] - clip_start

        s = _fmt_time(t_start)
        e = _fmt_time(t_end)

        text = " ".join(w["word"] for w in chunk)
        if uppercase:
            text = text.upper()
        text = _escape_ass(text)

        # Fade in 200ms, fade out 200ms
        anim = "{\\fad(200,200)}"

        lines.append(f"Dialogue: 0,{s},{e},Default,,0,0,0,,{anim}{text}")

    return header + "\n".join(lines) + "\n"


def _build_ass_typewriter(
    words: list[dict],
    clip_start: float,
    words_per_line: int,
    uppercase: bool,
    style: dict,
) -> str:
    """
    Typewriter style: words appear one-by-one, accumulating on screen.
    Each line group shows progressive word reveals.
    """
    header = build_ass_header(style)
    lines = []

    chunks = _group_words(words, words_per_line)

    for chunk in chunks:
        if not chunk:
            continue

        # Each word in the chunk gets its own dialogue event
        # showing all words accumulated up to that point
        chunk_end = max(0.5, chunk[-1]["end"] - clip_start)

        for word_idx, w in enumerate(chunk):
            t_start = max(0.0, w["start"] - clip_start)
            t_end = chunk_end

            s = _fmt_time(t_start)
            e = _fmt_time(t_end)

            # Show all words up to and including this one
            accumulated = " ".join(
                cw["word"] for cw in chunk[:word_idx + 1]
            )
            if uppercase:
                accumulated = accumulated.upper()

            # Add blinking cursor
            cursor = "▌" if word_idx < len(chunk) - 1 else ""
            text = _escape_ass(accumulated + cursor)

            lines.append(f"Dialogue: 0,{s},{e},Default,,0,0,0,,{text}")

    return header + "\n".join(lines) + "\n"


def _build_ass_basic(
    words: list[dict],
    clip_start: float,
    words_per_line: int,
    uppercase: bool,
    style: dict,
) -> str:
    """Basic subtitle style (subtitle_bar and fallback)."""
    header = build_ass_header(style)
    lines = []

    chunks = _group_words(words, words_per_line)

    for chunk in chunks:
        if not chunk:
            continue

        t_start = max(0.0, chunk[0]["start"] - clip_start)
        t_end = chunk[-1]["end"] - clip_start

        text = " ".join(w["word"] for w in chunk)
        if uppercase:
            text = text.upper()
        text = _escape_ass(text)

        s = _fmt_time(t_start)
        e = _fmt_time(t_end)
        lines.append(f"Dialogue: 0,{s},{e},Default,,0,0,0,,{text}")

    return header + "\n".join(lines) + "\n"

# ─── Multi-style ASS header (for burn_all_captions) ──────────────────────────

def _build_multi_style_header(
    speech_style: dict,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
) -> str:
    """
    Build an ASS header with three styles:
      - Default (speech captions)
      - HookOverlay (hook text at top)
      - CtaOverlay (CTA text at bottom)
    """
    def _style_line(name, st):
        bold_flag = -1 if st.get("bold") else 0
        border_style = st.get("border_style", 1)
        return (
            f"Style: {name},{st['font_name']},{st['font_size']},"
            f"{st['primary_color']},{st['secondary_color']},"
            f"{st['outline_color']},{st['back_color']},"
            f"{bold_flag},0,0,0,100,100,1,0,"
            f"{border_style},{st['outline']},{st['shadow']},"
            f"{st['alignment']},60,60,{st['margin_v']},1"
        )

    default_line = _style_line("Default", speech_style)
    hook_line = _style_line("HookOverlay", HOOK_OVERLAY_STYLE)
    cta_line = _style_line("CtaOverlay", CTA_OVERLAY_STYLE)

    return f"""\
[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{default_line}
{hook_line}
{cta_line}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


# ─── _lines variants (return raw dialogue lines, no header) ──────────────────

def _build_ass_karaoke_lines(words, clip_start, words_per_line, uppercase):
    """Return karaoke dialogue lines (no header)."""
    lines = []
    chunks = _group_words(words, words_per_line)
    for chunk in chunks:
        if not chunk:
            continue
        t_start = max(0.0, chunk[0]["start"] - clip_start)
        t_end = chunk[-1]["end"] - clip_start
        s = _fmt_time(t_start)
        e = _fmt_time(t_end)
        karaoke_parts = []
        for w in chunk:
            word_text = w["word"].strip()
            if uppercase:
                word_text = word_text.upper()
            word_dur = max(0.1, w["end"] - w["start"])
            kf_cs = int(word_dur * 100)
            karaoke_parts.append(f"{{\\kf{kf_cs}}}{word_text}")
        text = " ".join(karaoke_parts)
        text = _escape_ass(text)
        lines.append(f"Dialogue: 0,{s},{e},Default,,0,0,0,,{text}")
    return lines


def _build_ass_scale_pop_lines(words, clip_start, words_per_line, uppercase):
    """Return scale_pop dialogue lines (no header)."""
    lines = []
    chunks = _group_words(words, words_per_line)
    for chunk in chunks:
        if not chunk:
            continue
        t_start = max(0.0, chunk[0]["start"] - clip_start)
        t_end = chunk[-1]["end"] - clip_start
        s = _fmt_time(t_start)
        e = _fmt_time(t_end)
        text = " ".join(w["word"] for w in chunk)
        if uppercase:
            text = text.upper()
        text = _escape_ass(text)
        pop_dur = 150
        bounce_dur = 80
        anim = (
            f"{{\\fscx0\\fscy0\\t(0,{pop_dur},\\fscx110\\fscy110)"
            f"\\t({pop_dur},{pop_dur + bounce_dur},\\fscx100\\fscy100)}}"
        )
        lines.append(f"Dialogue: 0,{s},{e},Default,,0,0,0,,{anim}{text}")
    return lines


def _build_ass_fade_lines(words, clip_start, words_per_line, uppercase):
    """Return fade dialogue lines (no header)."""
    lines = []
    chunks = _group_words(words, words_per_line)
    for chunk in chunks:
        if not chunk:
            continue
        t_start = max(0.0, chunk[0]["start"] - clip_start)
        t_end = chunk[-1]["end"] - clip_start
        s = _fmt_time(t_start)
        e = _fmt_time(t_end)
        text = " ".join(w["word"] for w in chunk)
        if uppercase:
            text = text.upper()
        text = _escape_ass(text)
        anim = "{\\fad(200,200)}"
        lines.append(f"Dialogue: 0,{s},{e},Default,,0,0,0,,{anim}{text}")
    return lines


def _build_ass_typewriter_lines(words, clip_start, words_per_line, uppercase):
    """Return typewriter dialogue lines (no header)."""
    lines = []
    chunks = _group_words(words, words_per_line)
    for chunk in chunks:
        if not chunk:
            continue
        chunk_end = max(0.5, chunk[-1]["end"] - clip_start)
        for word_idx, w in enumerate(chunk):
            t_start = max(0.0, w["start"] - clip_start)
            t_end = chunk_end
            s = _fmt_time(t_start)
            e = _fmt_time(t_end)
            accumulated = " ".join(cw["word"] for cw in chunk[:word_idx + 1])
            if uppercase:
                accumulated = accumulated.upper()
            cursor = "▌" if word_idx < len(chunk) - 1 else ""
            text = _escape_ass(accumulated + cursor)
            lines.append(f"Dialogue: 0,{s},{e},Default,,0,0,0,,{text}")
    return lines


def _build_ass_basic_lines(words, clip_start, words_per_line, uppercase):
    """Return basic subtitle dialogue lines (no header)."""
    lines = []
    chunks = _group_words(words, words_per_line)
    for chunk in chunks:
        if not chunk:
            continue
        t_start = max(0.0, chunk[0]["start"] - clip_start)
        t_end = chunk[-1]["end"] - clip_start
        text = " ".join(w["word"] for w in chunk)
        if uppercase:
            text = text.upper()
        text = _escape_ass(text)
        s = _fmt_time(t_start)
        e = _fmt_time(t_end)
        lines.append(f"Dialogue: 0,{s},{e},Default,,0,0,0,,{text}")
    return lines


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _group_words(words: list[dict], n: int, max_gap: float = 0.4) -> list[list[dict]]:
    """Group words into chunks of at most n words, or break if there is a silence gap."""
    chunks = []
    current_chunk = []
    
    for word in words:
        if not current_chunk:
            current_chunk.append(word)
            continue
            
        gap = word["start"] - current_chunk[-1]["end"]
        
        if len(current_chunk) >= n or gap > max_gap:
            chunks.append(current_chunk)
            current_chunk = [word]
        else:
            current_chunk.append(word)
            
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks


def _fmt_time(seconds: float) -> str:
    """Format seconds as ASS time: H:MM:SS.cc"""
    s = int(seconds)
    cs = int(round((seconds - s) * 100))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h}:{m:02d}:{sec:02d}.{cs:02d}"


def _escape_ass(text: str) -> str:
    """Escape special ASS characters, but preserve \\kf and \\t tags."""
    # Don't escape braces that are part of ASS override tags
    # Only escape literal braces that aren't part of tags
    text = text.replace("\\", "\\\\")
    # Restore ASS tags that were double-escaped
    text = text.replace("\\\\kf", "\\kf")
    text = text.replace("\\\\t(", "\\t(")
    text = text.replace("\\\\fscx", "\\fscx")
    text = text.replace("\\\\fscy", "\\fscy")
    text = text.replace("\\\\fad(", "\\fad(")
    text = text.replace("\\\\N", "\\N")
    text = text.replace("\n", "\\N")
    return text


# ─── FFmpeg burn ──────────────────────────────────────────────────────────────

def _burn(src: str, ass_path: str, out_path: str):
    """Burn ASS subtitles into video with FFmpeg."""
    escaped = ass_path.replace("\\", "/")
    if ":" in escaped and not escaped.startswith("/"):
        escaped = escaped.replace(":", "\\:", 1)

    # Point to our bundled fonts directory
    fonts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fonts")).replace("\\", "/")
    if ":" in fonts_dir and not fonts_dir.startswith("/"):
        fonts_dir = fonts_dir.replace(":", "\\:", 1)

    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-vf", f"ass='{escaped}':fontsdir='{fonts_dir}'",
        *get_video_codec_args(Config()),
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        out_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Caption burn failed:\n{result.stderr[-500:]}")
