"""
AI Story Planner — the brain of the auto-editor
=================================================
Sends all clip profiles + edit style to the AI.
Returns an Edit Decision List (EDL): which segments from which clips,
in what order, with what transitions.

Target duration is DYNAMIC — computed from total source content and edit style.
"""

import json
import re
import time
import traceback

from config import Config
from pipeline.edit_styles import get_style, compute_target_duration


# ─── Prompt builder ───────────────────────────────────────────────────────────

def _build_planner_prompt(
    profiles: list[dict],
    edit_style_name: str,
    total_source_duration: float,
    preferred_transitions: list[str],
) -> str:
    """Build the complete prompt for the AI story planner."""
    style = get_style(edit_style_name)
    target_min, target_max = compute_target_duration(edit_style_name, total_source_duration)

    # Build clip descriptions
    clip_descriptions = []
    for i, p in enumerate(profiles):
        desc = f"Clip {i+1} ({p['file']}, {p['duration']:.1f}s): \"{p['summary']}\"\n"
        desc += f"  Transcript: {p['transcript_text'][:500]}\n"

        if p.get("audio_hints"):
            desc += f"  {p['audio_hints'][:200]}\n"

        clip_descriptions.append(desc)

    clips_text = "\n".join(clip_descriptions)
    transitions_text = ", ".join(preferred_transitions)

    prompt = f"""You are a professional short-form video editor creating a viral clip.

You have these source clips from the same creator/topic:

{clips_text}

Total source material: {total_source_duration:.0f} seconds across {len(profiles)} clips.

{style['ai_prompt_additions']}

## Instructions
Create an edit plan for a video between {target_min:.0f}–{target_max:.0f} seconds long.

Pick the BEST segments from ANY of the source clips and arrange them for maximum impact.
The target duration should reflect the quality and density of the content:
- If there's lots of great content, lean toward {target_max:.0f}s
- If content is sparse or repetitive, lean toward {target_min:.0f}s
- Never pad with boring content just to hit a target

## Rules
- Pick segments from ANY clip — you don't have to use all clips
- Each segment must be {style['min_segment']}–{style['max_segment']} seconds
- Segments from the SAME source clip must not overlap
- Start and end on natural speech boundaries (not mid-word/sentence)
- Start with the STRONGEST hook from ANY clip
- The total assembled video should be {target_min:.0f}–{target_max:.0f} seconds
- Choose transitions from: {transitions_text}
- Use "cut" between segments from the same clip, other transitions between different clips

## Return Format
Return ONLY a valid JSON array. No markdown, no explanation.
Each object:
{{
    "source_file": "filename.mp4",
    "start": 0.0,
    "end": 15.5,
    "reason": "Strong opening hook — grabs attention",
    "transition": "crossfade"
}}

The first segment's transition should be "none".
Return ONLY the JSON array."""

    return prompt


# ─── Plan via AI ──────────────────────────────────────────────────────────────

def plan_edit(
    profiles: list[dict],
    cfg: Config,
    edit_style_name: str = "podcast",
    variation_index: int = 0,
) -> list[dict]:
    """
    Use AI to create an edit decision list (EDL) from clip profiles.
    
    Returns a list of edit segments:
    [
        {
            "source_file": "002.mp4",
            "start": 0.0,
            "end": 8.5,
            "reason": "Strong opening hook",
            "transition": "none",
        },
        ...
    ]
    """
    style = get_style(edit_style_name)
    total_duration = sum(p["duration"] for p in profiles)
    target_min, target_max = compute_target_duration(edit_style_name, total_duration)

    print(f"\n  ── AI Story Planner ──")
    print(f"  Style: {style['name']} ({style['icon']})")
    print(f"  Source: {total_duration:.0f}s across {len(profiles)} clips")
    print(f"  Target: {target_min:.0f}–{target_max:.0f}s (dynamic)")

    prompt = _build_planner_prompt(
        profiles,
        edit_style_name,
        total_duration,
        style["preferred_transitions"],
    )

    if variation_index > 0:
        prompt += f"\n\nIMPORTANT: This is variation #{variation_index + 1}. Please choose a DIFFERENT set of clips, a DIFFERENT hook, and a DIFFERENT narrative flow than the most obvious choices, while still making it engaging."

    temperature = 0.3 + (variation_index * 0.15)
    temperature = min(temperature, 0.9)

    # Try AI providers
    edl = _try_plan_groq(prompt, cfg, temperature)
    if not edl:
        edl = _try_plan_gemini(prompt, cfg, temperature)
    if not edl:
        print("  ⚠ AI planning failed — using fallback sequential plan")
        edl = _fallback_plan(profiles, style)

    # Validate the EDL
    edl = _validate_edl(edl, profiles, style)

    print(f"\n  Edit plan: {len(edl)} segments")
    total_planned = sum(s["end"] - s["start"] for s in edl)
    print(f"  Planned duration: {total_planned:.1f}s")
    for i, seg in enumerate(edl, 1):
        dur = seg["end"] - seg["start"]
        print(f"    {i}. [{seg['source_file']}] "
              f"{seg['start']:.1f}s–{seg['end']:.1f}s ({dur:.1f}s) "
              f"→ {seg['transition']} | {seg['reason'][:50]}")

    return edl


def _try_plan_groq(prompt: str, cfg: Config, temperature: float = 0.3) -> list[dict]:
    """Try planning with Groq."""
    if cfg.GROQ_API_KEY in ("", "YOUR_GROQ_API_KEY_HERE"):
        return []

    import requests

    backoff = [2, 10, 25]
    for attempt in range(3):
        try:
            time.sleep(backoff[attempt])
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {cfg.GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": cfg.GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                },
                timeout=60,
            )
            if resp.status_code == 429:
                retry = resp.headers.get("Retry-After")
                wait = min(int(retry), 60) if retry else backoff[min(attempt, 2)]
                print(f"  ⏳ Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            return _parse_json(text)

        except Exception as e:
            print(f"  ⚠ Groq planner attempt {attempt+1}/3: {e}")
            traceback.print_exc()

    return []


def _try_plan_gemini(prompt: str, cfg: Config, temperature: float = 0.3) -> list[dict]:
    """Try planning with Gemini."""
    if cfg.GEMINI_API_KEY in ("", "YOUR_GEMINI_API_KEY_HERE"):
        return []

    try:
        import google.generativeai as genai
    except ImportError:
        return []

    genai.configure(api_key=cfg.GEMINI_API_KEY)
    model = genai.GenerativeModel(cfg.GEMINI_MODEL)

    backoff = [4, 15, 30]
    for attempt in range(3):
        try:
            time.sleep(backoff[attempt])
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature)
            )
            text = response.text.strip()
            return _parse_json(text)

        except Exception as e:
            print(f"  ⚠ Gemini planner attempt {attempt+1}/3: {e}")

    return []


def _fallback_plan(profiles: list[dict], style: dict) -> list[dict]:
    """Fallback: take the first N seconds from each clip sequentially."""
    edl = []
    seg_dur = (style["min_segment"] + style["max_segment"]) / 2

    for i, p in enumerate(profiles):
        end = min(seg_dur, p["duration"])
        edl.append({
            "source_file": p["file"],
            "start": 0.0,
            "end": round(end, 2),
            "reason": f"Auto-selected from {p['file']}",
            "transition": "none" if i == 0 else style["default_transition"],
        })

    return edl


def _validate_edl(edl: list[dict], profiles: list[dict], style: dict) -> list[dict]:
    """Validate and sanitize the AI's edit decision list."""
    # Build a lookup of valid files
    valid_files = {p["file"]: p for p in profiles}
    validated = []

    for seg in edl:
        try:
            source = seg.get("source_file", "")
            start = float(seg.get("start", 0))
            end = float(seg.get("end", 0))
            reason = str(seg.get("reason", ""))[:120]
            transition = str(seg.get("transition", "cut"))

            # Check source file exists
            if source not in valid_files:
                continue

            # Check bounds
            clip_dur = valid_files[source]["duration"]
            start = max(0.0, start)
            end = min(end, clip_dur)

            # Check minimum segment length
            if end - start < style["min_segment"] * 0.5:
                continue

            # Cap segment length
            if end - start > style["max_segment"] * 1.5:
                end = start + style["max_segment"]

            validated.append({
                "source_file": source,
                "start": round(start, 2),
                "end": round(end, 2),
                "reason": reason,
                "transition": transition if transition in (
                    "none", "cut", "crossfade", "zoom",
                    "whip_pan", "glitch", "fade_to_black"
                ) else style["default_transition"],
            })
        except (TypeError, ValueError, KeyError):
            continue

    # First segment should always have "none" transition
    if validated:
        validated[0]["transition"] = "none"

    return validated


def _parse_json(text: str) -> list:
    """Strip markdown fences and parse JSON from AI response."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.strip("`").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError(f"Could not parse JSON: {text[:300]}")
