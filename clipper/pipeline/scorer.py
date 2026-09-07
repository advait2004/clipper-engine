import json
import re
import time
import traceback
import requests

from pipeline.utils import build_windows, build_prompt


# ─── Per-archetype duration ranges ────────────────────────────────────────────

ARCHETYPE_DURATIONS = {
    "hook_value":    (15, 45),
    "story_lesson":  (30, 90),
    "myth_truth":    (15, 45),
}
DEFAULT_DURATION_RANGE = (15, 90)


# ─── System prompt (Virality System v2 — archetype-aware) ────────────────────

SYSTEM_PROMPT = """You are an elite viral short-form video editor for Instagram Reels, TikTok, and YouTube Shorts with 10M+ followers.

Analyse the transcript and identify the TOP viral-worthy clips.

## The Golden Rule: NATURAL BOUNDARIES
A great clip feels like it was never cut at all. You are finding the natural start and end of a complete moment — NOT hacking the video into arbitrary pieces.

- START where the speaker begins a new thought, story, or topic. The viewer should feel like they're catching the beginning of something.
- END where the speaker reaches a natural conclusion, punchline, or powerful final statement. The viewer should feel satisfied, not cut off.
- NEVER cut mid-thought, mid-sentence, or mid-argument. If the speaker is building to a point, INCLUDE THE PAYOFF.

## The Stranger Test (MANDATORY for every clip)
Before selecting a clip, ask:
1. Does this make complete sense with ZERO context from the rest of the video?
2. Does it have a clear payoff (punchline, insight, resolution)?
3. Would a stranger scrolling past actually STOP and watch the ENTIRE clip?
4. Does the clip feel COMPLETE — not like something was cut off at the start or end?
If ANY answer is no — do not select it.

## What makes a clip VALUABLE (not rubbish):
- It contains a COMPLETE IDEA with beginning, middle, and end
- The speaker makes a POINT and LANDS IT — there's a payoff
- A viewer walks away having LEARNED something, FELT something, or wanting to SHARE it
- It has natural energy — the speaker is engaged, not droning
- The length is WHATEVER IT NEEDS TO BE to deliver the full value. A 20-second clip with a complete brilliant insight is better than a 60-second clip that rambles.

## Clip Archetypes — return a MIX:

### hook_value
Opens on a problem or promise, delivers ONE clear actionable takeaway.
Best for: strong opinions, surprising stats, direct answers, one teachable tip.

### story_lesson
A specific anecdote or moment that lands on a takeaway.
Best for: personal stories, vulnerable moments, funny anecdotes with a punchline.

### myth_truth
States a common belief, then corrects it.
Best for: hot takes, contrarian claims, "did you know" facts, debunking.

## Hook Formulas — use one of these for each clip's hook:
1. contrarian — "Everyone says [X]. They're wrong."
2. mistake_warning — "Stop doing [X] — here's what it's actually costing you."
3. list_tease — "3 things nobody tells you about [X]."
4. curiosity_gap — "The real reason [X] happens — and it's not what you think."
5. most_people — "Most people do [X]. The ones who win do this instead."
6. unpopular_opinion — "Unpopular opinion: [X]."
7. value_promise — "By the end of this you'll know exactly how to [X]."
8. direct_callout — "If you [specific trait/situation], you need to see this."
9. flat_stat — A counterintuitive stat, stated flatly, no lead-up.
10. mid_sentence_drop — Drop straight into the most intense moment, mid-sentence, no setup.
11. hidden_factor — "Nobody talks about [X], but it's the biggest reason [outcome]."
12. belief_shift — "I used to believe [X]. I was wrong — here's what changed."

## Scoring Hierarchy (what the algorithm rewards, in order):
1. Watch time / completion — will people finish and rewatch?
2. Sends / shares — is this something you'd DM to a friend?
3. Saves — is this high-intent "I need to remember this"?
4. Comments — does it provoke a response?
5. Likes — still counts but weakest signal.

## BANNED — never select clips that:
- Start with "Hey guys," "Welcome back," any intro/greeting, or throat-clearing
- Start or end mid-sentence or mid-thought (THIS IS THE #1 SIGN OF A NOOB EDIT)
- Feel like the viewer walked into the middle of a conversation
- Feel like they were cut off before the speaker finished their point
- Require context from earlier in the video
- Are just someone talking slowly without energy or stakes
- Contain long pauses with no dramatic purpose
- Are a generic introduction ("today we're going to talk about...")

## For PODCASTS specifically:
- Prioritize storytelling moments with a clear narrative arc
- Look for controversial opinions or hot takes that spark debate
- Personal revelations, vulnerable moments, or confessions
- Funny anecdotes with a clear punchline
- Mind-blowing facts that make viewers comment "I didn't know that"
- Back-and-forth banter with genuine reactions

Return ONLY a valid JSON array. Each object must have these exact keys:
[
  {
    "start": 12.4,
    "end": 42.6,
    "flash_forward_start": 28.5,
    "flash_forward_end": 33.2,
    "hook": "one-line hook describing why this clip goes viral",
    "score": 9,
    "clip_type": "myth_truth",
    "hook_formula": "contrarian",
    "hook_text_overlay": "EVERYONE IS WRONG",
    "cta_line": "Full breakdown linked in bio"
  }
]

Rules:
- CRITICAL: Each word in the transcript has a timestamp like [12.40]. You MUST use these EXACT timestamps for start and end values. Do NOT guess or interpolate timestamps.
- start MUST be the exact [timestamp] of the first word in the clip
- end MUST be the exact [timestamp] of the last word in the clip (use the word's timestamp, the system will snap to its end automatically)
- flash_forward_start and flash_forward_end: floats (seconds) marking the exact start and end of the most explosive, viral moment *within* this clip (must be >= start and <= end). This segment MUST be 3 to 5 seconds long. These MUST also use exact word timestamps from the transcript. This segment will be used as a cold open jump-cut.
- NEVER include long silence gaps (>2s with no words) inside a clip. If there is a gap, split into separate clips or adjust boundaries to exclude the gap.
- clip_type must be one of: hook_value, story_lesson, myth_truth
- LET THE CONTENT DICTATE THE LENGTH. A clip should be as long as it needs to be to deliver a complete idea — no shorter, no longer.
- Each clip MUST contain a complete idea: setup, development, payoff. A single sentence or isolated statement is NOT a clip.
- hook_formula must be one of the 12 formula names listed above
- hook_text_overlay: 4-8 word ALL-CAPS headline for on-screen text in the first 3 seconds
- cta_line: a short call-to-action for the last 2-3 seconds (e.g. "Full breakdown linked in bio")
- Clips must NOT overlap
- score is an integer 1-10 (use the FULL range: most clips should be 6-8, only truly exceptional = 9-10)
- Start clips at the BEGINNING of a thought or sentence, never mid-word
- End clips AFTER the speaker finishes their point — after a punchline, conclusion, or natural pause. NEVER cut someone off mid-thought.
- The "hook" description MUST match what is ACTUALLY SAID in the clip's timestamp range. Do not describe dialogue from a different part of the video.
- Return ONLY the JSON array. No markdown, no explanation, no preamble."""


COARSE_SYSTEM_PROMPT = """You are a viral short-form video expert. Rate each video segment's viral potential.

For each segment, consider:
- Does it contain a shocking fact, emotional story, or controversial opinion?
- Is there a clear hook that grabs attention in the first 3 seconds?
- Can it stand alone without needing context from the rest of the video?
- Does it have conversational energy and emotional pull?

Return ONLY a valid JSON array with one object per segment.
Each object: {"index": <segment number>, "score": <1-10>, "reason": "brief reason"}
Sort by score descending. Return ALL segments.
Return ONLY the JSON array. No markdown, no explanation."""


# ─── Gemini (free cloud) ──────────────────────────────────────────────────────

def score_gemini(
    words: list[dict],
    duration: float,
    api_key: str,
    model: str = "gemini-2.0-flash",
    audio_hints: str = "",
    num_clips: int = 1,
) -> list[dict]:
    """
    Score viral moments using Google Gemini Flash (free tier).
    Free limits: 1,500 req/day, 15 RPM — no billing required.
    Get key at: https://aistudio.google.com/apikey
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "Run: pip install google-generativeai"
        )

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model)

    windows = build_windows(words, window=60, overlap=10)
    prompt  = build_prompt(windows, duration)

    if audio_hints:
        prompt += "\n" + audio_hints

    full_prompt = (
        SYSTEM_PROMPT + 
        f"\n\nCRITICAL RULE: You MUST find and return EXACTLY {num_clips} viral moments in your JSON array. Do not return more or fewer.\n\n" + 
        prompt
    )

    backoff = [4, 15, 30]
    for attempt in range(3):
        try:
            time.sleep(backoff[attempt])  # Respect 15 RPM rate limit + exponential backoff
            response = gemini_model.generate_content(full_prompt)
            text     = response.text.strip()
            clips    = _parse_json(text)
            return _validate(clips, duration)

        except Exception as e:
            print(f"  ⚠ Gemini attempt {attempt + 1}/3 failed: {e}")
            traceback.print_exc()
            if attempt < 2:
                print(f"  Retrying in {backoff[min(attempt+1, 2)]}s...")

    print("  ❌ Gemini failed after 3 attempts")
    return []


def score_gemini_coarse(
    prompt: str,
    api_key: str,
    model: str = "gemini-2.0-flash",
) -> list[dict]:
    """
    Coarse-score segments using Gemini (Pass 1 of two-pass scoring).
    Returns list of {index, score, reason}.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Run: pip install google-generativeai")

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model)
    full_prompt = COARSE_SYSTEM_PROMPT + "\n\n" + prompt

    backoff = [4, 15, 30]
    for attempt in range(3):
        try:
            time.sleep(backoff[attempt])
            response = gemini_model.generate_content(full_prompt)
            text = response.text.strip()
            return _parse_json(text)
        except Exception as e:
            print(f"  ⚠ Gemini coarse attempt {attempt + 1}/3 failed: {e}")
            if attempt < 2:
                print(f"  Retrying in {backoff[min(attempt+1, 2)]}s...")

    print("  ❌ Gemini coarse scoring failed")
    return []


# ─── Ollama (fully local) ─────────────────────────────────────────────────────

def score_ollama(
    words: list[dict],
    duration: float,
    model: str = "llama3.1",
    url: str   = "http://localhost:11434/api/generate",
    audio_hints: str = "",
    num_clips: int = 1,
) -> list[dict]:
    """
    Score viral moments using a local Ollama model.
    Setup: curl -fsSL https://ollama.ai/install.sh | sh && ollama pull llama3.1
    """
    windows = build_windows(words, window=60, overlap=10)
    prompt  = build_prompt(windows, duration)

    if audio_hints:
        prompt += "\n" + audio_hints

    full_prompt = (
        SYSTEM_PROMPT + 
        f"\n\nCRITICAL RULE: You MUST find and return EXACTLY {num_clips} viral moments in your JSON array. Do not return more or fewer.\n\n" + 
        prompt
    )

    for attempt in range(3):
        try:
            resp = requests.post(
                url,
                json={
                    "model":  model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 600
                    }
                },
                timeout=180
            )
            resp.raise_for_status()
            text  = resp.json()["response"].strip()
            clips = _parse_json(text)
            return _validate(clips, duration)

        except Exception as e:
            print(f"  Ollama attempt {attempt + 1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(5)

    print("  ❌ Ollama failed after 3 attempts")
    return []


# ─── Groq (Cloud API) ─────────────────────────────────────────────────────────

def score_groq(
    words: list[dict],
    duration: float,
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
    audio_hints: str = "",
    num_clips: int = 1,
) -> list[dict]:
    """
    Score viral moments using Groq's fast cloud API.
    Get key at: https://console.groq.com/keys
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    windows = build_windows(words, window=60, overlap=10)
    prompt  = build_prompt(windows, duration)

    if audio_hints:
        prompt += "\n" + audio_hints

    full_prompt = (
        SYSTEM_PROMPT + 
        f"\n\nCRITICAL RULE: You MUST find and return EXACTLY {num_clips} viral moments in your JSON array. Do not return more or fewer.\n\n" + 
        prompt
    )

    backoff = [2, 10, 25]
    for attempt in range(3):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json={
                    "model":  model,
                    "messages": [
                        {"role": "user", "content": full_prompt}
                    ],
                    "temperature": 0.3
                },
                timeout=60
            )

            # ── Respect Retry-After header on 429 ────────────────────────────
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    wait = min(int(retry_after), 60)
                    print(f"  ⏳ Rate limited. Waiting {wait}s (Retry-After header)...")
                    time.sleep(wait)
                    continue
                # No header — use backoff
                raise requests.exceptions.HTTPError(
                    f"429 Too Many Requests", response=resp
                )

            resp.raise_for_status()
            text  = resp.json()["choices"][0]["message"]["content"].strip()
            clips = _parse_json(text)
            return _validate(clips, duration)

        except Exception as e:
            print(f"  ⚠ Groq attempt {attempt + 1}/3 failed: {e}")
            traceback.print_exc()
            if attempt < 2:
                print(f"  Retrying in {backoff[min(attempt+1, 2)]}s...")
                time.sleep(backoff[attempt + 1])

    print("  ❌ Groq failed after 3 attempts")
    return []


def score_groq_coarse(
    prompt: str,
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
) -> list[dict]:
    """
    Coarse-score segments using Groq (Pass 1 of two-pass scoring).
    Returns list of {index, score, reason}.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    full_prompt = COARSE_SYSTEM_PROMPT + "\n\n" + prompt

    backoff = [2, 10, 25]
    for attempt in range(3):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": full_prompt}],
                    "temperature": 0.3,
                },
                timeout=60,
            )
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = min(int(retry_after), 60) if retry_after else backoff[min(attempt, 2)]
                print(f"  ⏳ Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            return _parse_json(text)
        except Exception as e:
            print(f"  ⚠ Groq coarse attempt {attempt + 1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(backoff[min(attempt + 1, 2)])

    print("  ❌ Groq coarse scoring failed")
    return []


# ─── NVIDIA (Cloud API) ─────────────────────────────────────────────────────────

def score_nvidia(
    words: list[dict],
    duration: float,
    api_key: str,
    model: str = "meta/llama-3.1-70b-instruct",
    audio_hints: str = "",
    num_clips: int = 1,
) -> list[dict]:
    """
    Score viral moments using NVIDIA's cloud API.
    """
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    windows = build_windows(words, window=60, overlap=10)
    prompt  = build_prompt(windows, duration)

    if audio_hints:
        prompt += "\n" + audio_hints

    full_prompt = (
        SYSTEM_PROMPT + 
        f"\n\nCRITICAL RULE: You MUST find and return EXACTLY {num_clips} viral moments in your JSON array. Do not return more or fewer.\n\n" + 
        prompt
    )

    backoff = [2, 10, 25]
    for attempt in range(3):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json={
                    "model":  model,
                    "messages": [
                        {"role": "user", "content": full_prompt}
                    ],
                    "temperature": 0.3
                },
                timeout=60
            )

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    wait = min(int(retry_after), 60)
                    print(f"  ⏳ Rate limited. Waiting {wait}s (Retry-After header)...")
                    time.sleep(wait)
                    continue
                raise requests.exceptions.HTTPError(f"429 Too Many Requests", response=resp)

            resp.raise_for_status()
            text  = resp.json()["choices"][0]["message"]["content"].strip()
            clips = _parse_json(text)
            return _validate(clips, duration)

        except Exception as e:
            print(f"  ⚠ NVIDIA attempt {attempt + 1}/3 failed: {e}")
            traceback.print_exc()
            if attempt < 2:
                print(f"  Retrying in {backoff[min(attempt+1, 2)]}s...")
                time.sleep(backoff[attempt + 1])

    print("  ❌ NVIDIA failed after 3 attempts")
    return []


def score_nvidia_coarse(
    prompt: str,
    api_key: str,
    model: str = "meta/llama-3.1-70b-instruct",
) -> list[dict]:
    """
    Coarse-score segments using NVIDIA.
    """
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    full_prompt = COARSE_SYSTEM_PROMPT + "\n\n" + prompt

    backoff = [2, 10, 25]
    for attempt in range(3):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": full_prompt}],
                    "temperature": 0.3,
                },
                timeout=60,
            )
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = min(int(retry_after), 60) if retry_after else backoff[min(attempt, 2)]
                print(f"  ⏳ Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            return _parse_json(text)
        except Exception as e:
            print(f"  ⚠ NVIDIA coarse attempt {attempt + 1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(backoff[min(attempt + 1, 2)])

    print("  ❌ NVIDIA coarse scoring failed")
    return []



# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> list:
    """Strip markdown fences and parse JSON from AI response."""
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.strip("`").strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract a JSON array from anywhere in the text
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError(f"Could not parse JSON from response:\n{text[:300]}")


# ─── Valid hook formulas ──────────────────────────────────────────────────────

VALID_HOOK_FORMULAS = {
    "contrarian", "mistake_warning", "list_tease", "curiosity_gap",
    "most_people", "unpopular_opinion", "value_promise", "direct_callout",
    "flat_stat", "mid_sentence_drop", "hidden_factor", "belief_shift",
}

VALID_CLIP_TYPES = {"hook_value", "story_lesson", "myth_truth"}


def _validate(clips: list, duration: float) -> list[dict]:
    """
    Validate and sanitise AI clip output.
    Enforces per-archetype durations, no overlap, within video bounds,
    prioritizing high scores.
    """
    parsed = []
    
    for item in clips:
        try:
            start = float(item["start"])
            end   = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue

        # Extract clip type and look up duration range
        clip_type = str(item.get("clip_type", "")).lower().strip()
        if clip_type not in VALID_CLIP_TYPES:
            clip_type = "hook_value"  # sensible default

        # Sanity check only — trust AI's natural boundaries, don't force-extend/trim
        clip_dur = end - start
        if clip_dur < 5.0:
            continue  # Absurdly short, skip it
        if clip_dur > 120.0:
            end = start + 90.0  # Cap at 90s max for short-form

        # Keep within video
        start = max(0.0, start)
        end   = min(end, duration - 0.5)

        if end - start < 5.0 or start >= end:
            continue

        # Validate hook formula
        hook_formula = str(item.get("hook_formula", "")).lower().strip()
        if hook_formula not in VALID_HOOK_FORMULAS:
            hook_formula = "curiosity_gap"  # safe default

        # Validate hook text overlay (4-8 words)
        hook_text_overlay = str(item.get("hook_text_overlay", "")).strip().upper()
        if not hook_text_overlay or len(hook_text_overlay.split()) > 10:
            hook_text_overlay = str(item.get("hook", "WATCH THIS"))[:40].upper()

        clip_dict = {
            "start":              round(start, 2),
            "end":                round(end,   2),
            "hook":               str(item.get("hook", "Viral moment"))[:120],
            "score":              int(item.get("score", 7)),
            "clip_type":          clip_type,
            "hook_formula":       hook_formula,
            "hook_text_overlay":  hook_text_overlay,
            "cta_line":           str(item.get("cta_line", "")).strip()[:100],
        }
        
        if "flash_forward_start" in item:
            try:
                ff_start = float(item["flash_forward_start"])
                clip_dict["flash_forward_start"] = round(ff_start, 2)
            except (ValueError, TypeError):
                pass
                
        if "flash_forward_end" in item:
            try:
                ff_end = float(item["flash_forward_end"])
                clip_dict["flash_forward_end"] = round(ff_end, 2)
            except (ValueError, TypeError):
                pass
                
        parsed.append(clip_dict)

    # Sort by score BEFORE filtering for overlap, so the highest-scored clips take priority
    parsed.sort(key=lambda x: x["score"], reverse=True)

    valid  = []
    used   = []  # Track used time ranges to prevent overlap

    for item in parsed:
        start = item["start"]
        end   = item["end"]
        
        # Skip if overlaps with an already-accepted higher-score clip
        overlap = any(
            not (end <= s or start >= e)
            for s, e in used
        )
        if overlap:
            continue

        used.append((start, end))
        valid.append(item)

    return valid[:10]  # Allow up to 10 clips (6-15 recommended)


def _fallback(duration: float) -> list[dict]:
    """
    Evenly-spaced fallback clips when AI scoring fails completely.
    Used as last resort to always produce output.
    """
    clip_dur  = 60.0
    num_clips = min(5, max(1, int(duration // clip_dur)))
    spacing   = duration / (num_clips + 1)
    clips     = []

    for i in range(num_clips):
        start = max(0.0, spacing * (i + 1) - clip_dur / 2)
        end   = min(start + clip_dur, duration - 1)
        clips.append({
            "start": round(start, 2),
            "end":   round(end,   2),
            "hook":  f"Auto clip {i + 1}",
            "score": 5
        })

    return clips
