"""
Script Parser — break a script into visual scenes with search queries
======================================================================
Uses the configured AI provider to decompose a text script into scenes,
each with a Pexels-optimised stock-footage search query.

Provider fallback: preferred → Gemini → Groq → NVIDIA → Ollama
"""

import json
import re
import requests
from config import Config


_PROMPT_TEMPLATE = """
You are an expert video director and stock footage researcher.
I am going to give you a script for a short-form vertical video (TikTok/Shorts).
Your job is to break the script down into logical visual scenes (usually 1-2 sentences per scene).
For EACH scene, generate an optimized search query for Pexels Video API to find the perfect B-roll.

Guidelines for Pexels Search Queries:
1. Keep it simple and literal (e.g., "man working on laptop", "city skyline night", "coffee pouring").
2. Do not use abstract concepts (e.g., "financial freedom" is bad, "money falling" is good).
3. Use 2-4 keywords maximum.
4. Avoid overly specific actions that are hard to find.

Output exactly a JSON array of objects. Do not include markdown blocks like ```json.
Each object must have exactly two keys: "text" (the exact text of the script for this scene) and "search_query" (the Pexels query).

Script to parse:
{script_text}
"""


def parse_script_to_scenes(script_text: str, cfg: Config) -> list[dict]:
    """
    Uses AI to break a script into logical visual scenes and generates
    an optimized Pexels search query for each scene.

    Returns:
        A list of dictionaries: [{"text": "...", "search_query": "..."}, ...]
    """
    print("  [Script Parser] Analyzing script and generating visual scenes...")

    prompt = _PROMPT_TEMPLATE.format(script_text=script_text)

    # Try providers in fallback order
    providers_tried = set()
    preferred = getattr(cfg, "AI_PROVIDER", "nvidia")

    # 1. Try preferred provider first
    result = _try_provider(preferred, prompt, cfg)
    providers_tried.add(preferred)
    if result:
        return result

    # 2. Fallback chain
    for provider in ["gemini", "groq", "nvidia", "ollama"]:
        if provider in providers_tried:
            continue
        print(f"  [Script Parser] Trying fallback: {provider}")
        result = _try_provider(provider, prompt, cfg)
        providers_tried.add(provider)
        if result:
            return result

    # 3. All providers failed — use simple sentence splitting
    print("  [Script Parser] All AI providers failed. Using fallback sentence splitter.")
    return _fallback_split(script_text)


def _try_provider(provider: str, prompt: str, cfg: Config) -> list[dict] | None:
    """Try a single AI provider. Returns parsed scenes or None."""
    try:
        if provider == "nvidia":
            return _parse_nvidia(prompt, cfg)
        elif provider == "gemini":
            return _parse_gemini(prompt, cfg)
        elif provider == "groq":
            return _parse_groq(prompt, cfg)
        elif provider == "ollama":
            return _parse_ollama(prompt, cfg)
    except Exception as e:
        print(f"  [Script Parser] {provider} failed: {e}")
    return None


# ─── NVIDIA ──────────────────────────────────────────────────────────────────

def _parse_nvidia(prompt: str, cfg: Config) -> list[dict] | None:
    key = getattr(cfg, "NVIDIA_API_KEY", "")
    if not key or key == "YOUR_NVIDIA_API_KEY_HERE":
        return None

    resp = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": cfg.NVIDIA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1500,
        },
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    scenes = _parse_json(content)
    print(f"  [Script Parser] NVIDIA generated {len(scenes)} scenes.")
    return scenes


# ─── Gemini ──────────────────────────────────────────────────────────────────

def _parse_gemini(prompt: str, cfg: Config) -> list[dict] | None:
    key = getattr(cfg, "GEMINI_API_KEY", "")
    if not key or key == "YOUR_GEMINI_API_KEY_HERE":
        return None

    try:
        import google.generativeai as genai
    except ImportError:
        print("  [Script Parser] google-generativeai not installed")
        return None

    genai.configure(api_key=key)
    model = genai.GenerativeModel(cfg.GEMINI_MODEL)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=0.2),
    )
    content = response.text.strip()
    scenes = _parse_json(content)
    print(f"  [Script Parser] Gemini generated {len(scenes)} scenes.")
    return scenes


# ─── Groq ────────────────────────────────────────────────────────────────────

def _parse_groq(prompt: str, cfg: Config) -> list[dict] | None:
    key = getattr(cfg, "GROQ_API_KEY", "")
    if not key or key == "YOUR_GROQ_API_KEY_HERE":
        return None

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": cfg.GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    scenes = _parse_json(content)
    print(f"  [Script Parser] Groq generated {len(scenes)} scenes.")
    return scenes


# ─── Ollama (local) ──────────────────────────────────────────────────────────

def _parse_ollama(prompt: str, cfg: Config) -> list[dict] | None:
    url = getattr(cfg, "OLLAMA_URL", "http://localhost:11434/api/generate")
    model = getattr(cfg, "OLLAMA_MODEL", "llama3.1")

    resp = requests.post(
        url,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    content = resp.json().get("response", "").strip()
    scenes = _parse_json(content)
    print(f"  [Script Parser] Ollama generated {len(scenes)} scenes.")
    return scenes


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> list[dict]:
    """Strip markdown fences and parse JSON array from AI response."""
    # Remove markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list) and result:
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting array from surrounding text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list) and result:
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON scenes from AI response: {text[:300]}")


def _fallback_split(script_text: str) -> list[dict]:
    """
    Fallback: split script into sentences and use generic search queries.
    Used when all AI providers fail.
    """
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", script_text.strip())
    scenes = []

    # Simple keyword extraction for search queries
    generic_queries = [
        "nature landscape aerial", "city skyline timelapse",
        "person thinking", "technology abstract", "ocean waves",
        "sunrise mountains", "busy street people walking",
        "workspace desk laptop", "abstract particles motion",
    ]

    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
        scenes.append({
            "text": sentence,
            "search_query": generic_queries[i % len(generic_queries)],
        })

    print(f"  [Script Parser] Fallback split into {len(scenes)} sentences.")
    return scenes
