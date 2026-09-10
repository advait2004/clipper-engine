import json
import requests
from clipper.config import Config

def parse_script_to_scenes(script_text: str, cfg: Config) -> list[dict]:
    """
    Uses the NVIDIA Nemotron API to break a script into logical visual scenes 
    and generates an optimized Pexels search query for each scene.
    
    Returns:
        A list of dictionaries: [{"text": "...", "search_query": "..."}, ...]
    """
    print("  [Script Parser] Analyzing script and generating visual scenes...")

    prompt = f"""
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
    headers = {
        "Authorization": f"Bearer {cfg.NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg.NVIDIA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1500,
    }

    try:
        resp = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        
        # Clean up possible markdown wrappers
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        scenes = json.loads(content)
        print(f"  [Script Parser] Generated {len(scenes)} scenes.")
        return scenes
        
    except Exception as e:
        print(f"  [Script Parser] Error: {e}")
        # Fallback to single scene
        return [{"text": script_text, "search_query": "nature landscape"}]
