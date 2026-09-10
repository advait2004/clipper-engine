"""
B-Roll Fetcher — download stock footage from Pexels for each scene
===================================================================
Queries the Pexels Video API for each scene's search query.
Prefers portrait orientation and HD quality.

Improvements over original:
- Per-scene fallback: single scene failure doesn't kill the job
- Fixed resolution selection: picks best HD file, not first match
- File caching: skips re-downloading if file already exists
"""

import os
import random
import requests
from config import Config


# Generic fallback queries when a scene's search yields nothing
_FALLBACK_QUERIES = [
    "abstract motion background",
    "nature landscape aerial",
    "city skyline timelapse",
    "ocean waves sunset",
    "particles bokeh",
    "clouds sky timelapse",
]


def fetch_broll_for_scenes(
    scenes: list[dict],
    output_dir: str,
    cfg: Config,
) -> list[str]:
    """
    Given a list of scenes with search queries, fetches a stock video
    for each scene from Pexels.

    Returns a list of local file paths to the downloaded videos.

    Args:
        scenes: List of {"text": "...", "search_query": "..."} dicts.
        output_dir: Directory to save downloaded videos.
        cfg: Config object (needs PEXELS_API_KEY).

    Returns:
        List of file paths, one per scene. Failed scenes get a fallback.
    """
    api_key = getattr(cfg, "PEXELS_API_KEY", "")
    if not api_key:
        raise ValueError(
            "PEXELS_API_KEY is not set in config. Cannot fetch stock footage.\n"
            "Get a free key at: https://www.pexels.com/api/"
        )

    print(f"  [B-Roll Fetcher] Fetching footage for {len(scenes)} scenes from Pexels...")
    os.makedirs(output_dir, exist_ok=True)

    headers = {"Authorization": api_key}
    downloaded_files = []

    for i, scene in enumerate(scenes):
        query = scene.get("search_query", "nature")
        file_path = os.path.join(output_dir, f"scene_{i}.mp4")

        # Skip if already downloaded (cache)
        if os.path.exists(file_path) and os.path.getsize(file_path) > 10_000:
            print(f"    [{i+1}/{len(scenes)}] Cached: {os.path.basename(file_path)}")
            downloaded_files.append(file_path)
            continue

        print(f"    [{i+1}/{len(scenes)}] Searching: '{query}'...")

        try:
            download_link = _find_best_video(query, headers)

            if not download_link:
                # Try a generic fallback query
                fallback_query = random.choice(_FALLBACK_QUERIES)
                print(f"    ⚠ No results for '{query}'. Trying fallback: '{fallback_query}'")
                download_link = _find_best_video(fallback_query, headers)

            if not download_link:
                raise Exception(f"No videos found on Pexels for '{query}' or fallback queries.")

            # Download the video
            print(f"    Downloading to {os.path.basename(file_path)}...")
            _download_file(download_link, file_path)
            downloaded_files.append(file_path)

        except Exception as e:
            print(f"    ❌ Failed to fetch video for scene {i+1}: {e}")
            # Don't crash — try to continue with remaining scenes
            # If we have ANY previously downloaded files, use the last one as a fallback
            if downloaded_files:
                print(f"    ⚠ Reusing previous scene's footage as fallback")
                downloaded_files.append(downloaded_files[-1])
            else:
                raise  # First scene failed — nothing to fall back to

    return downloaded_files


def _find_best_video(query: str, headers: dict) -> str | None:
    """
    Search Pexels for a video matching the query.
    Prefers portrait orientation, falls back to any orientation.

    Returns the best download URL or None.
    """
    # Try portrait first (ideal for 9:16 videos)
    url = f"https://api.pexels.com/videos/search?query={query}&orientation=portrait&per_page=5"
    videos = _pexels_search(url, headers)

    if not videos:
        # Fallback to any orientation
        url = f"https://api.pexels.com/videos/search?query={query}&per_page=5"
        videos = _pexels_search(url, headers)

    if not videos:
        return None

    # Pick a random video from results to keep things fresh
    video = random.choice(videos)
    return _get_best_file_link(video)


def _pexels_search(url: str, headers: dict) -> list[dict]:
    """Execute a Pexels API search and return video list."""
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json().get("videos", [])
    except Exception as e:
        print(f"    Pexels API error: {e}")
        return []


def _get_best_file_link(video: dict) -> str | None:
    """
    Pick the best quality video file from a Pexels video object.
    
    Sorts by resolution (width × height) descending and picks
    the best HD file (≥720p but ≤1080p preferred to avoid huge downloads).
    """
    video_files = video.get("video_files", [])
    if not video_files:
        return None

    # Sort by resolution descending
    video_files.sort(
        key=lambda x: x.get("width", 0) * x.get("height", 0),
        reverse=True,
    )

    # Prefer files around 1080p (avoid 4K which are huge)
    for vf in video_files:
        w = vf.get("width", 0)
        h = vf.get("height", 0)
        if 720 <= max(w, h) <= 1920:
            return vf.get("link")

    # If nothing in the 720-1920 range, take the highest resolution
    return video_files[0].get("link")


def _download_file(url: str, output_path: str):
    """Download a file from URL with streaming."""
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
