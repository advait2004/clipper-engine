import os
import random
import requests
from clipper.config import Config

def fetch_broll_for_scenes(scenes: list[dict], output_dir: str, cfg: Config) -> list[str]:
    """
    Given a list of scenes with search queries, fetches a stock video for each scene from Pexels.
    Returns a list of local file paths to the downloaded videos.
    """
    if not cfg.PEXELS_API_KEY:
        raise ValueError("PEXELS_API_KEY is not set in config. Cannot fetch stock footage.")

    print(f"  [B-Roll Fetcher] Fetching footage for {len(scenes)} scenes from Pexels...")
    
    headers = {
        "Authorization": cfg.PEXELS_API_KEY
    }
    
    downloaded_files = []
    
    for i, scene in enumerate(scenes):
        query = scene.get("search_query", "nature")
        print(f"    Searching: '{query}'...")
        
        # We request portrait orientation directly to save cropping time if possible,
        # but fallback to landscape if not enough results.
        url = f"https://api.pexels.com/videos/search?query={query}&orientation=portrait&per_page=5"
        
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            videos = data.get("videos", [])
            if not videos:
                print(f"    ⚠ No portrait videos found for '{query}'. Trying any orientation...")
                url = f"https://api.pexels.com/videos/search?query={query}&per_page=5"
                resp = requests.get(url, headers=headers, timeout=30)
                videos = resp.json().get("videos", [])
                
            if not videos:
                print(f"    ⚠ No videos found at all for '{query}'. Using fallback.")
                url = f"https://api.pexels.com/videos/search?query=abstract&orientation=portrait&per_page=5"
                resp = requests.get(url, headers=headers, timeout=30)
                videos = resp.json().get("videos", [])
                
            if not videos:
                raise Exception("Could not find any fallback videos on Pexels.")
                
            # Pick a random video from the top 5 to keep things fresh
            video = random.choice(videos)
            
            # Find highest quality HD link (e.g. 1080x1920 or 1920x1080)
            video_files = video.get("video_files", [])
            # Sort by resolution (width * height)
            video_files.sort(key=lambda x: x.get("width", 0) * x.get("height", 0), reverse=True)
            
            # Prefer 1080p, but take best available
            best_file = video_files[0]
            for vf in video_files:
                if vf.get("width", 0) >= 1080 or vf.get("height", 0) >= 1080:
                    best_file = vf
                    
            download_link = best_file["link"]
            
            # Download it
            file_path = os.path.join(output_dir, f"scene_{i}.mp4")
            print(f"    Downloading to {os.path.basename(file_path)}...")
            
            with requests.get(download_link, stream=True) as r:
                r.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
            downloaded_files.append(file_path)
            
        except Exception as e:
            print(f"    ❌ Failed to fetch video for scene {i}: {e}")
            raise
            
    return downloaded_files
