# 🎬 Automated Video Clipping Engine

Turns any YouTube video into viral-ready **9:16 clips with captions** — fully automated, completely free.

## How it works

```
YouTube URL
    ↓ yt-dlp
Download (best quality up to 1080p)
    ↓ faster-whisper
Transcribe (word-level timestamps)
    ↓ Gemini Flash / Ollama
AI scores top 5 viral moments
    ↓ FFmpeg
Cut each clip
    ↓ MediaPipe + FFmpeg
Reframe to 9:16 (face-aware crop)
    ↓ FFmpeg + ASS subtitles
Burn captions
    ↓
output/final/ ✅
```

---

## Setup

### Step 1 — Install FFmpeg (required)

**Mac**
```bash
brew install ffmpeg
```

**Linux**
```bash
sudo apt update && sudo apt install -y ffmpeg
```

**Windows**
Download from https://ffmpeg.org/download.html and add `ffmpeg.exe` to your PATH.

Verify: `ffmpeg -version`

---

### Step 2 — Install Python packages

```bash
pip install -r requirements.txt
```

Python 3.9+ required. First run downloads Whisper model weights (~150 MB).

---

### Step 3 — Set up your free AI provider

#### Option A: Gemini Flash (recommended — fastest)

1. Go to https://aistudio.google.com/apikey
2. Sign in with Google
3. Click **Create API key**
4. **DO NOT enable billing** — free tier vanishes permanently if billing is enabled
5. Open `config.py` and set:

```python
AI_PROVIDER    = "gemini"
GEMINI_API_KEY = "paste-your-key-here"
```

Free limits: **1,500 requests/day**, 15 requests/minute — enough for 300+ videos daily.

#### Option B: Ollama (fully local — zero internet after setup)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model (~4.7 GB one-time download)
ollama pull llama3.1
```

Then in `config.py`:

```python
AI_PROVIDER = "ollama"
```

No rate limits, fully private, works offline.

---

### Step 4 — Run

```bash
# Single video
python main.py https://youtu.be/VIDEO_ID

# Multiple videos
python main.py https://youtu.be/ID1 https://youtu.be/ID2

# From a file (one URL per line, # lines are ignored)
python main.py --file urls.txt

# Override AI provider
python main.py https://youtu.be/ID --ai ollama

# Override clips per video
python main.py https://youtu.be/ID --clips 3

# Interactive mode (no arguments)
python main.py
```

Output clips appear in `output/final/` as `.mp4` files.

---

## Posting workflow (Whop / Iman Gadzhi)

1. Engine saves finished clips to `output/final/`
2. Open TikTok / Reels / YouTube Shorts on your phone
3. Post a clip manually
4. **Immediately** open Whop and submit the post URL to the campaign
5. **Must submit within 1 hour** of posting — submissions after 1 hour are rejected
6. Repeat for each clip across all three platforms

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `AI_PROVIDER` | `"gemini"` | `"gemini"` or `"ollama"` |
| `GEMINI_MODEL` | `"gemini-2.0-flash"` | Gemini model (Flash = free) |
| `WHISPER_MODEL` | `"base"` | `tiny` / `base` / `small` / `medium` |
| `CLIPS_PER_VIDEO` | `5` | Max clips generated per video |
| `MIN_CLIP_SECONDS` | `30` | Minimum clip length |
| `MAX_CLIP_SECONDS` | `90` | Maximum clip length |
| `CAPTION_WORDS_PER_LINE` | `4` | Words shown per caption block |
| `CAPTION_FONT_SIZE` | `75` | Caption size (for 1080×1920) |
| `CAPTION_UPPERCASE` | `True` | ALL CAPS captions |
| `DELETE_RAW` | `True` | Auto-delete downloaded video |
| `DELETE_TEMP` | `True` | Auto-delete intermediate files |

---

## Processing time (no GPU, typical laptop)

| Stage | Time |
|---|---|
| Download (1-hour video) | 1–2 min |
| Transcribe (`base` model) | 3–6 min |
| AI scoring (Gemini) | 5–10 sec |
| AI scoring (Ollama) | 30–90 sec |
| Cut + Reframe + Caption (per clip) | 2–4 min |
| **Total per video (5 clips)** | **~20–35 min** |

For 100 clips/day (20 videos × 5 clips): run overnight or during the day in the background.

---

## Batch processing (urls.txt)

Create a file `urls.txt`:
```
# Dhruv Rathee episodes
https://youtu.be/AAAA
https://youtu.be/BBBB

# Ankur Warikoo
https://youtu.be/CCCC
```

Run:
```bash
python main.py --file urls.txt
```

---

## Output structure

```
output/
├── final/
│   ├── 20260705_143022_clip1_final.mp4   ← ready to post
│   ├── 20260705_143022_clip2_final.mp4
│   ├── 20260705_143022_clip3_final.mp4
│   ├── 20260705_143022_clip4_final.mp4
│   ├── 20260705_143022_clip5_final.mp4
│   └── 20260705_143022_metadata.json     ← timestamps + hooks
├── raw/    (auto-deleted after processing)
└── temp/   (auto-deleted after processing)
```

---

## Troubleshooting

**`ffmpeg: command not found`**
Install ffmpeg (Step 1 above).

**`ModuleNotFoundError: No module named 'faster_whisper'`**
Run `pip install -r requirements.txt`

**Gemini returns 429 (rate limit)**
You've hit 15 requests/minute. The engine adds a 4-second delay automatically. If it persists, set `AI_PROVIDER = "ollama"`.

**Captions misaligned**
Increase `WHISPER_MODEL` to `"small"` in `config.py` for better accuracy.

**Video is already portrait (9:16)**
Engine detects this and skips the reframe step automatically.

**Face not centred in crop**
The face detection samples 24 frames and uses the median position. For fast cuts or multi-person videos, you can increase `num_samples` in `reframer.py` line 47.
