import os
from pathlib import Path


class Config:
    # ─── Directories ──────────────────────────────────────────────────────────
    BASE_DIR   = Path(__file__).parent
    OUTPUT_DIR = str(BASE_DIR / "output" / "final")
    TEMP_DIR   = str(BASE_DIR / "output" / "temp")
    RAW_DIR    = str(BASE_DIR / "output" / "raw")

    # ─── YouTube Download Settings ────────────────────────────────────────────
    YOUTUBE_COOKIES_BROWSER = "chrome"  # e.g., "chrome", "edge", "firefox"

    # ─── Performance Settings ─────────────────────────────────────────────────
    USE_GPU = True  # Enabled: Hardware acceleration via NVENC active
    MOMENT_SCORER = True  # Multi-signal flash forward: fuses text + audio + video to pick best cold-open

    # ─── AI Provider ──────────────────────────────────────────────────────────
    # "gemini"  → free cloud (1,500 req/day, no credit card needed)
    # "ollama"  → fully local, zero cost, needs ollama installed
    # "groq"    → very fast cloud (generous free tier)
    # "nvidia"  → high performance cloud API (NVIDIA NIM)
    AI_PROVIDER = "nvidia"

    # ─── Gemini (free tier) ───────────────────────────────────────────────────
    # 1. Go to https://aistudio.google.com/apikey
    # 2. Create API key (no credit card, no billing)
    # 3. NEVER enable billing on this project — free tier vanishes if you do
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL   = "gemini-2.0-flash"   # Free-tier Flash model

    # ─── Ollama (local fallback) ──────────────────────────────────────────────
    # Setup: curl -fsSL https://ollama.ai/install.sh | sh && ollama pull llama3.1
    OLLAMA_MODEL = "llama3.1"
    OLLAMA_URL   = "http://localhost:11434/api/generate"

    # ─── Groq (Cloud API) ─────────────────────────────────────────────────────
    # 1. Go to https://console.groq.com/keys
    # 2. Create API key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL   = "llama-3.3-70b-versatile"

    # ─── NVIDIA (Cloud API) ───────────────────────────────────────────────────
    # 1. Go to https://build.nvidia.com/
    # 2. Create API key
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
    NVIDIA_MODEL   = "nvidia/nemotron-3-super-120b-a12b"


    # ─── Whisper transcription ────────────────────────────────────────────────
    # "tiny"   = fastest, lower accuracy
    # "base"   = good balance (recommended for CPU)
    # "small"  = better accuracy, ~2x slower
    # "medium" = best accuracy, ~4x slower
    # "large-v3" = Ultimate accuracy (requires GPU)
    WHISPER_MODEL = "large-v3"

    # ─── Clip settings ────────────────────────────────────────────────────────
    CLIPS_PER_VIDEO  = 10    # Max clips to generate per video
    MIN_CLIP_SECONDS = 15   # Automatically extend AI clips to at least 15s
    MAX_CLIP_SECONDS = 90   # Maximum clip duration

    # ─── Scoring improvements ─────────────────────────────────────────────────
    # Delay (seconds) between AI API calls to avoid rate limits (429 errors)
    SCORER_DELAY_BETWEEN_CHUNKS = 2

    # Two-pass scoring: coarse scan all chunks first, then deep-score top ones
    # Reduces API calls and focuses AI attention on the best content
    TWO_PASS_SCORING = True
    TWO_PASS_TOP_CHUNKS = 8   # How many top chunks to deep-score in pass 2

    # Audio energy analysis: detect volume spikes, laughter, dramatic pauses
    # Provides local (free) hints to the AI before scoring
    AUDIO_ENERGY_HINTS = True

    # Smart clip boundaries: snap start/end to natural speech pauses
    # Prevents clips from starting/ending mid-sentence
    SMART_CLIP_BOUNDARIES = True

    # ─── Caption style ────────────────────────────────────────────────────────
    CAPTION_WORDS_PER_LINE = 4      # Words per caption block
    CAPTION_FONT_SIZE      = 75     # Font size (for 1080x1920)
    CAPTION_UPPERCASE      = True   # Convert captions to uppercase

    # ─── Sound effects ────────────────────────────────────────────────────────
    SOUND_EFFECTS_ENABLED = False  # Set False to skip (faster processing)

    # Volume levels (0.0 = silent, 1.0 = full)
    # Tune these if speech gets drowned out
    SFX_BG_VOLUME     = 0.08   # Ambient background music  (very subtle)
    SFX_WHOOSH_VOLUME = 0.60   # Intro whoosh at clip start
    SFX_IMPACT_VOLUME = 0.50   # Impact boom at hook keywords

    # ─── Advanced Reframing ──────────────────────────────────────────────────
    FACE_MESH_ENABLED = True         # Use Face Mesh (478 landmarks) for deep analysis
    SPEAKER_TRACKING = True          # Track active speaker via lip movement
    SPEAKER_DIARIZATION = False      # Audio speaker diarization (disabled due to HF model download hangs)
    CINEMATIC_PANNING = True         # Smooth ease-in-out camera pans
    DYNAMIC_SPLIT_SCREEN = False     # Disabled due to false positives
    INTERVIEW_LAYOUT_MODE = "snap_split" # Instantly snap to split screen on simultaneous speech
    SCENE_CLASSIFICATION = True      # Auto-detect scene type for strategy selection
    BROLL_KEN_BURNS = True           # Slow zoom/pan on faceless segments
    DEPTH_ESTIMATION = False         # MiDaS depth for foreground isolation (heavy)

    # Virtual camera tuning
    PAN_HOLD_DELAY = 0.3             # Seconds to hold before panning to new speaker
    PAN_MAX_VELOCITY = 0.30          # Max pan speed (fraction of frame width/sec)
    PAN_MIN_DURATION = 0.4           # Minimum pan duration (seconds)
    PAN_MAX_DURATION = 1.5           # Maximum pan duration (seconds)
    SPLIT_ACTIVE_RATIO = 0.60       # Active speaker's share in dynamic split (0.5–0.7)

    # Speaker diarization (optional)
    HF_TOKEN = os.getenv("HF_TOKEN", "")  # HuggingFace token for Pyannote model

    # ─── Caption style ────────────────────────────────────────────────────────
    # Available: "karaoke", "minimal", "bold_pop", "subtitle_bar", "typewriter"
    CAPTION_STYLE          = "karaoke"
    CAPTION_WORDS_PER_LINE = 4      # Words per caption block (overrides style default if set)
    CAPTION_FONT_SIZE      = 75     # Font size (overrides style default if set)
    CAPTION_UPPERCASE      = True   # Convert captions to uppercase

    # ─── Smart Edit (AI Auto-Editor) ──────────────────────────────────────────
    SMART_EDIT_DEFAULT_EDIT_STYLE    = "podcast"    # Default edit preset
    SMART_EDIT_DEFAULT_CAPTION_STYLE = "karaoke"    # Default caption style
    SMART_EDIT_MAX_SOURCE_CLIPS      = 20           # Max clips to analyze
    SMART_EDIT_CROSSFADE_DURATION    = 0.5          # Default crossfade (seconds)

    # ─── Virality System ──────────────────────────────────────────────────────
    VIRALITY_MODE            = True     # Enable archetype-aware clip discovery
    HOOK_TEXT_OVERLAY         = True     # Burn 4-8 word hook headline at 0-3s
    CTA_OVERLAY              = True     # Burn CTA text in last 2-3s
    CTA_DEFAULT_TEXT         = ""       # If set, overrides AI-generated CTA with this fixed text
    FILLER_WORD_REMOVAL      = True     # Strip fillers from captions
    INTRO_ZOOM               = True     # 10% zoom on first 0.5s of each clip
    FLASH_FORWARD_HOOK       = True     # Cut 3s internal highlight and paste at the start

    # ─── Script-to-Video Mode ─────────────────────────────────────────────────
    PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
    TTS_VOICE      = "en-US-ChristopherNeural"  # Default edge-tts voice
    TTS_WORD_LEVEL = True   # Re-transcribe TTS audio with Whisper for accurate karaoke timestamps

    # ─── Cleanup ───────────────────────────────────────────────────────────
    DELETE_RAW  = True    # Delete raw downloaded video after processing
    DELETE_TEMP = True    # Delete intermediate temp files

    def __init__(self):
        for d in [self.OUTPUT_DIR, self.TEMP_DIR, self.RAW_DIR]:
            os.makedirs(d, exist_ok=True)
