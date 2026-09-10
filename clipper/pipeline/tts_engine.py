import os
import re
import subprocess
from clipper.config import Config

def parse_vtt_time(time_str: str) -> float:
    """Converts VTT timestamp (HH:MM:SS.mmm) to seconds."""
    parts = time_str.split(':')
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

def parse_vtt_to_words(vtt_path: str) -> list[dict]:
    """
    Parses an edge-tts VTT file and returns word-level timestamps.
    Returns: [{"word": "hello", "start": 0.1, "end": 0.5}, ...]
    """
    words = []
    with open(vtt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_start = 0.0
    current_end = 0.0
    
    for line in lines:
        line = line.strip()
        if not line or line == "WEBVTT":
            continue
            
        # Match timestamp line: "00:00:00.100 --> 00:00:00.500"
        time_match = re.match(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s-->\s(\d{2}:\d{2}:\d{2}\.\d{3})", line)
        if time_match:
            current_start = parse_vtt_time(time_match.group(1))
            current_end = parse_vtt_time(time_match.group(2))
            continue
            
        # If it's a text line and not a number block
        if not line.isdigit() and "-->" not in line:
            # edge-tts generates line-level VTT by default, but it's very granular.
            # To simulate word-level for the captioner, we split the block.
            # (If it's already word-level, this just returns the single word).
            text_words = line.split()
            if not text_words:
                continue
                
            duration = current_end - current_start
            word_dur = duration / len(text_words)
            
            for i, w in enumerate(text_words):
                w_clean = w.strip()
                if w_clean:
                    words.append({
                        "word": w_clean,
                        "start": current_start + (i * word_dur),
                        "end": current_start + ((i + 1) * word_dur)
                    })
                    
    return words

def generate_tts(text: str, output_audio: str, output_vtt: str, cfg: Config) -> list[dict]:
    """
    Generates TTS audio and VTT subtitles using edge-tts.
    Returns the parsed word-level timestamps.
    """
    print(f"  [TTS Engine] Generating voiceover (Voice: {cfg.TTS_VOICE})...")
    
    cmd = [
        "edge-tts",
        "--voice", cfg.TTS_VOICE,
        "--text", text,
        "--write-media", output_audio,
        "--write-subtitles", output_vtt
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"  [TTS Engine] Error running edge-tts: {e.stderr.decode('utf-8')}")
        raise
        
    if not os.path.exists(output_vtt):
        raise FileNotFoundError("edge-tts failed to generate VTT file.")
        
    words = parse_vtt_to_words(output_vtt)
    print(f"  [TTS Engine] Generated {len(words)} words of audio.")
    return words
