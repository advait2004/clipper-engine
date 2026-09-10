import os
import subprocess
from clipper.pipeline.utils import get_video_duration

def build_faceless_video(broll_files: list[str], tts_audio_path: str, output_path: str):
    """
    Stitches multiple b-roll videos together, loops them if necessary, 
    and muxes them with the TTS audio track.
    """
    print("  [Video Builder] Stitching B-roll and syncing with TTS audio...")
    
    # Get total audio duration (using ffprobe)
    cmd_probe = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", tts_audio_path
    ]
    try:
        audio_dur_str = subprocess.check_output(cmd_probe).decode().strip()
        audio_dur = float(audio_dur_str)
    except Exception as e:
        print(f"  [Video Builder] Error getting audio duration: {e}")
        audio_dur = 60.0 # fallback
        
    # We will use FFmpeg filter_complex to concatenate all b-roll clips,
    # scale them to 1080x1920 (crop if necessary), and loop if needed.
    
    # Create a concat file
    concat_txt = os.path.join(os.path.dirname(output_path), "broll_concat.txt")
    with open(concat_txt, "w") as f:
        # We loop through the files multiple times just in case they are too short
        # (A bit hacky, but works perfectly for simple FFmpeg concat without complex looping)
        for _ in range(10): 
            for broll in broll_files:
                # Convert backslashes for FFmpeg concat demuxer if on Windows
                bpath = broll.replace("\\", "/")
                f.write(f"file '{bpath}'\n")
                
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_txt,
        "-i", tts_audio_path,
        "-filter_complex",
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1[v]",
        "-map", "[v]",
        "-map", "1:a",
        "-t", str(audio_dur),
        "-c:v", "h264_nvenc", "-preset", "p6", "-b:v", "5M",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"  [Video Builder] FFmpeg error: {e.stderr.decode('utf-8')}")
        raise
        
    print(f"  [Video Builder] Successfully built base video: {output_path}")
