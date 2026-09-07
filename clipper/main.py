#!/usr/bin/env python3
"""
Automated Video Clipping Engine
================================
Turns any YouTube video into viral-ready 9:16 clips with captions.

Usage
-----
Single video:
    python main.py https://youtu.be/VIDEO_ID

Multiple videos:
    python main.py https://youtu.be/ID1 https://youtu.be/ID2

From a text file (one URL per line):
    python main.py --file urls.txt

Interactive mode:
    python main.py
"""

import argparse
import json
import os
import sys
import time

# Force UTF-8 encoding for standard output/error on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from datetime import datetime

from config import Config
from pipeline import downloader, transcriber, scorer, clipper, reframer, captioner
from pipeline import sound_effects
from pipeline.utils import (
    filter_words_for_clip, safe_remove, build_windows, build_prompt,
    build_chunk_summary, build_coarse_prompt, snap_clip_boundaries,
)
from pipeline.pacing import clean_words_for_captions


# ─── AI scoring fallback chain ────────────────────────────────────────────────

def _is_fallback_clips(clips: list[dict]) -> bool:
    """Detect if scorer returned dumb fallback clips (all 'Auto clip N', score=5)."""
    if not clips:
        return True
    return all(
        c.get("hook", "").startswith("Auto clip") and c.get("score") == 5
        for c in clips
    )


def _score_with_fallback(
    words: list[dict],
    duration: float,
    cfg: Config,
    audio_hints: str = "",
    check_cancel=None
) -> list[dict]:
    """
    Score viral moments with improvements:
    1. Two-pass scoring (coarse scan → deep analysis on top chunks)
    2. Per-chunk delay to avoid rate limiting
    3. Per-chunk fallback chain (Groq → Gemini → Ollama)
    4. Audio energy hints passed to AI
    """
    MAX_BLOCK = 600.0  # 10 minutes
    if duration <= MAX_BLOCK:
        return _score_single_block(words, duration, cfg, audio_hints)

    print(f"\n  Video is {duration/60:.1f} mins long. Chunking into 10-min blocks...")

    # ── Build chunks ──────────────────────────────────────────────────────────
    chunks = []
    chunk_start = 0.0
    idx = 1
    while chunk_start < duration:
        chunk_end = min(chunk_start + MAX_BLOCK, duration)
        chunk_words = [w for w in words if chunk_start <= w["start"] < chunk_end]
        if chunk_words:
            chunks.append({
                "index": idx,
                "start": chunk_start,
                "end": chunk_end,
                "words": chunk_words,
            })
            idx += 1
        chunk_start = chunk_end

    print(f"  Created {len(chunks)} chunks")

    # ── Two-pass scoring ──────────────────────────────────────────────────────
    if cfg.TWO_PASS_SCORING and len(chunks) > cfg.TWO_PASS_TOP_CHUNKS:
        top_chunks = _two_pass_coarse_scan(chunks, duration, cfg, audio_hints)
    else:
        top_chunks = chunks  # Score all chunks directly

    # ── Deep-score each selected chunk ────────────────────────────────────────
    all_clips = []
    scored_count = 0
    failed_count = 0
    delay = cfg.SCORER_DELAY_BETWEEN_CHUNKS

    for i, chunk in enumerate(top_chunks):
        if check_cancel: check_cancel()
        print(f"\n  --- Deep-scoring chunk {chunk['index']}/{len(chunks)} "
              f"[{chunk['start']:.0f}s – {chunk['end']:.0f}s] ---")

        clips = _score_single_block(chunk["words"], duration, cfg, audio_hints)

        if clips and not _is_fallback_clips(clips):
            all_clips.extend(clips)
            scored_count += 1
        else:
            failed_count += 1

        # Rate-limit delay between chunks (skip after last chunk)
        if i < len(top_chunks) - 1 and delay > 0:
            print(f"  ⏳ Waiting {delay}s before next chunk (rate limit protection)...")
            for _ in range(int(delay)):
                if check_cancel: check_cancel()
                time.sleep(1)

    print(f"\n  Scoring complete: {scored_count} chunks scored, {failed_count} failed")

    if not all_clips:
        return scorer._fallback(duration)

    all_clips.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate overlapping clips from different chunks
    valid = []
    used = []
    for item in all_clips:
        start = item["start"]
        end = item["end"]
        overlap = any(not (end <= s or start >= e) for s, e in used)
        if overlap:
            continue
        used.append((start, end))
        valid.append(item)
        if len(valid) >= cfg.CLIPS_PER_VIDEO:
            break

    return valid


def _two_pass_coarse_scan(
    chunks: list[dict],
    duration: float,
    cfg: Config,
    audio_hints: str = "",
) -> list[dict]:
    """
    Pass 1: Send condensed summaries of ALL chunks in a single API call.
    Returns the top N chunks sorted by viral potential.
    """
    print(f"\n  ── Pass 1: Coarse scanning {len(chunks)} chunks (1 API call) ──")

    # Build condensed summaries
    chunk_summaries = []
    for chunk in chunks:
        summary = build_chunk_summary(chunk["words"], max_words=120)
        chunk_summaries.append({
            "index": chunk["index"],
            "start": chunk["start"],
            "end": chunk["end"],
            "summary": summary,
        })

    prompt = build_coarse_prompt(chunk_summaries, duration, audio_hints)

    # Try coarse scoring with the preferred provider
    rankings = []
    provider = cfg.AI_PROVIDER

    if provider == "gemini" and cfg.GEMINI_API_KEY not in ("", "YOUR_GEMINI_API_KEY_HERE"):
        rankings = scorer.score_gemini_coarse(prompt, cfg.GEMINI_API_KEY, cfg.GEMINI_MODEL)
    elif provider == "groq" and cfg.GROQ_API_KEY not in ("", "YOUR_GROQ_API_KEY_HERE"):
        rankings = scorer.score_groq_coarse(prompt, cfg.GROQ_API_KEY, cfg.GROQ_MODEL)
    elif provider == "nvidia" and cfg.NVIDIA_API_KEY not in ("", "YOUR_NVIDIA_API_KEY_HERE"):
        rankings = scorer.score_nvidia_coarse(prompt, cfg.NVIDIA_API_KEY, cfg.NVIDIA_MODEL)

    # Fallback: try other providers for coarse scoring
    if not rankings:
        if provider != "gemini" and cfg.GEMINI_API_KEY not in ("", "YOUR_GEMINI_API_KEY_HERE"):
            print("  🔄 Trying Gemini for coarse scan...")
            rankings = scorer.score_gemini_coarse(prompt, cfg.GEMINI_API_KEY, cfg.GEMINI_MODEL)
        if not rankings and provider != "groq" and cfg.GROQ_API_KEY not in ("", "YOUR_GROQ_API_KEY_HERE"):
            print("  🔄 Trying Groq for coarse scan...")
            rankings = scorer.score_groq_coarse(prompt, cfg.GROQ_API_KEY, cfg.GROQ_MODEL)
        if not rankings and provider != "nvidia" and cfg.NVIDIA_API_KEY not in ("", "YOUR_NVIDIA_API_KEY_HERE"):
            print("  🔄 Trying NVIDIA for coarse scan...")
            rankings = scorer.score_nvidia_coarse(prompt, cfg.NVIDIA_API_KEY, cfg.NVIDIA_MODEL)

    if not rankings:
        print("  ⚠ Coarse scan failed — scoring ALL chunks (slower)")
        return chunks

    # Sort by score and pick the top N chunks
    rankings.sort(key=lambda r: r.get("score", 0), reverse=True)
    top_n = cfg.TWO_PASS_TOP_CHUNKS
    top_indices = set()

    print(f"\n  Coarse rankings:")
    for r in rankings:
        idx = r.get("index", 0)
        sc = r.get("score", 0)
        reason = r.get("reason", "")[:60]
        marker = " ⭐" if len(top_indices) < top_n and idx > 0 else ""
        if idx > 0 and len(top_indices) < top_n:
            top_indices.add(idx)
        print(f"    Segment {idx}: {sc}/10 — {reason}{marker}")

    selected = [c for c in chunks if c["index"] in top_indices]
    print(f"\n  ── Pass 2: Deep-scoring top {len(selected)} chunks ──")

    # Add a delay after the coarse scan API call
    if cfg.SCORER_DELAY_BETWEEN_CHUNKS > 0:
        time.sleep(cfg.SCORER_DELAY_BETWEEN_CHUNKS)

    return selected


def _score_single_block(
    words: list[dict],
    duration: float,
    cfg: Config,
    audio_hints: str = "",
) -> list[dict]:
    """
    Try the user's preferred AI provider, then fall back through others.
    Chain: preferred → Gemini → Groq → Ollama → dumb fallback.
    Now passes audio_hints to each scorer.
    """
    providers_tried = set()
    clips = []

    # 1. Try preferred provider
    provider = cfg.AI_PROVIDER
    clips = _try_provider(provider, words, duration, cfg, audio_hints)
    providers_tried.add(provider)

    if clips and not _is_fallback_clips(clips):
        return clips

    # 2. Fallback chain: try other providers
    fallback_order = ["gemini", "groq", "ollama"]
    for fb_provider in fallback_order:
        if fb_provider in providers_tried:
            continue

        print(f"  🔄 Trying fallback provider: {fb_provider}")
        clips = _try_provider(fb_provider, words, duration, cfg, audio_hints)
        providers_tried.add(fb_provider)

        if clips and not _is_fallback_clips(clips):
            print(f"  ✅ {fb_provider} succeeded as fallback!")
            return clips

    # 3. All AI providers failed — return whatever we have (even fallback clips)
    print("  ⚠ All AI providers failed for this block")
    return clips or scorer._fallback(duration)


def _try_provider(
    provider: str,
    words: list[dict],
    duration: float,
    cfg: Config,
    audio_hints: str = "",
) -> list[dict]:
    """Try a single AI provider, return clips or empty list."""
    try:
        if provider == "gemini":
            if cfg.GEMINI_API_KEY in ("", "YOUR_GEMINI_API_KEY_HERE"):
                print(f"  ⚠ Gemini API key not set — skipping")
                return []
            return scorer.score_gemini(
                words, duration, cfg.GEMINI_API_KEY, cfg.GEMINI_MODEL,
                audio_hints=audio_hints, num_clips=cfg.CLIPS_PER_VIDEO,
            )

        elif provider == "groq":
            if cfg.GROQ_API_KEY in ("", "YOUR_GROQ_API_KEY_HERE"):
                print(f"  ⚠ Groq API key not set — skipping")
                return []
            return scorer.score_groq(
                words, duration, cfg.GROQ_API_KEY, cfg.GROQ_MODEL,
                audio_hints=audio_hints, num_clips=cfg.CLIPS_PER_VIDEO,
            )

        elif provider == "nvidia":
            if cfg.NVIDIA_API_KEY in ("", "YOUR_NVIDIA_API_KEY_HERE"):
                print(f"  ⚠ NVIDIA API key not set — skipping")
                return []
            return scorer.score_nvidia(
                words, duration, cfg.NVIDIA_API_KEY, cfg.NVIDIA_MODEL,
                audio_hints=audio_hints, num_clips=cfg.CLIPS_PER_VIDEO,
            )

        elif provider == "ollama":
            return scorer.score_ollama(
                words, duration, cfg.OLLAMA_MODEL, cfg.OLLAMA_URL,
                audio_hints=audio_hints, num_clips=cfg.CLIPS_PER_VIDEO,
            )

    except Exception as e:
        print(f"  ❌ Provider {provider} crashed: {e}")
        import traceback; traceback.print_exc()

    return []


# ─── Single video ─────────────────────────────────────────────────────────────

def process_video(url: str, cfg: Config, progress_callback=None) -> list[str]:
    """
    Full pipeline for one video URL.
    Returns list of paths to finished clips.
    """
    def set_progress(percent: int, stage_msg: str):
        if progress_callback:
            progress_callback(percent, stage_msg)

    def check_cancel():
        if progress_callback:
            progress_callback(None, None)

    url = url.strip(' "\'')
    stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = []

    _banner(f"Processing: {url}")

    raw_path = None
    is_local_file = False
    try:
        # ── 1. Download ───────────────────────────────────────────────────────
        if url.startswith("http://") or url.startswith("https://"):
            _step(1, "Downloading")
            set_progress(5, "Downloading video...")
            raw_path = downloader.download(url, cfg.RAW_DIR, getattr(cfg, "YOUTUBE_COOKIES_BROWSER", None))
        else:
            _step(1, "Using local file")
            set_progress(5, "Using local video file...")
            if not os.path.exists(url):
                raise FileNotFoundError(f"Local video file not found: {url}")
            if os.path.isdir(url):
                raise ValueError(f"Path is a directory, not a file: {url}. Please use the Smart Edit tab for folders.")
            raw_path = url
            is_local_file = True

        # ── 2. Transcribe ─────────────────────────────────────────────────────
        _step(2, "Transcribing")
        set_progress(25, "Transcribing audio (this may take a few minutes)...")
        words, duration = transcriber.transcribe(raw_path, cfg.WHISPER_MODEL)

        if not words:
            print("  ⚠ No speech detected — skipping video")
            return []

        # ── 2.5 Audio energy analysis (local, no API cost) ───────────────────
        audio_data = {}  # Always initialize for moment scorer
        audio_hints = ""
        if cfg.AUDIO_ENERGY_HINTS:
            _step("2.5", "Analyzing audio energy")
            set_progress(45, "Analyzing audio energy...")
            try:
                from pipeline.audio_analyzer import analyze_audio_energy, format_audio_hints
                audio_data = analyze_audio_energy(raw_path)
                audio_hints = format_audio_hints(audio_data)
                if audio_hints:
                    print(f"  Audio hints generated ({len(audio_hints)} chars)")
            except Exception as e:
                print(f"  ⚠ Audio analysis skipped: {e}")

        # ── 3. AI scoring (with two-pass + fallback chain) ────────────────────
        _step(3, "Finding viral moments (AI)")
        set_progress(50, "Finding viral moments with AI...")
        if cfg.TWO_PASS_SCORING:
            print("  Mode: Two-pass scoring (coarse scan → deep analysis)")
        clips = _score_with_fallback(words, duration, cfg, audio_hints, check_cancel)

        if not clips:
            print("  ⚠ No clips scored — skipping video")
            return []

        # ── 3.1 Reject clips with large internal silence gaps ─────────────
        validated_clips = []
        for clip in clips:
            clip_words = [w for w in words if clip["start"] <= w["start"] <= clip["end"]]
            has_big_gap = False
            if len(clip_words) >= 2:
                for j in range(1, len(clip_words)):
                    gap = clip_words[j]["start"] - clip_words[j-1]["end"]
                    if gap > 3.0:
                        has_big_gap = True
                        break
            # Also reject if very few words for the duration (mostly silence)
            clip_dur = clip["end"] - clip["start"]
            word_density = len(clip_words) / max(clip_dur, 0.1)
            if has_big_gap:
                print(f"    ⚠ Rejected clip [{clip['start']:.1f}s–{clip['end']:.1f}s]: "
                      f"contains >3s silence gap (hook: {clip.get('hook', '')[:50]})")
            elif word_density < 0.5 and clip_dur > 5:
                print(f"    ⚠ Rejected clip [{clip['start']:.1f}s–{clip['end']:.1f}s]: "
                      f"too sparse ({len(clip_words)} words in {clip_dur:.1f}s)")
            else:
                validated_clips.append(clip)
        
        if validated_clips:
            clips = validated_clips
        else:
            print("  ⚠ All clips rejected by silence filter — keeping originals")

        # ── 3.5 Smart clip boundary snapping ─────────────────────────────────
        if cfg.SMART_CLIP_BOUNDARIES:
            print("\n  Snapping clip boundaries to speech pauses...")
            for clip in clips:
                old_start, old_end = clip["start"], clip["end"]
                new_start, new_end = snap_clip_boundaries(
                    words, clip["start"], clip["end"],
                    min_dur=cfg.MIN_CLIP_SECONDS,
                    max_dur=cfg.MAX_CLIP_SECONDS,
                )
                clip["start"] = new_start
                clip["end"] = new_end
                if old_start != new_start or old_end != new_end:
                    print(f"    Snapped: [{old_start:.1f}s–{old_end:.1f}s] → "
                          f"[{new_start:.1f}s–{new_end:.1f}s]")

        _print_clips(clips)

        # Save metadata alongside output clips
        meta_file = os.path.join(cfg.OUTPUT_DIR, f"{stamp}_metadata.json")
        with open(meta_file, "w") as f:
            json.dump({"url": url, "duration": duration, "clips": clips}, f, indent=2)

        # ── 4–7. Per-clip: Cut → Reframe → Captions → Sound ─────────────────
        total_clips = min(len(clips), cfg.CLIPS_PER_VIDEO)
        for i, clip in enumerate(clips[: total_clips], start=1):
            check_cancel()
            clip_progress_base = 60 + (35 * (i - 1) // max(1, total_clips))
            set_progress(clip_progress_base, f"Processing clip {i} of {total_clips}...")

            cid      = f"{stamp}_clip{i}"
            cut_path = os.path.join(cfg.TEMP_DIR, f"{cid}_cut.mp4")
            rf_path  = os.path.join(cfg.TEMP_DIR, f"{cid}_rf.mp4")
            final    = os.path.join(cfg.OUTPUT_DIR, f"{cid}_final.mp4")

            print(f"\n  ── Clip {i}/{min(len(clips), cfg.CLIPS_PER_VIDEO)} "
                  f"[{clip['start']:.1f}s – {clip['end']:.1f}s] "
                  f"score={clip['score']}/10")
            print(f"     Hook: {clip['hook']}")

            cap_path = os.path.join(cfg.TEMP_DIR, f"{cid}_cap.mp4")

            try:
                # 4. Cut
                check_cancel()
                print("  [4/7] Cutting ...")
                is_virality = getattr(cfg, "VIRALITY_MODE", False)
                use_zoom = is_virality and getattr(cfg, "INTRO_ZOOM", False)
                use_flash_forward = is_virality and getattr(cfg, "FLASH_FORWARD_HOOK", False)
                
                flash_start = None
                flash_dur = 3.0
                if use_flash_forward:
                    flash_start = clip.get("flash_forward_start")
                    flash_end = clip.get("flash_forward_end")
                    if flash_start is None:
                        # Fallback: if AI forgot the field, grab the 3s segment at 25% into the clip
                        clip_dur = clip["end"] - clip["start"]
                        flash_start = clip["start"] + (clip_dur * 0.25)

                    if flash_end is None:
                        flash_end = flash_start + 4.5
                    elif flash_end - flash_start < 3.0:
                        # AI sometimes returns a flash forward that is only 1 second long.
                        # Force it to be at least 3 seconds long before snapping.
                        flash_end = flash_start + 3.0

                    if flash_start < clip["start"] or flash_start > clip["end"] - 3.0:
                        flash_start = clip["start"]
                        flash_end = flash_start + 4.5

                    # ── Multi-Signal Moment Scorer ────────────────────────────
                    # Overrides the LLM's text-only pick with a fused score
                    # combining text power words, audio energy, and video face data.
                    if getattr(cfg, "MOMENT_SCORER", False):
                        try:
                            from pipeline.moment_scorer import score_moments
                            best_start, best_end = score_moments(
                                raw_path, clip["start"], clip["end"],
                                words, audio_data,
                                llm_flash_start=flash_start,
                                llm_flash_end=flash_end,
                            )
                            flash_start = best_start
                            flash_end = best_end
                        except Exception as e:
                            print(f"    ⚠ MomentScorer failed ({e}), using LLM pick")

                    # Dynamic Snapping: Snap the start and end of the flash-forward 
                    # hook to natural speech pauses so words aren't chopped in half.
                    snapped_flash_start, snapped_flash_end = snap_clip_boundaries(
                        words, flash_start, flash_end,
                        min_dur=1.5, max_dur=15.0, pause_threshold=0.25, search_window=0.5
                    )
                    
                    # Ensure the dynamic hook doesn't overlap the end of the clip awkwardly
                    if snapped_flash_start < clip["start"] or snapped_flash_end >= clip["end"]:
                        snapped_flash_start = clip["start"]
                        snapped_flash_end = flash_end
                        
                    flash_start = snapped_flash_start
                    flash_dur = snapped_flash_end - snapped_flash_start
                
                clipper.cut(raw_path, clip["start"], clip["end"], cut_path, use_intro_zoom=use_zoom, flash_forward_start=flash_start, flash_forward_dur=flash_dur)

                # 5. Reframe to 9:16
                check_cancel()
                print("  [5/7] Reframing to 9:16 ...")
                reframer.reframe(cut_path, rf_path, cfg=cfg)

                # 6. Burn captions
                check_cancel()
                print("  [6/7] Burning captions ...")
                
                if use_flash_forward and flash_start is not None:
                    hook_words_raw = filter_words_for_clip(words, flash_start, flash_start + flash_dur)
                    main_words_raw = filter_words_for_clip(words, clip["start"], clip["end"])
                    
                    clip_words = []
                    # Shift hook words to start at clip["start"]
                    for w in hook_words_raw:
                        offset = w["start"] - flash_start
                        clip_words.append({
                            "word": w["word"], 
                            "start": clip["start"] + offset, 
                            "end": clip["start"] + (w["end"] - flash_start)
                        })
                    # Shift main words by +flash_dur
                    for w in main_words_raw:
                        offset = w["start"] - clip["start"]
                        clip_words.append({
                            "word": w["word"], 
                            "start": clip["start"] + flash_dur + offset, 
                            "end": clip["start"] + flash_dur + (w["end"] - clip["start"])
                        })
                    clip_duration = (clip["end"] - clip["start"]) + flash_dur
                else:
                    clip_words = filter_words_for_clip(words, clip["start"], clip["end"])
                    clip_duration = clip["end"] - clip["start"]

                # Pacing: clean filler words from captions
                if is_virality and getattr(cfg, "FILLER_WORD_REMOVAL", False):
                    clip_words, _ = clean_words_for_captions(clip_words, remove_fillers=True)

                # Overlays
                hook_overlay = clip.get("hook_text_overlay", "") if is_virality and getattr(cfg, "HOOK_TEXT_OVERLAY", False) else ""
                
                # Check if there is a fixed default CTA text in config
                default_cta = getattr(cfg, "CTA_DEFAULT_TEXT", "")
                cta_line = default_cta if default_cta else clip.get("cta_line", "")
                if not (is_virality and getattr(cfg, "CTA_OVERLAY", False)):
                    cta_line = ""

                captioner.burn_all_captions(
                    rf_path,
                    clip_words,
                    clip["start"],
                    clip_duration,
                    cap_path,
                    style_name=getattr(cfg, 'CAPTION_STYLE', 'karaoke'),
                    words_per_line=cfg.CAPTION_WORDS_PER_LINE,
                    font_size=cfg.CAPTION_FONT_SIZE,
                    uppercase=cfg.CAPTION_UPPERCASE,
                    hook_text_overlay=hook_overlay,
                    cta_line=cta_line,
                )

                # 7. Sound effects
                check_cancel()
                if cfg.SOUND_EFFECTS_ENABLED:
                    print("  [7/7] Adding sound effects ...")
                    sound_effects.add_sound_effects(
                        cap_path,
                        clip_words,
                        clip["start"],
                        final,
                        bg_volume=cfg.SFX_BG_VOLUME,
                        whoosh_volume=cfg.SFX_WHOOSH_VOLUME,
                        impact_volume=cfg.SFX_IMPACT_VOLUME,
                    )
                else:
                    import shutil
                    shutil.copy2(cap_path, final)

                output.append(final)
                size_mb = os.path.getsize(final) / 1_048_576
                print(f"  ✅ Saved: {os.path.basename(final)} ({size_mb:.1f} MB)")

            except Exception as e:
                print(f"  ❌ Clip {i} failed: {e}")
                import traceback; traceback.print_exc()

            finally:
                if cfg.DELETE_TEMP:
                    safe_remove(cut_path)
                    safe_remove(rf_path)
                    safe_remove(cap_path)

    except Exception as e:
        print(f"\n❌ Fatal error on {url}: {e}")
        import traceback; traceback.print_exc()
        raise e

    finally:
        if cfg.DELETE_RAW and raw_path and not is_local_file:
            safe_remove(raw_path)

    set_progress(100, "Done")
    return output


# ─── Batch ────────────────────────────────────────────────────────────────────

def batch_process(urls: list[str], cfg: Config):
    """Process a list of video URLs."""
    all_clips: list[str] = []

    for idx, url in enumerate(urls, start=1):
        print(f"\n{'-'*60}")
        print(f"VIDEO {idx} / {len(urls)}")
        try:
            clips = process_video(url, cfg)
            all_clips.extend(clips)
        except Exception:
            print(f"Skipping video due to error: {url}")

    _banner("Done")
    print(f"Generated {len(all_clips)} clips → {cfg.OUTPUT_DIR}")
    for path in all_clips:
        print(f"  📹 {os.path.basename(path)}")

    return all_clips


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Automated Video Clipping Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "urls", nargs="*",
        help="YouTube URLs to process"
    )
    parser.add_argument(
        "--file", "-f",
        metavar="FILE",
        help="Text file with one YouTube URL per line"
    )
    parser.add_argument(
        "--ai", choices=["gemini", "ollama", "groq", "nvidia"],
        default=None,
        help="Override AI provider from config (gemini | ollama | groq | nvidia)"
    )
    parser.add_argument(
        "--clips", type=int, default=None,
        help="Override number of clips per video"
    )
    parser.add_argument(
        "--no-two-pass", action="store_true",
        help="Disable two-pass scoring (score all chunks directly)"
    )
    parser.add_argument(
        "--no-audio-hints", action="store_true",
        help="Disable audio energy analysis hints"
    )
    args = parser.parse_args()

    cfg = Config()
    if args.ai:
        cfg.AI_PROVIDER = args.ai
    if args.clips:
        cfg.CLIPS_PER_VIDEO = args.clips
    if args.no_two_pass:
        cfg.TWO_PASS_SCORING = False
    if args.no_audio_hints:
        cfg.AUDIO_ENERGY_HINTS = False

    # Collect URLs
    urls = list(args.urls)
    if args.file:
        with open(args.file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)

    # Interactive fallback
    if not urls:
        print("--- Automated Video Clipping Engine ---")
        print(f"AI provider    : {cfg.AI_PROVIDER}")
        print(f"Whisper        : {cfg.WHISPER_MODEL}")
        print(f"Clips/video    : {cfg.CLIPS_PER_VIDEO}")
        print(f"Two-pass       : {'ON' if cfg.TWO_PASS_SCORING else 'OFF'}")
        print(f"Audio hints    : {'ON' if cfg.AUDIO_ENERGY_HINTS else 'OFF'}")
        print(f"Smart boundaries: {'ON' if cfg.SMART_CLIP_BOUNDARIES else 'OFF'}")
        print()
        raw = input("Enter YouTube URL(s), comma-separated: ").strip()
        urls = [u.strip() for u in raw.split(",") if u.strip()]

    if not urls:
        print("No URLs provided. Exiting.")
        sys.exit(0)

    batch_process(urls, cfg)


# ─── Print helpers ────────────────────────────────────────────────────────────

def _banner(msg: str):
    print(f"\n{'-'*60}")
    print(f"  {msg}")
    print(f"{'-'*60}")


def _step(n, label: str):
    print(f"\n  [{n}/7] {label} ...")


def _print_clips(clips: list[dict]):
    print(f"\n  Found {len(clips)} viral moments:")
    for i, c in enumerate(clips, 1):
        dur = c["end"] - c["start"]
        print(
            f"    {i}. [{c['start']:.0f}s–{c['end']:.0f}s] "
            f"{dur:.0f}s  score={c['score']}/10  |  {c['hook']}"
        )


if __name__ == "__main__":
    main()

