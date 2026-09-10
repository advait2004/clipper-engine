import uuid
import os
import shutil
from datetime import datetime
from typing import Dict
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from main import process_video
from config import Config
from pipeline.concat import find_videos_in_folder, concatenate_videos
from pipeline.caption_styles import list_styles as list_caption_styles
from pipeline.transitions import list_transitions
from pipeline.edit_styles import list_styles as list_edit_styles

app = FastAPI(title="Clipper Engine API")

# In-memory stores
jobs: Dict[str, dict] = {}
batches: Dict[str, dict] = {}


class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


class JobRequest(BaseModel):
    url: str
    caption_style: str = "karaoke"
    virality_mode: bool = True
    clips_per_video: int = 1


class BatchJobRequest(BaseModel):
    folder_path: str
    mode: str = "extract_highlights"
    edit_style: str = "podcast"
    caption_style: str = "karaoke"
    num_variations: int = 1
    clips_per_video: int = 1


class ScriptJobRequest(BaseModel):
    script_text: str
    caption_style: str = "karaoke"


class JobCancelledError(Exception):
    pass


def process_job_task(
    job_id: str,
    url: str,
    caption_style: str = "karaoke",
    virality_mode: bool = True,
    clips_per_video: int = 1,
):
    jobs[job_id]["status"] = JobStatus.PROCESSING

    def update_progress(percent=None, stage=None):
        if job_id in jobs:
            if jobs[job_id].get("cancelled"):
                raise JobCancelledError("Job cancelled by user")
            if percent is not None and stage is not None:
                jobs[job_id]["progress"] = percent
                jobs[job_id]["stage"] = stage

    try:
        cfg = Config()
        cfg.CAPTION_STYLE = caption_style
        cfg.VIRALITY_MODE = virality_mode
        cfg.CLIPS_PER_VIDEO = clips_per_video

        clips = process_video(url, cfg, progress_callback=update_progress)

        jobs[job_id]["status"] = JobStatus.DONE
        jobs[job_id]["clips"] = clips
        jobs[job_id]["progress"] = 100
        jobs[job_id]["stage"] = "Completed"
    except JobCancelledError as e:
        jobs[job_id]["status"] = JobStatus.ERROR
        jobs[job_id]["error"] = str(e)
    except Exception as e:
        jobs[job_id]["status"] = JobStatus.ERROR
        jobs[job_id]["error"] = str(e)


def process_script_task(job_id: str, script_text: str, caption_style: str):
    from main import process_script
    
    jobs[job_id]["status"] = JobStatus.PROCESSING

    def check_cancel():
        if jobs.get(job_id, {}).get("cancelled"):
            raise JobCancelledError("Job was cancelled by the user")

    def update_progress(pct: int, msg: str):
        if jobs.get(job_id, {}).get("cancelled"):
            raise JobCancelledError("Job was cancelled by the user")
        jobs[job_id]["progress"] = pct
        jobs[job_id]["stage"] = msg

    cfg = Config()
    cfg.CAPTION_STYLE = caption_style
    # Ensure virality mode things are enabled for script videos
    cfg.VIRALITY_MODE = True
    cfg.HOOK_TEXT_OVERLAY = False # Usually scripts don't need the text overlay hook at start

    try:
        clips = process_script(script_text, cfg, set_progress=update_progress, check_cancel=check_cancel)
        jobs[job_id]["status"] = JobStatus.DONE
        jobs[job_id]["clips"] = clips
        jobs[job_id]["progress"] = 100
        jobs[job_id]["stage"] = "Completed"
    except JobCancelledError:
        jobs[job_id]["status"] = JobStatus.ERROR
        jobs[job_id]["error"] = "Cancelled"
    except Exception as e:
        jobs[job_id]["status"] = JobStatus.ERROR
        jobs[job_id]["error"] = str(e)


def process_batch_task(
    batch_id: str,
    folder_path: str,
    mode: str,
    edit_style: str,
    caption_style: str,
    num_variations: int,
    clips_per_video: int = 1,
):
    """Process a batch of videos from a folder."""
    try:
        video_paths = find_videos_in_folder(folder_path)
        if not video_paths:
            batches[batch_id]["status"] = JobStatus.ERROR
            batches[batch_id]["error"] = f"No video files found in: {folder_path}"
            return

        cfg = Config()
        cfg.CLIPS_PER_VIDEO = clips_per_video
        max_clips = cfg.SMART_EDIT_MAX_SOURCE_CLIPS
        if len(video_paths) > max_clips:
            video_paths = video_paths[:max_clips]
            print(f"  ⚠ Limiting to {max_clips} source clips")

        batches[batch_id]["total_files"] = len(video_paths)
        batches[batch_id]["file_names"] = [os.path.basename(p) for p in video_paths]

        if mode == "merge_all":
            # ── Smart Edit mode ───────────────────────────────────────────────
            _process_smart_edit(
                batch_id, video_paths, cfg, edit_style, caption_style, num_variations
            )
        elif mode == "jumpcut_each":
            # ── Jump-cut each video individually ──────────────────────────────
            _process_jumpcut_each(
                batch_id, video_paths, cfg, edit_style, caption_style, num_variations
            )
        else:
            # ── Independent batch mode ────────────────────────────────────────
            _process_independent_batch(
                batch_id, video_paths, cfg, caption_style
            )

    except Exception as e:
        batches[batch_id]["status"] = JobStatus.ERROR
        batches[batch_id]["error"] = str(e)
        import traceback
        traceback.print_exc()


def _process_jumpcut_each(
    batch_id: str,
    video_paths: list[str],
    cfg: Config,
    edit_style: str,
    caption_style: str,
    num_variations: int,
):
    """Jump-cut each video individually."""
    from pipeline.content_analyzer import analyze_folder
    from pipeline.story_planner import plan_edit
    from pipeline.assembler import assemble
    from pipeline import captioner, sound_effects
    from pipeline.utils import safe_remove

    batches[batch_id]["status"] = JobStatus.PROCESSING
    batches[batch_id]["mode"] = "jumpcut_each"
    batches[batch_id]["job_ids"] = []

    def update_batch(stage: str, pct: int = None):
        batches[batch_id]["stage"] = stage
        if pct is not None:
            batches[batch_id]["progress"] = pct

    total_videos = len(video_paths)
    all_generated_clips = []

    for vid_idx, path in enumerate(video_paths):
        if batches[batch_id].get("cancelled"):
            break
            
        base_pct = int((vid_idx / total_videos) * 100)
        update_batch(f"Analyzing video {vid_idx + 1}/{total_videos}...", base_pct)

        def analysis_progress(pct, msg):
            batches[batch_id]["progress"] = base_pct + int((pct / 100) * (50 / total_videos))
            batches[batch_id]["stage"] = f"[{vid_idx + 1}/{total_videos}] {msg}"

        profiles = analyze_folder([path], cfg, progress_callback=analysis_progress)

        if not profiles or not profiles[0].get("words"):
            print(f"  ⚠ No speech in {path}, skipping.")
            continue

        for var_idx in range(num_variations):
            var_pct = base_pct + int((50 / total_videos)) + int((var_idx / num_variations) * (50 / total_videos))
            update_batch(f"[{vid_idx + 1}/{total_videos}] AI planning var {var_idx + 1}...", var_pct)
            edl = plan_edit(profiles, cfg, edit_style_name=edit_style, variation_index=var_idx)

            if not edl:
                continue

            update_batch(f"[{vid_idx + 1}/{total_videos}] Assembling var {var_idx + 1}...", var_pct + 5)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            assembled_path = os.path.join(cfg.TEMP_DIR, f"jumpcut_{stamp}_var{var_idx+1}_assembled.mp4")

            assembled_path, merged_words = assemble(edl, profiles, assembled_path, cfg.TEMP_DIR, cfg)

            update_batch(f"[{vid_idx + 1}/{total_videos}] Burning captions...", var_pct + 10)
            cap_path = os.path.join(cfg.TEMP_DIR, f"jumpcut_{stamp}_var{var_idx+1}_cap.mp4")
            captioner.burn_captions(assembled_path, merged_words, 0.0, cap_path, style_name=caption_style)

            final_path = os.path.join(cfg.OUTPUT_DIR, f"jumpcut_{stamp}_var{var_idx+1}_final.mp4")

            if cfg.SOUND_EFFECTS_ENABLED:
                update_batch(f"[{vid_idx + 1}/{total_videos}] Sound effects...", var_pct + 15)
                sound_effects.add_sound_effects(
                    cap_path, merged_words, 0.0, final_path,
                    bg_volume=cfg.SFX_BG_VOLUME,
                    whoosh_volume=cfg.SFX_WHOOSH_VOLUME,
                    impact_volume=cfg.SFX_IMPACT_VOLUME,
                )
            else:
                import shutil
                shutil.copy2(cap_path, final_path)

            all_generated_clips.append(final_path)

            if cfg.DELETE_TEMP:
                safe_remove(assembled_path)
                safe_remove(cap_path)

    if not all_generated_clips:
        batches[batch_id]["status"] = JobStatus.ERROR
        batches[batch_id]["error"] = "No clips were successfully generated"
        return

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": JobStatus.DONE,
        "url": "jumpcut_each",
        "clips": all_generated_clips,
        "error": None,
        "progress": 100,
        "stage": "Completed",
        "file_name": f"Jump-Cut Summaries ({len(all_generated_clips)} variants)",
    }
    batches[batch_id]["job_ids"] = [job_id]
    batches[batch_id]["status"] = JobStatus.DONE
    batches[batch_id]["stage"] = "Jump-cut summaries completed!"
    batches[batch_id]["progress"] = 100


def _process_smart_edit(
    batch_id: str,
    video_paths: list[str],
    cfg: Config,
    edit_style: str,
    caption_style: str,
    num_variations: int,
):
    """Smart Edit: analyze → plan → assemble → post-process."""
    from pipeline.content_analyzer import analyze_folder
    from pipeline.story_planner import plan_edit
    from pipeline.assembler import assemble
    from pipeline import reframer, captioner
    from pipeline import sound_effects
    from pipeline.utils import safe_remove

    batches[batch_id]["status"] = JobStatus.PROCESSING
    batches[batch_id]["mode"] = "smart_edit"

    def update_batch(stage: str, pct: int = None):
        batches[batch_id]["stage"] = stage
        if pct is not None:
            batches[batch_id]["progress"] = pct

    # ── 1. Analyze each clip ──────────────────────────────────────────────
    update_batch("Analyzing clips...", 5)

    def analysis_progress(pct, msg):
        batches[batch_id]["progress"] = pct
        batches[batch_id]["stage"] = msg

    profiles = analyze_folder(video_paths, cfg, progress_callback=analysis_progress)

    if not profiles or all(not p.get("words") for p in profiles):
        batches[batch_id]["status"] = JobStatus.ERROR
        batches[batch_id]["error"] = "No speech detected in any clip"
        return

    batches[batch_id]["job_ids"] = []
    generated_clips = []

    for var_idx in range(num_variations):
        print(f"\n─── Generating Variation {var_idx + 1} of {num_variations} ───")
        # ── 2. AI story planning ──────────────────────────────────────────────
        update_batch(f"AI planning edit {var_idx + 1}/{num_variations}...", 50 + (var_idx * 5))
        edl = plan_edit(profiles, cfg, edit_style_name=edit_style, variation_index=var_idx)

        if not edl:
            if var_idx == 0:
                batches[batch_id]["status"] = JobStatus.ERROR
                batches[batch_id]["error"] = "AI could not create an edit plan"
                return
            else:
                print("  ⚠ Failed to generate this variation, skipping.")
                continue

        # ── 3. Assemble segments ──────────────────────────────────────────────
        update_batch(f"Assembling variation {var_idx + 1}...", 60 + (var_idx * 5))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        assembled_path = os.path.join(cfg.TEMP_DIR, f"smart_{stamp}_var{var_idx+1}_assembled.mp4")

        assembled_path, merged_words = assemble(
            edl, profiles, assembled_path, cfg.TEMP_DIR, cfg
        )

        # ── 4. Burn captions ─────────────────────────────────────────────────
        update_batch(f"Burning captions for variation {var_idx + 1}...", 80 + (var_idx * 5))
        cap_path = os.path.join(cfg.TEMP_DIR, f"smart_{stamp}_var{var_idx+1}_cap.mp4")
        captioner.burn_captions(
            assembled_path,
            merged_words,
            0.0,  # clip_start is 0 because merged_words are already offset
            cap_path,
            style_name=caption_style,
        )

        # ── 6. Sound effects ─────────────────────────────────────────────────
        final_path = os.path.join(cfg.OUTPUT_DIR, f"smart_{stamp}_var{var_idx+1}_final.mp4")

        if cfg.SOUND_EFFECTS_ENABLED:
            update_batch(f"Adding sound effects (var {var_idx + 1})...", 90)
            sound_effects.add_sound_effects(
                cap_path, merged_words, 0.0, final_path,
                bg_volume=cfg.SFX_BG_VOLUME,
                whoosh_volume=cfg.SFX_WHOOSH_VOLUME,
                impact_volume=cfg.SFX_IMPACT_VOLUME,
            )
        else:
            import shutil
            shutil.copy2(cap_path, final_path)

        generated_clips.append(final_path)

        # ── Cleanup ───────────────────────────────────────────────────────────
        if cfg.DELETE_TEMP:
            safe_remove(assembled_path)
            safe_remove(cap_path)

    if not generated_clips:
        batches[batch_id]["status"] = JobStatus.ERROR
        batches[batch_id]["error"] = "No variations were successfully generated"
        return

    # ── Done ──────────────────────────────────────────────────────────────
    # Create a single child job entry for the result
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": JobStatus.DONE,
        "url": "smart_edit",
        "clips": generated_clips,
        "error": None,
        "progress": 100,
        "stage": "Completed",
        "file_name": f"Smart Edit ({len(generated_clips)} variants)",
    }
    batches[batch_id]["job_ids"] = [job_id]
    batches[batch_id]["status"] = JobStatus.DONE
    batches[batch_id]["stage"] = "Smart edit completed!"
    batches[batch_id]["progress"] = 100

    print(f"\n  ✅ Smart Edit complete: Generated {len(generated_clips)} variations.")


def _process_independent_batch(
    batch_id: str,
    video_paths: list[str],
    cfg: Config,
    caption_style: str,
):
    """Process each video independently."""
    job_ids = []
    for i, path in enumerate(video_paths):
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "status": JobStatus.PENDING,
            "url": path,
            "clips": [],
            "error": None,
            "progress": 0,
            "stage": "Queued",
            "file_name": os.path.basename(path),
        }
        job_ids.append(job_id)

    batches[batch_id]["job_ids"] = job_ids
    batches[batch_id]["status"] = JobStatus.PROCESSING

    for i, job_id in enumerate(job_ids):
        if batches[batch_id].get("cancelled"):
            break
        batches[batch_id]["stage"] = (
            f"Processing file {i + 1}/{len(job_ids)}: "
            f"{jobs[job_id]['file_name']}"
        )
        process_job_task(job_id, jobs[job_id]["url"], caption_style)

    # Final status
    all_done = all(
        jobs[jid]["status"] == JobStatus.DONE
        for jid in batches[batch_id].get("job_ids", [])
    )
    any_error = any(
        jobs[jid]["status"] == JobStatus.ERROR
        for jid in batches[batch_id].get("job_ids", [])
    )

    if all_done:
        batches[batch_id]["status"] = JobStatus.DONE
        batches[batch_id]["stage"] = "All files completed"
    elif any_error:
        batches[batch_id]["status"] = JobStatus.DONE
        batches[batch_id]["stage"] = "Completed with some errors"
    else:
        batches[batch_id]["status"] = JobStatus.DONE


# ─── Styles API ───────────────────────────────────────────────────────────────

@app.get("/api/styles")
async def get_styles():
    """Return available edit styles, caption styles, and transitions."""
    return {
        "edit_styles": list_edit_styles(),
        "caption_styles": list_caption_styles(),
        "transitions": list_transitions(),
    }


# ─── Single job endpoints ─────────────────────────────────────────────────────

@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    if job_id in jobs:
        jobs[job_id]["cancelled"] = True
        return {"status": "cancelling"}
    return JSONResponse(status_code=404, content={"error": "Job not found"})


@app.post("/api/jobs")
async def create_job(req: JobRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": JobStatus.PENDING,
        "url": req.url,
        "clips": [],
        "error": None,
        "progress": 0,
        "stage": "Starting...",
    }
    background_tasks.add_task(process_job_task, job_id, req.url, req.caption_style, req.virality_mode, req.clips_per_video)
    return {"job_id": job_id}


@app.post("/api/script-job")
async def create_script_job(req: ScriptJobRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": JobStatus.PENDING,
        "url": "Script-to-Video",
        "clips": [],
        "error": None,
        "progress": 0,
        "stage": "Starting...",
    }
    background_tasks.add_task(process_script_task, job_id, req.script_text, req.caption_style)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return jobs[job_id]


# ─── Batch job endpoints ──────────────────────────────────────────────────────

@app.post("/api/batch-jobs")
async def create_batch_job(req: BatchJobRequest, background_tasks: BackgroundTasks):
    batch_id = str(uuid.uuid4())
    batches[batch_id] = {
        "status": JobStatus.PENDING,
        "folder_path": req.folder_path,
        "mode": "smart_edit" if req.mode in ("merge_all", "jumpcut_each") else "batch",
        "edit_style": req.edit_style,
        "caption_style": req.caption_style,
        "job_ids": [],
        "total_files": 0,
        "file_names": [],
        "error": None,
        "stage": "Scanning folder...",
        "progress": 0,
        "processing_mode": req.mode,
    }
    background_tasks.add_task(
        process_batch_task, batch_id, req.folder_path,
        req.mode, req.edit_style, req.caption_style, req.num_variations, req.clips_per_video,
    )
    return {"batch_id": batch_id}


@app.get("/api/batch-jobs/{batch_id}")
async def get_batch_job(batch_id: str):
    if batch_id not in batches:
        return JSONResponse(status_code=404, content={"error": "Batch not found"})

    batch = batches[batch_id]
    job_statuses = []
    all_clips = []
    for jid in batch.get("job_ids", []):
        job = jobs.get(jid, {})
        job_statuses.append({
            "job_id": jid,
            "file_name": job.get("file_name", "unknown"),
            "status": job.get("status", "unknown"),
            "progress": job.get("progress", 0),
            "stage": job.get("stage", ""),
            "clips": job.get("clips", []),
            "error": job.get("error"),
        })
        all_clips.extend(job.get("clips", []))

    return {
        "batch_id": batch_id,
        "status": batch["status"],
        "processing_mode": batch.get("processing_mode", "extract_highlights"),
        "mode": batch.get("mode", "batch"),
        "edit_style": batch.get("edit_style", ""),
        "caption_style": batch.get("caption_style", ""),
        "total_files": batch["total_files"],
        "file_names": batch["file_names"],
        "stage": batch["stage"],
        "progress": batch.get("progress", 0),
        "error": batch["error"],
        "jobs": job_statuses,
        "all_clips": all_clips,
    }


@app.post("/api/batch-jobs/{batch_id}/cancel")
async def cancel_batch(batch_id: str):
    if batch_id not in batches:
        return JSONResponse(status_code=404, content={"error": "Batch not found"})
    batches[batch_id]["cancelled"] = True
    for jid in batches[batch_id].get("job_ids", []):
        if jid in jobs:
            jobs[jid]["cancelled"] = True
    return {"status": "cancelling"}


# ─── Static files & index ─────────────────────────────────────────────────────

output_dir = os.path.join(os.path.dirname(__file__), "output", "final")
os.makedirs(output_dir, exist_ok=True)
app.mount("/clips", StaticFiles(directory=output_dir), name="clips")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()
