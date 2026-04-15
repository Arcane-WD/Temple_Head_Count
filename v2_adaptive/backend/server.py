import os
import sys
import uuid
import json
import time
import queue
import threading
import shutil
import subprocess
import traceback

import cv2
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

import config
from core.counter import TempleCounter
from utils.video_io import get_video_properties, create_video_writer

app = FastAPI(title="Temple Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ──────────────────────────────────────────────────────
jobs: dict = {}


class ProcessRequest(BaseModel):
    filename: str


# ── GET /api/videos ──────────────────────────────────────────────────────────
@app.get("/api/videos")
def list_videos():
    """List available .mp4 files in the input directory."""
    os.makedirs(config.INPUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(config.INPUT_DIR) if f.endswith(".mp4")]
    return {"videos": sorted(files)}


# ── POST /api/upload ─────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a new video file to the input directory."""
    if not file.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported.")
    
    os.makedirs(config.INPUT_DIR, exist_ok=True)
    file_path = os.path.join(config.INPUT_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"filename": file.filename, "message": "Upload successful"}


# ── POST /api/process ────────────────────────────────────────────────────────
@app.post("/api/process")
def start_processing(req: ProcessRequest):
    """Start processing a video. Returns a job_id for tracking progress."""
    video_path = os.path.join(config.INPUT_DIR, req.filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {req.filename}")

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "starting",
        "filename": req.filename,
        "frame": 0,
        "total_frames": 0,
        "visitors": 0,
        "male": 0,
        "female": 0,
        "unknown": 0,
        "fps": 0,
        "warnings": [],
        "errors": [],
        "timeline": [],        # [{frame, visitors, male, female, unknown}]
        "output_file": None,
        "done": False,
    }

    thread = threading.Thread(target=_run_pipeline, args=(job_id, video_path), daemon=True)
    thread.start()

    return {"job_id": job_id}


def _run_pipeline(job_id: str, video_path: str):
    """Background pipeline runner."""
    pipeline_start = time.time()
    job = jobs[job_id]

    try:
        os.makedirs(config.ANNOTATED_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)

        # ── Frame count check ────────────────────────────────────────────
        temp_cap = cv2.VideoCapture(video_path)
        total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_cap.release()
        job["total_frames"] = total_frames

        if total_frames < 30:
            job["errors"].append({
                "code": "LOW_FRAME_COUNT",
                "message": f"Video has only {total_frames} frames.",
                "layman": "This video clip is too short for meaningful analysis. A longer recording is needed for accurate counting."
            })
            job["status"] = "error"
            job["done"] = True
            return

        # ── Calibration ──────────────────────────────────────────────────
        # (Calibration step removed: Gate lines are no longer needed for Re-ID counting)

        # ── Processing ───────────────────────────────────────────────────
        job["status"] = "processing"
        cap, w, h, fps = get_video_properties(video_path)
        job["fps"] = fps

        basename = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"temple_output_{basename}.mp4"
        output_path = os.path.join(config.ANNOTATED_DIR, output_filename)

        # V2 runs tracking every frame — output fps equals source fps
        output_fps = max(1, fps)

        # Async video writer: decouple disk I/O from ML loop
        write_queue = queue.Queue(maxsize=30)
        write_done = threading.Event()

        def writer_thread():
            # Open an FFmpeg subprocess directly
            command = [
                'ffmpeg',
                '-y', # Overwrite
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', '1280x720',
                '-pix_fmt', 'bgr24',
                '-r', str(output_fps),
                '-i', '-', # Read from stdin
                '-c:v', 'libx264',
                '-preset', 'superfast',
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            
            while True:
                item = write_queue.get()
                if item is None:
                    break
                process.stdin.write(item.tobytes())
                write_queue.task_done()
                
            process.stdin.close()
            process.wait()
            write_done.set()

        wt = threading.Thread(target=writer_thread, daemon=True)
        wt.start()

        engine = TempleCounter()

        # ── Local tracking vars (avoid GIL contention on job dict) ────
        current_frame = 0
        current_visitors = 0
        current_male = 0
        current_female = 0
        current_unknown = 0
        pipeline_done = threading.Event()

        # ── Reporting thread: syncs locals → job dict every 2s ────────
        def reporting_thread():
            while not pipeline_done.is_set():
                time.sleep(2)
                job["frame"] = current_frame
                job["visitors"] = current_visitors
                job["male"] = current_male
                job["female"] = current_female
                job["unknown"] = current_unknown
                job["timeline"].append({
                    "frame": current_frame,
                    "visitors": current_visitors,
                    "male": current_male,
                    "female": current_female,
                    "unknown": current_unknown,
                })

        rt = threading.Thread(target=reporting_thread, daemon=True)
        rt.start()

        # ── ML hot loop (no job dict writes here) ─────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, visitors = engine.process_frame(frame, current_frame)

            # V2 always returns an annotated frame — queue it every frame
            if annotated_frame is not None:
                small_frame = cv2.resize(annotated_frame, (1280, 720))
                if not write_queue.full():
                    write_queue.put_nowait(small_frame)

            current_frame += 1
            current_visitors = visitors
            current_male = engine.male_count
            current_female = engine.female_count
            current_unknown = engine.unknown_count

        # ── Cleanup ───────────────────────────────────────────────────
        cap.release()

        # Stop reporting thread
        pipeline_done.set()
        rt.join(timeout=5)

        # Final commit to job dict (ensures last-frame accuracy)
        job["frame"] = current_frame
        job["visitors"] = current_visitors
        job["male"] = current_male
        job["female"] = current_female
        job["unknown"] = current_unknown

        # Stop writer thread
        write_queue.put(None)
        write_done.wait()

        # Ending time log
        elapsed = time.time() - pipeline_start
        processing_fps = current_frame / max(elapsed, 0.001)


        # Final timeline point
        job["timeline"].append({
            "frame": current_frame,
            "visitors": current_visitors,
            "male": current_male,
            "female": current_female,
            "unknown": current_unknown,
        })

        job["output_file"] = output_filename
        job["status"] = "complete"
        job["done"] = True

        # ── Final summary log (mirrors CLI output for demo visibility) ────
        device_label = "GPU" if config.DEVICE == "cuda" else "CPU"

        print(f"\n{'='*60}")
        print(f"")
        print(f"  ██  {device_label} RESULT  ██")
        print(f"  Frames     : {current_frame}")
        print(f"  Time       : {elapsed:.1f}s")
        print(f"  Speed      : {processing_fps:.1f} fps")
        print(f"  Visitors   : {current_visitors}")
        print(f"  ")
        print(f"  ")
        print(f"  Male       : {current_male}")
        print(f"  Female     : {current_female}")
        print(f"  Unknown    : {current_unknown}")
        print(f"")
        print(f"{'='*60}\n")

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[PIPELINE ERROR] {e}\n{tb}")
        job["errors"].append({
            "code": "PIPELINE_ERROR",
            "message": str(e),
            "layman": "An unexpected error occurred during video processing. Please try again or use a different video."
        })
        job["status"] = "error"
        job["done"] = True


# ── GET /api/status/{job_id} (SSE) ───────────────────────────────────────────
@app.get("/api/status/{job_id}")
def stream_status(job_id: str):
    """Server-Sent Events stream for real-time progress updates."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    def event_generator():
        last_frame = -1
        while True:
            job = jobs.get(job_id)
            if job is None:
                break

            if job["frame"] != last_frame or job["done"]:
                data = json.dumps({
                    "status": job["status"],
                    "frame": job["frame"],
                    "total_frames": job["total_frames"],
                    "visitors": job["visitors"],
                    "male": job["male"],
                    "female": job["female"],
                    "unknown": job["unknown"],
                    "fps": job["fps"],
                    "warnings": job["warnings"],
                    "errors": job["errors"],
                    "done": job["done"],
                })
                yield f"data: {data}\n\n"
                last_frame = job["frame"]

            if job["done"]:
                break

            time.sleep(0.25)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── GET /api/results/{job_id} ────────────────────────────────────────────────
@app.get("/api/results/{job_id}")
def get_results(job_id: str):
    """Get final results for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return {
        "status": job["status"],
        "visitors": job["visitors"],
        "male": job["male"],
        "female": job["female"],
        "unknown": job["unknown"],
        "timeline": job["timeline"],
        "warnings": job["warnings"],
        "errors": job["errors"],
        "output_file": job["output_file"],
    }


# ── GET /api/video/{filename} ────────────────────────────────────────────────
@app.get("/api/video/{filename}")
def serve_video(filename: str):
    """Serve a video file (input or annotated output)."""
    # Check annotated output first, then input
    output_path = os.path.join(config.ANNOTATED_DIR, filename)
    input_path = os.path.join(config.INPUT_DIR, filename)

    if os.path.exists(output_path):
        return FileResponse(output_path, media_type="video/mp4")
    elif os.path.exists(input_path):
        return FileResponse(input_path, media_type="video/mp4")
    else:
        raise HTTPException(status_code=404, detail="Video file not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
