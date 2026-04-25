import cv2
import time
import os
import sys
import json
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.counter import TempleCounter
import config

def run_gpu(video_path: str):
    print(f"\n{'='*60}")
    print(f"  ██  CUDA GPU PIPELINE (with Output & Plotter)  ██")
    print(f"  Video: {os.path.basename(video_path)}")
    print(f"{'='*60}\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open '{video_path}'")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_min = total_frames / fps / 60

    print(f"  Device     : CUDA (RTX 3050 Ti)")
    print(f"  Frames     : {total_frames} ({duration_min:.1f} min)")
    print(f"  Skip       : every {config.REID_SKIP_FRAMES} frames\n")

    counter = TempleCounter(override_device="cuda")
    frame_idx = 0
    t0 = time.time()

    hdr = f"{'FRAME':>7} | {'VISITORS':>8} | {'MALE':>4} | {'FEM':>4} | {'UNK':>4} | {'ACTIVE':>6} | {'TIME':>6}"
    print(hdr)
    print("-" * len(hdr))

    last_milestone = 0
    timeline = []
    
    basename = os.path.splitext(os.path.basename(video_path))[0]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, visitors = counter.process_frame(frame, frame_idx)

        current_milestone = frame_idx // 100
        if current_milestone > last_milestone:
            last_milestone = current_milestone
            elapsed = time.time() - t0
            m = counter.male_count
            f = counter.female_count
            u = counter.unknown_count
            active = len(counter.reid_tracker.active_tracks)
            print(f"{frame_idx:>7} | {visitors:>8} | {m:>4} | {f:>4} | {u:>4} | {active:>6} | {elapsed:>5.0f}s")
            
            timeline.append({
                "frame": frame_idx,
                "visitors": visitors,
                "male": m,
                "female": f,
                "unknown": u,
                "active": active,
                "time_sec": elapsed
            })

        frame_idx += 1

    elapsed = time.time() - t0
    cap.release()

    real_fps = frame_idx / max(elapsed, 0.001)
    print(f"\n{'='*60}")
    print(f"  ██  GPU RESULT  ██")
    print(f"  Frames     : {frame_idx}")
    print(f"  Time       : {elapsed:.1f}s")
    print(f"  Speed      : {real_fps:.1f} fps")
    print(f"  Visitors   : {counter.reid_tracker.cumulative_visitors}")
    print(f"{'='*60}\n")
    
    # Log JSON
    log_path = os.path.join(config.LOG_DIR, f"log_{basename}.json")
    os.makedirs(config.LOG_DIR, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({"timeline": timeline, "summary": {"visitors": counter.reid_tracker.cumulative_visitors, "fps": real_fps, "elapsed": elapsed}}, f, indent=4)
        
    print(f"  Data logged to: {log_path}")
    print("  Starting plot viewer...")
    
    # Plot results
    plotter_script = os.path.join(os.path.dirname(__file__), "plot_results.py")
    if os.path.exists(plotter_script):
        subprocess.Popen([sys.executable, plotter_script, "--log", log_path])
    else:
        print("  Error: plot_results.py not found.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPU Demo")
    parser.add_argument("--video", type=str, default="demo_clip_1.mp4")
    args = parser.parse_args()

    video_path = os.path.join(config.INPUT_DIR, args.video)
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found. Ensure it is inside global_assets/input_vids.")
        sys.exit(1)

    run_gpu(video_path)
