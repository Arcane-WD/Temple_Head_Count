"""
DEMO: CUDA GPU Pipeline
Run this in one terminal while demo_cpu.py runs in another to show the speed difference.
"""
import cv2
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.counter import TempleCounter
import config

def run_gpu(video_path: str):
    print(f"\n{'='*60}")
    print(f"  ██  CUDA GPU PIPELINE  ██")
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
    print(f"  Male       : {counter.male_count}")
    print(f"  Female     : {counter.female_count}")
    print(f"  Unknown    : {counter.unknown_count}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPU Demo")
    parser.add_argument("--video", type=str, default="demo_clip_1.mp4")
    args = parser.parse_args()

    video_path = os.path.join(config.INPUT_DIR, args.video)
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found. Run extract_clips.py first!")
        sys.exit(1)

    run_gpu(video_path)
