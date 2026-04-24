"""
DEMO: CUDA GPU Pipeline (V2 Adaptive)
Supports --out-json for benchmark comparison output.
"""
import cv2
import time
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.counter import TempleCounter
import config

def run_gpu(video_path: str, out_json: str = None):
    print(f"\n{'='*60}")
    print(f"  ██  V2 ADAPTIVE — CUDA GPU PIPELINE  ██")
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
    print(f"  Interval   : Re-ID triggered every {config.DETECTION_INTERVAL} frames\n")

    counter = TempleCounter(override_device="cuda")
    frame_idx = 0
    t0 = time.time()
    timeline = []  # Benchmark data collected every 50 frames

    hdr = f"{'FRAME':>7} | {'VISITORS':>8} | {'MALE':>4} | {'FEM':>4} | {'UNK':>4} | {'ACTIVE':>6} | {'HEALED':>6} | {'TIME':>6}"
    print(hdr)
    print("-" * len(hdr))

    last_milestone = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, visitors = counter.process_frame(frame, frame_idx)

        # Collect benchmark timeline every 50 frames
        if frame_idx % 50 == 0:
            elapsed_now = time.time() - t0
            timeline.append({
                "frame": frame_idx,
                "visitors": visitors,
                "male": counter.male_count,
                "female": counter.female_count,
                "unknown": counter.unknown_count,
                "active": len(counter.reid_tracker.active_identities),
                "healed": counter.healed_switches,
                "elapsed_s": round(elapsed_now, 2),
                "fps": round(frame_idx / max(elapsed_now, 0.001), 1),
            })

        current_milestone = frame_idx // 100
        if current_milestone > last_milestone:
            last_milestone = current_milestone
            elapsed_now = time.time() - t0
            m = counter.male_count
            f = counter.female_count
            u = counter.unknown_count
            active = len(counter.reid_tracker.active_identities)
            healed = counter.healed_switches
            print(f"{frame_idx:>7} | {visitors:>8} | {m:>4} | {f:>4} | {u:>4} | {active:>6} | {healed:>6} | {elapsed_now:>5.0f}s")

        frame_idx += 1

    elapsed = time.time() - t0
    cap.release()

    real_fps = frame_idx / max(elapsed, 0.001)
    print(f"\n{'='*60}")
    print(f"  ██  V2 GPU RESULT  ██")
    print(f"  Frames     : {frame_idx}")
    print(f"  Time       : {elapsed:.1f}s")
    print(f"  Speed      : {real_fps:.1f} fps")
    print(f"  Visitors   : {counter.reid_tracker.cumulative_visitors}")
    print(f"  Male       : {counter.male_count}")
    print(f"  Female     : {counter.female_count}")
    print(f"  Unknown    : {counter.unknown_count}")
    print(f"  ID Healed  : {counter.healed_switches} (Fragmentations repaired)")
    print(f"{'='*60}\n")

    if out_json:
        result = {
            "version": "v2_adaptive",
            "video": os.path.basename(video_path),
            "total_frames": frame_idx,
            "total_time_s": round(elapsed, 2),
            "fps": round(real_fps, 1),
            "visitors": counter.reid_tracker.cumulative_visitors,
            "male": counter.male_count,
            "female": counter.female_count,
            "unknown": counter.unknown_count,
            "healed_switches": counter.healed_switches,
            "timeline": timeline,
        }
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[Benchmark] Results written to: {out_json}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V2 Adaptive GPU Demo")
    parser.add_argument("--video", type=str, default="demo_clip_1.mp4")
    parser.add_argument("--out-json", type=str, default=None, help="Write structured results to this JSON path")
    args = parser.parse_args()

    video_path = os.path.join(config.INPUT_DIR, args.video)
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        sys.exit(1)

    run_gpu(video_path, out_json=args.out_json)
