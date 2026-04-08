import cv2
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.counter import TempleCounter
import config


def run_pipeline(video_path: str, device: str = None):
    print(f"\n{'='*60}")
    print(f"  STARTING RE-ID PIPELINE BENCHMARK: [{device.upper()}]")
    print(f"  Video: {video_path}")
    print(f"{'='*60}\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open '{video_path}'")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    print(f"  Model      : {config.REID_MODEL} (skip every {config.REID_SKIP_FRAMES} frames)")
    print(f"  Threshold  : {config.REID_MATCH_THRESHOLD}")
    print(f"  Device     : {device}")
    print(f"  Frames     : {total_frames} ({total_frames / fps / 60:.1f} min)\n")

    counter = TempleCounter(override_device=device)
    frame_idx = 0
    t0 = time.time()

    # Print every 10th processed frame in the detailed table
    log_interval = 10
    processed_count = 0

    hdr = f"{'FRAME':>7} | {'VISITORS':>8} | {'MALE':>4} | {'FEM':>4} | {'UNK':>4} | {'ACTIVE':>6}"
    print(hdr)
    print("-" * len(hdr))

    last_milestone = 0  # track 1000-frame milestone prints

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, visitors = counter.process_frame(frame, frame_idx)

        if frame_idx % config.REID_SKIP_FRAMES == 0:
            processed_count += 1
            if processed_count % log_interval == 0:
                m = counter.male_count
                f = counter.female_count
                u = counter.unknown_count
                active = len(counter.reid_tracker.active_tracks)
                print(f"{frame_idx:>7} | {visitors:>8} | {m:>4} | {f:>4} | {u:>4} | {active:>6}")

        # Print a bold summary line every 1000 raw frames
        current_milestone = frame_idx // 1000
        if current_milestone > last_milestone:
            last_milestone = current_milestone
            elapsed = time.time() - t0
            m = counter.male_count
            f = counter.female_count
            u = counter.unknown_count
            v = counter.reid_tracker.cumulative_visitors
            print(f"  --- FRAME {frame_idx:>7} | {elapsed:.0f}s | Visitors: {v} | M:{m} F:{f} U:{u} ---")

        frame_idx += 1
        
        if frame_idx >= 2000:
            print("\n  [Terminating early at 2000 frames for benchmark]")
            break

    elapsed = time.time() - t0
    cap.release()

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Time       : {elapsed:.1f}s ({frame_idx / max(elapsed, 0.001):.1f} fps)")
    print(f"  Visitors   : {counter.reid_tracker.cumulative_visitors}")
    print(f"  Male       : {counter.male_count}")
    print(f"  Female     : {counter.female_count}")
    print(f"  Unknown    : {counter.unknown_count}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_video = os.path.join(config.INPUT_DIR, "temple_vid_1.mp4")
    
    print("\n\n" + "#"*60)
    print(">>> 1. RUNNING CPU-ONLY BENCHMARK")
    print("#"*60)
    run_pipeline(test_video, device="cpu")
    
    print("\n\n" + "#"*60)
    print(">>> 2. RUNNING CUDA GPU BENCHMARK")
    print("#"*60)
    run_pipeline(test_video, device="cuda")
