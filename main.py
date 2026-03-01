import cv2
import config
from core.counter import TempleCounter
from utils.video_io import get_video_properties, create_video_writer
from calibrate import auto_calibrate_gate
import os

def run_pipeline():
    os.makedirs(config.ANNOTATED_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    vid_number = input("Choose video number (e.g. 1): ")
    video_path = os.path.join(config.INPUT_DIR, f"InpVid{vid_number}.mp4")

    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    # --- OPTIONAL AUTO CALIBRATION ---
    if config.AUTO_CALIBRATE:
        print("🔧 Checking automatic gate calibration requirements...")
        temp_cap = cv2.VideoCapture(video_path)
        total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_cap.release()

        if total_frames < config.MIN_FRAMES_FOR_CALIBRATION:
            print(f"⚠️ Video too short ({total_frames} frames). Using manual line.")
        else:
            dynamic_frames = min(config.MAX_CALIBRATION_FRAMES, int(total_frames * config.CALIBRATION_FRACTION))
            calibrated_line = auto_calibrate_gate(video_path, frames_to_analyze=dynamic_frames)
            if calibrated_line:
                config.GATE_LINE = calibrated_line

    # --- Setup Video ---
    cap, w, h, fps = get_video_properties(video_path)
    output_path = os.path.join(config.ANNOTATED_DIR, f"temple_output_{vid_number}.mp4")
    out = create_video_writer(output_path, w, h, fps)

    # Initialize engine
    engine = TempleCounter()
    frame_count = 0

    print(f"🚀 Processing: {video_path}")

    # --- Processing Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 🚨 FIX: Pass frame_count as frame_idx
        annotated_frame, in_count, out_count = engine.process_frame(frame, frame_count)
        out.write(annotated_frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Frame {frame_count} | IN: {in_count} | OUT: {out_count}")

    cap.release()
    out.release()
    print(f"✅ Processing Complete. Total Entries: {in_count}")

if __name__ == "__main__":
    run_pipeline()