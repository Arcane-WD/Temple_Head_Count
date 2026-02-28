import cv2
import config
from core.counter import TempleCounter
from utils.video_io import get_video_properties, create_video_writer
from calibrate import auto_calibrate_gate
import os

def run_pipeline():

    os.makedirs(config.ANNOTATED_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    vid_number = input("Choose video to calibrate: ")
    # --- OPTIONAL AUTO CALIBRATION ---
    if config.AUTO_CALIBRATE:
        print("🔧 Checking automatic gate calibration requirements...")
        
        # 1. Safely check video length
        temp_cap = cv2.VideoCapture(config.INPUT_VIDEO+f"{vid_number}.mp4")
        total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_cap.release()

        # 2. Short video bypass
        if total_frames < config.MIN_FRAMES_FOR_CALIBRATION:
            print(f"⚠️ Video too short ({total_frames} frames). Skipping auto-calibration.")
            print(f"✅ Using manual config.GATE_LINE: {config.GATE_LINE}")
        else:
            # 3. Dynamic frame calculation
            dynamic_frames = min(
                config.MAX_CALIBRATION_FRAMES, 
                int(total_frames * config.CALIBRATION_FRACTION)
            )
            print(f"⚙️ Calibration frames selected: {dynamic_frames}")
            
            calibrated_line = auto_calibrate_gate(
                config.INPUT_VIDEO+f"{vid_number}.mp4",
                frames_to_analyze=dynamic_frames
            )

            if calibrated_line is not None:
                config.GATE_LINE = calibrated_line
                print(f"✅ Using calibrated gate line: {config.GATE_LINE}")
            else:
                print("⚠️ Calibration failed. Using manual config.GATE_LINE.")

    # --- Setup Video ---
    cap, w, h, fps = get_video_properties(config.INPUT_VIDEO+f"{vid_number}.mp4")
    output_path = os.path.join(config.ANNOTATED_DIR, "temple_output"+f"{vid_number}.mp4")
    out = create_video_writer(output_path, w, h, fps)

    # ⚠️ IMPORTANT: Initialize engine AFTER calibration so it picks up the updated GATE_LINE
    engine = TempleCounter()
    frame_count = 0
    in_count = 0
    out_count = 0

    print(f"🚀 Processing started: {config.INPUT_VIDEO+f"{vid_number}.mp4"}")

    # --- Processing Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, in_count, out_count = engine.process_frame(frame)
        out.write(annotated_frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Frame {frame_count} | IN: {in_count} | OUT: {out_count}")

    cap.release()
    out.release()

    print(f"✅ Processing Complete. Total Entries: {in_count}")

if __name__ == "__main__":
    run_pipeline()