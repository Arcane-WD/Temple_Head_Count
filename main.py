import cv2
import config
from core.counter import TempleCounter
from utils.video_io import get_video_properties, create_video_writer
from calibrate import auto_calibrate_gate
import os

def run_pipeline():

    os.makedirs(config.ANNOTATED_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # --- OPTIONAL AUTO CALIBRATION ---
    if config.AUTO_CALIBRATE:
        print("🔧 Running automatic gate calibration...")
        calibrated_line = auto_calibrate_gate(
            config.INPUT_VIDEO,
            frames_to_analyze=config.CALIBRATION_FRAMES
        )

        if calibrated_line is not None:
            config.GATE_LINE = calibrated_line
            print(f"✅ Using calibrated gate line: {config.GATE_LINE}")
        else:
            print("⚠️ Calibration failed. Using existing config.GATE_LINE.")

    # --- Setup Video ---
    cap, w, h, fps = get_video_properties(config.INPUT_VIDEO)
    output_path = os.path.join(config.ANNOTATED_DIR, "temple_output.mp4")
    out = create_video_writer(output_path, w, h, fps)

    engine = TempleCounter()
    frame_count = 0
    in_count = 0
    out_count = 0

    print(f"🚀 Processing started: {config.INPUT_VIDEO}")

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