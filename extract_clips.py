import os
import random
import subprocess
import cv2

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames / fps

def extract_random_clip(input_path, output_path, duration_sec=300):
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    total_duration = get_video_duration(input_path)
    if total_duration <= duration_sec:
        print(f"Video {input_path} is too short ({total_duration}s) for a {duration_sec}s clip.")
        return

    # Pick a random start time, ensuring we have enough video left for duration_sec
    max_start = total_duration - duration_sec
    start_time = random.uniform(0, max_start)
    
    # Format start time to HH:MM:SS
    start_time_str = time_formatter(start_time)

    print(f"Extracting a {duration_sec}s clip from {input_path} starting at {start_time_str}...")
    
    # Run FFmpeg to extract without re-encoding to preserve exact frame mapping,
    # or re-encode mildly if exact cutting is needed. -c copy is instantaneous.
    command = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", input_path,
        "-t", str(duration_sec),
        "-c:v", "copy",
        "-c:a", "copy",
        output_path
    ]
    
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Saved: {output_path}")

def time_formatter(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "global_assets", "input_vids")
    
    vid1_in = os.path.join(input_dir, "temple_vid_1.mp4")
    vid2_in = os.path.join(input_dir, "temple_vid_2.mp4")
    
    vid1_out = os.path.join(input_dir, "demo_clip_1.mp4")
    vid2_out = os.path.join(input_dir, "demo_clip_2.mp4")

    print(f"Generating 2-minute demo clips in {input_dir}\n")
    extract_random_clip(vid1_in, vid1_out, 300)
    extract_random_clip(vid2_in, vid2_out, 300)
    print("\nExtraction complete! You can use demo_clip_1.mp4 and demo_clip_2.mp4 for the final showcase.")
