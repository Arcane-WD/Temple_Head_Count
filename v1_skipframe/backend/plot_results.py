import json
import argparse
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    
    if not os.path.exists(args.log):
        print(f"Log file not found: {args.log}")
        return
        
    with open(args.log, "r") as f:
        data = json.load(f)
        
    timeline = data.get("timeline", [])
    if not timeline:
        print("No timeline data found.")
        return
        
    frames = [t["frame"] for t in timeline]
    visitors = [t["visitors"] for t in timeline]
    males = [t["male"] for t in timeline]
    females = [t["female"] for t in timeline]
    active = [t["active"] for t in timeline]
    
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Temple Analytics - Pipeline Results', fontsize=18, fontweight='bold', color='white')
    
    axs[0, 0].plot(frames, visitors, label="Total Visitors", color='#00ffff', linewidth=2)
    axs[0, 0].set_title("Cumulative Visitors")
    axs[0, 0].set_xlabel("Frames")
    axs[0, 0].set_ylabel("Count")
    axs[0, 0].grid(True, alpha=0.2)
    axs[0, 0].legend()
    
    axs[0, 1].plot(frames, active, label="Active Tracks", color='#ffa500', linewidth=2)
    axs[0, 1].set_title("Active Track Load (Re-ID matching)")
    axs[0, 1].set_xlabel("Frames")
    axs[0, 1].set_ylabel("Tracks")
    axs[0, 1].grid(True, alpha=0.2)
    axs[0, 1].legend()
    
    axs[1, 0].plot(frames, males, label="Male", color='#4da6ff', linewidth=2)
    axs[1, 0].plot(frames, females, label="Female", color='#ff66b3', linewidth=2)
    axs[1, 0].set_title("Demographics Over Time")
    axs[1, 0].set_xlabel("Frames")
    axs[1, 0].set_ylabel("Count")
    axs[1, 0].grid(True, alpha=0.2)
    axs[1, 0].legend()
    
    axs[1, 1].axis('off')
    summary = data.get("summary", {})
    text_str =  f"--- PIPELINE SUMMARY ---\n\n"
    text_str += f"Final Visitors : {summary.get('visitors')}\n"
    text_str += f"Processing FPS : {summary.get('fps', 0):.1f}\n"
    text_str += f"Total Time     : {summary.get('elapsed', 0):.1f}s"
    
    axs[1, 1].text(0.1, 0.5, text_str, fontsize=16, va='center', family='monospace', color='#e6e6e6')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
