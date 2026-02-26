"""
Exercise 2 — Video Surveillance Agent (LLaVA via Ollama)

Pipeline:
  1. Extract one frame every 2 seconds from the input video (OpenCV)
  2. Ask LLaVA "is there a person in this frame?" for each frame
  3. Track state transitions (empty → occupied, occupied → empty)
  4. Print a timestamped entry/exit report
"""

import os
import cv2
import ollama

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MODEL = "llava:13b"
VIDEO_PATH = "/content/cs6501workshop/Topic6VLM/2minute.mov"
FRAME_DIR      = "frames"
FRAME_INTERVAL = 2          # seconds between sampled frames
MAX_SIDE       = 216        # resize frames before sending to LLaVA

# Prompt designed to elicit a clean yes/no answer
DETECTION_PROMPT = (
    "Look at this image carefully. "
    "Is there a person (human being) visible anywhere in the scene? "
    "Reply with only one word: YES or NO."
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Extract frames from video
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, frame_dir: str, interval_secs: float = FRAME_INTERVAL):
    """
    Extract one frame every `interval_secs` seconds from `video_path`.
    Saves frames as JPEGs in `frame_dir`.
    Returns a list of (timestamp_seconds, filepath) tuples.
    """
    os.makedirs(frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps      = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * interval_secs)   # frame count between samples
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    print(f"Video: {video_path}")
    print(f"  FPS: {fps:.1f}  |  Total frames: {total}  |  Duration: {duration:.1f}s")
    print(f"  Sampling every {interval_secs}s → ~{int(duration / interval_secs)} frames\n")

    frames   = []
    frame_n  = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_n % interval == 0:
            timestamp = frame_n / fps

            # Resize before saving to keep LLaVA calls fast
            h, w      = frame.shape[:2]
            if max(h, w) > MAX_SIDE:
                scale = MAX_SIDE / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)

            path = os.path.join(frame_dir, f"frame_{frame_n:06d}.jpg")
            cv2.imwrite(path, frame)
            frames.append((timestamp, path))

        frame_n += 1

    cap.release()
    print(f"Extracted {len(frames)} frames to '{frame_dir}/'")
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Ask LLaVA if a person is present in a single frame
# ─────────────────────────────────────────────────────────────────────────────

def detect_person(frame_path: str) -> bool:
    """
    Sends one frame to LLaVA with a yes/no prompt.
    Returns True if LLaVA says YES, False otherwise.
    """
    resp = ollama.chat(
        model=MODEL,
        messages=[{
            "role":    "user",
            "content": DETECTION_PROMPT,
            "images":  [frame_path],
        }]
    )
    answer = resp["message"]["content"].strip().upper()
    # Accept any response that starts with YES
    return answer.startswith("YES")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Run detection over all frames and find transitions
# ─────────────────────────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def run_surveillance(frames: list) -> list:
    """
    Iterates over all (timestamp, path) frames, calls LLaVA on each,
    and records entry/exit events when the person-present state changes.

    Returns a list of event dicts: {type, timestamp, frame_path}
    """
    events      = []
    prev_state  = False   # False = empty, True = person present
    total       = len(frames)

    print("\nScanning frames...\n")
    print(f"{'Frame':<8} {'Time':<8} {'Person?':<10} {'Event'}")
    print("─" * 45)

    for i, (timestamp, path) in enumerate(frames):
        present = detect_person(path)

        # Detect state transitions
        event_label = ""
        if present and not prev_state:
            event_label = "⬅️  ENTERED"
            events.append({"type": "enter", "timestamp": timestamp, "frame": path})
        elif not present and prev_state:
            event_label = "➡️  EXITED"
            events.append({"type": "exit",  "timestamp": timestamp, "frame": path})

        print(f"{i+1:<8} {format_time(timestamp):<8} {'YES' if present else 'NO':<10} {event_label}")
        prev_state = present

    return events


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Print the final report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(events: list):
    print("\n" + "═" * 45)
    print("  SURVEILLANCE REPORT")
    print("═" * 45)

    if not events:
        print("No person detected in any frame.")
        return

    for e in events:
        action = "ENTERED scene" if e["type"] == "enter" else "EXITED scene"
        print(f"  {format_time(e['timestamp'])}  —  Person {action}")

    print("═" * 45)

    # Summarise time-in-scene for each entry/exit pair
    enters = [e for e in events if e["type"] == "enter"]
    exits  = [e for e in events if e["type"] == "exit"]

    print("\nTime in scene:")
    for i, enter in enumerate(enters):
        if i < len(exits):
            duration = exits[i]["timestamp"] - enter["timestamp"]
            print(f"  Visit {i+1}: {format_time(enter['timestamp'])} → "
                  f"{format_time(exits[i]['timestamp'])}  "
                  f"({duration:.0f}s)")
        else:
            print(f"  Visit {i+1}: {format_time(enter['timestamp'])} → still in scene at end")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    frames = extract_frames(VIDEO_PATH, FRAME_DIR, FRAME_INTERVAL)
    events = run_surveillance(frames)
    print_report(events)