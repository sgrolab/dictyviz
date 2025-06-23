import os
import sys
import cv2
from tkinter import Tk, filedialog

def make_movie_from_frames(output_dir, output_filename="optical_flow_movie.avi", fps=10):
    # Gather flow frames sorted by filename
    frames = sorted([f for f in os.listdir(output_dir) if f.startswith("flow_") and f.endswith(".png")])
    
    if not frames:
        print(f"No flow PNG frames found in: {output_dir}")
        return

    # Read first frame to get video dimensions
    first_frame_path = os.path.join(output_dir, frames[0])
    frame = cv2.imread(first_frame_path)
    if frame is None:
        print(f"Error reading first frame: {first_frame_path}")
        return

    height, width, _ = frame.shape

    # Set up video writer
    output_path = os.path.join(output_dir, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for fname in frames:
        img = cv2.imread(os.path.join(output_dir, fname))
        if img is not None:
            writer.write(img)
        else:
            print(f"Warning: Skipping unreadable frame {fname}")

    writer.release()
    print(f"Movie saved to: {output_path}")

def select_output_folder():
    print("Please select the folder that contains flow PNG frames (e.g., optical_flow_output_<video>)")
    Tk().withdraw()
    folder = filedialog.askdirectory(title="Select Output Folder")
    return folder if folder and os.path.isdir(folder) else None

def main():
    if len(sys.argv) >= 2:
        output_dir = sys.argv[1]
    else:
        output_dir = select_output_folder()

    if not output_dir:
        print("Error: No valid folder selected.")
        sys.exit(1)

    make_movie_from_frames(output_dir)

if __name__ == "__main__":
    main()