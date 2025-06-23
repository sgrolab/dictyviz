import os
import sys
import datetime
import numpy as np
import cv2
import imageio
from tkinter import Tk, filedialog

# Open file picker to select a .avi video
def select_video_file():
    print("Please select a .avi video file")
    Tk().withdraw()  # Hide the root window
    video_path = filedialog.askopenfilename(
        title="Select Video",
        filetypes=[("AVI files", "*.avi")]
    )
    return video_path if video_path and os.path.isfile(video_path) else None

# Compute dense optical flow using Farneback and save outputs
def compute_farneback_optical_flow(video_path, output_dir, log_file):
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()

    if not ret:
        print(f"Error: Could not read video from {video_path}", file=log_file)
        return

    prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(first_frame)
    hsv[..., 1] = 255  # Full saturation

    frame_index = 0
    flow_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5, levels=3, winsize=10,
            iterations=5, poly_n=7, poly_sigma=1.5, flags=0
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        imageio.imwrite(os.path.join(output_dir, f"flow_{frame_index:04d}.png"), rgb_flow)
        flow_list.append(flow)

        prev = curr
        frame_index += 1

    cap.release()
    np.save(os.path.join(output_dir, "flow_raw.npy"), np.stack(flow_list))
    print(f"Saved {frame_index} frames and raw flow data to {output_dir}", file=log_file)

# Set up output and run the process
def main():
    video_path = select_video_file()
    if not video_path:
        print("Error: No valid .avi file selected.")
        sys.exit(1)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    parent_dir = os.path.dirname(video_path)
    output_dir = os.path.join(parent_dir, f"optical_flow_output_{video_name}")
    log_path = os.path.join(output_dir, "opticalFlow_out.txt")

    os.makedirs(output_dir, exist_ok=True)

    with open(log_path, 'w') as f:
        print("Video file:", video_path, file=f)
        print("Output directory:", output_dir, file=f)
        print("Optical flow computation started at:", datetime.datetime.now(), "\n", file=f)

        compute_farneback_optical_flow(video_path, output_dir, f)

        print("Optical flow computation completed at:", datetime.datetime.now(), "\n", file=f)

if __name__ == '__main__':
    main()