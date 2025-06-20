import sys
import os
import datetime
import numpy as np
import cv2
import imageio
from tkinter import Tk, filedialog

def find_video_near_zarr(zarr_folder):
    parent = os.path.dirname(zarr_folder)
    movies_dir = os.path.join(parent, "movies")

    if not os.path.isdir(movies_dir):
        print(f"Error: 'movies/' folder not found near {zarr_folder}")
        return None

    # Open file chooser in movies folder
    print(f"Please select a video file from: {movies_dir}")
    Tk().withdraw()  # hide the root tkinter window
    video_path = filedialog.askopenfilename(
        initialdir=movies_dir,
        title="Select a video file",
        filetypes=[("AVI files", "*.avi")]
    )

    if not video_path or not os.path.isfile(video_path):
        print("No video file selected or invalid path.")
        return None

    return video_path

def compute_farneback_optical_flow(video_path, output_dir, log_file):
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()

    if not ret:
        print(f"Error: Could not read video from {video_path}", file=log_file)
        return

    prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(first_frame)
    hsv[..., 1] = 255  # full saturation for HSV

    os.makedirs(output_dir, exist_ok=True)

    frame_index = 0
    flow_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2  # direction → hue
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # magnitude → brightness
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        imageio.imwrite(os.path.join(output_dir, f"flow_{frame_index:04d}.png"), rgb_flow)
        flow_list.append(flow)

        prev = curr
        frame_index += 1

    cap.release()
    np.save(os.path.join(output_dir, "flow_raw.npy"), np.stack(flow_list))
    print(f"Saved {frame_index} flow frames and raw data to: {output_dir}", file=log_file)

def main(zarr_folder):
    if not os.path.isdir(zarr_folder):
        print(f"Error: The provided path '{zarr_folder}' is not a valid directory.")
        sys.exit(1)

    video_path = find_video_near_zarr(zarr_folder)
    if not video_path:
        sys.exit(1)

    parent_dir = os.path.dirname(zarr_folder)
    zarr_name = os.path.basename(zarr_folder).replace(".zarr", "")
    output_dir = os.path.join(parent_dir, f"{zarr_name}_optical_flow_output")
    log_path = os.path.join(parent_dir, f"{zarr_name}_opticalFlow_out.txt")

    with open(log_path, 'w') as f:
        print('Zarr folder:', zarr_folder, file=f)
        print('Video file:', video_path, file=f)
        print('Output directory:', output_dir, file=f)
        print('Optical flow calculation started at', datetime.datetime.now(), '\n', file=f)

        compute_farneback_optical_flow(video_path, output_dir, f)

        print('Optical flow calculation completed at', datetime.datetime.now(), '\n', file=f)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 opticalFlow.py <zarr_folder>")
        sys.exit(1)

    zarr_folder = sys.argv[1]
    main(zarr_folder)