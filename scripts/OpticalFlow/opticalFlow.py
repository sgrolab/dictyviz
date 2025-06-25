import sys
import os
import datetime
import numpy as np
import cv2
import imageio
import re
from tkinter import Tk, filedialog

# prompts user to select a nearby video file from the 'movies' folder relative to the zarr folder
def find_movie(zarr_folder):
    parent = os.path.dirname(zarr_folder)
    movies_dir = os.path.join(parent, "movies")

    if not os.path.isdir(movies_dir):
        print(f"Error: 'movies/' folder not found near {zarr_folder}")
        return None

    print(f"Please select a video file from: {movies_dir}")
    Tk().withdraw()  # hide the default tkinter window
    video_path = filedialog.askopenfilename(
        initialdir=movies_dir,
        title="Select a video file",
        filetypes=[("AVI files", "*.avi")]
    )

    if not video_path or not os.path.isfile(video_path):
        print("No video file selected or invalid path.")
        return None

    return video_path

# computes dense optical flow using Farneback method and saves visualizations + raw flow data
def compute_farneback_optical_flow(video_path, output_dir, log_file):
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()

    if not ret:
        print(f"Error: Could not read video from {video_path}", file=log_file)
        return

    prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)  # converts the first frame from color to grayscale since optical flow works better on single-channel intensity images (only cares about change in brightness not color)
    hsv = np.zeros_like(first_frame) # creates an empty image with the same shape as the first frame  (will later hold the flow visiualization)
    hsv[..., 1] = 255  # set saturation to max for coloring

    frame_index = 0
    flow_list = [] # stores optical flow arrays 

    while True: # loops through all frames 
        ret, frame = cap.read()
        if not ret:
            break

        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts current frame to grayscale for conersion 

        # calculates dense optical flow between prev and current frames
        flow = cv2.calcOpticalFlowFarneback(
            prev=prev, next=curr, flow=None,
            pyr_scale=0.5, levels=3, winsize=10,
            iterations=5, poly_n=7, poly_sigma=1.5, flags=0
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2  # angle → color 
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # magnitude → brightness
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # convert to RGB image

        # saves RGB visualization of flow
        imageio.v2.imwrite(os.path.join(output_dir, f"flow_{frame_index:04d}.png"), rgb_flow)
        flow_list.append(flow)

        prev = curr
        frame_index += 1 # update to the prev frame to the next for iteration and increments frame index 

    cap.release() # closes video file 

    # saves all raw flow arrays as a .npy file
    np.save(os.path.join(output_dir, "flow_raw.npy"), np.stack(flow_list))
    print(f"Saved {frame_index} flow frames and raw data to: {output_dir}", file=log_file)

# get list of all PNG frames in output_dir and create AVI video
def make_movie(output_dir, output_filename="optical_flow_movie.avi", fps=10):
    frames = sorted([f for f in os.listdir(output_dir) if f.startswith("flow_") and f.endswith(".png")])
    
    if not frames:
        print(f"No flow PNG frames found in: {output_dir}")
        return

    # read the first frame to get video dimensions
    first_frame_path = os.path.join(output_dir, frames[0])
    frame = cv2.imread(first_frame_path)
    height, width, _ = frame.shape

    # set up video writer
    output_path = os.path.join(output_dir, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"Error: Failed to open video writer at {output_path}")
        return

    for fname in frames:
        img = cv2.imread(os.path.join(output_dir, fname))
        writer.write(img)

    writer.release()
    print(f"Movie saved to: {output_path}")


def main(zarr_folder):
    if not os.path.isdir(zarr_folder):
        print(f"Error: The provided path '{zarr_folder}' is not a valid directory.")
        sys.exit(1)

    video_path = find_movie(zarr_folder)
    if not video_path:
        sys.exit(1)

    parent_dir = os.path.dirname(zarr_folder)
    print(f"[DEBUG] Parent directory: {parent_dir}")

    zarr_name = os.path.basename(zarr_folder).replace(".zarr", "")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    output_dir = os.path.join(parent_dir, f"optical_flow_output_{video_name}")
    print(f"[DEBUG] output directory: {output_dir}")

    log_path = os.path.join(output_dir, "opticalFlow_out.txt")

    os.makedirs(output_dir, exist_ok=True)

    # writes logs and run optical flow computation
    with open(log_path, 'w') as f:
        print('Zarr folder:', zarr_folder, file=f)
        print('Video file:', video_path, file=f)
        print('Output directory:', output_dir, file=f)
        print('Optical flow calculation started at', datetime.datetime.now(), '\n', file=f)

        compute_farneback_optical_flow(video_path, output_dir, f)

        print('Optical flow calculation completed at', datetime.datetime.now(), '\n', file=f)
        print('Now generating movie...', file=f)

    make_movie(output_dir)

# entry point that expects the zarr folder path as argument
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 opticalFlow_full_pipeline.py <zarr_folder>")
        sys.exit(1)

    zarr_folder = sys.argv[1]
    main(zarr_folder)