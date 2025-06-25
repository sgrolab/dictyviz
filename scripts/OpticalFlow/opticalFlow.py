import os
import sys
import re
import cv2
import numpy as np
import imageio
import datetime
import zarr

# function to compute farneback optical flow for a given video
def compute_farneback_optical_flow(zarr_path, cropID, output_dir, log_file):

    parent_dir = os.path.dirname(zarr_path)

    maxProjectionsRoot = zarr.open(parent_dir + '/analysis/max_projections' + cropID, mode='r+')
    maxZ = maxProjectionsRoot['maxZ']

    num_frames, num_channels, height, width, = maxZ.shape 

    hsv = np.zeros((height, width, 3), dtype=np.uint8)  # initialize hsv image
    hsv[..., 1] = 255  # set saturation to maximum
    flow_list = []  # list to store optical flow data
    
    prev_frame = maxZ[0,0,0,:,:]
    prev_frame = prev_frame.astype(np.uint8)

    for frame_index in range (1, num_frames):
        curr_frame = maxZ[frame_index, 0, 0, :, :]
        curr_frame = curr_frame.astype(np.uint8)

        # compute optical flow using farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev=prev_frame, next=curr_frame, flow=None,
            pyr_scale=0.5, levels=3, winsize=10,
            iterations=5, poly_n=7, poly_sigma=1.5, flags=0
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # calculate magnitude and angle
        hsv[..., 0] = ang * 180 / np.pi / 2  # set hue based on angle
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # normalize magnitude

        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # convert hsv to rgb
        imageio.imwrite(os.path.join(output_dir, f"flow_{frame_index:04d}.png"), rgb_flow)  # save flow image

        flow_list.append(flow)  # append flow data to list
        prev_frame = curr_frame  # update previous frame

    np.save(os.path.join(output_dir, "flow_raw.npy"), np.stack(flow_list))  # save raw flow data
    print(f"Saved {frame_index} flow frames and raw data to: {output_dir}", file=log_file)

# function to create a movie from optical flow images
def make_movie(output_dir, output_filename="optical_flow_movie.avi", fps=10):
    frames = sorted([f for f in os.listdir(output_dir) if f.startswith("flow_") and f.endswith(".png")])  # get sorted list of flow images
    
    if not frames:  # check if there are any flow images
        print(f"No flow PNG frames found in: {output_dir}")
        return

    first_frame_path = os.path.join(output_dir, frames[0])  # get path to the first frame
    frame = cv2.imread(first_frame_path)  # read the first frame
    if frame is None:  # check if the first frame was successfully read
        print(f"Error reading first frame: {first_frame_path}")
        return

    height, width, _ = frame.shape  # get dimensions of the frame
    output_path = os.path.join(output_dir, output_filename)  # set output movie path
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # set codec for the movie
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # initialize video writer

    for fname in frames:  # iterate through all flow images
        img = cv2.imread(os.path.join(output_dir, fname))  # read the image
        if img is not None:  # check if the image was successfully read
            writer.write(img)  # write the image to the movie
        else:
            print(f"Warning: Skipping unreadable frame {fname}")  # log warning for unreadable frames

    writer.release()  # release video writer
    print(f"Movie saved to: {output_path}")

# main function to handle command-line arguments and execute the workflow
def main():
    if len(sys.argv) < 2:  # check if the video path argument is provided
        print("Usage: python opticalFlow.py")
        sys.exit(1)

    zarr_path = sys.argv[1]  # get the video path from command-line arguments
    if not os.path.exists(zarr_path):  # check if the path exists
        print(f"Error: Path not found or invalid path: {zarr_path}")
        sys.exit(1)
    
    # use cropID if provided, otherwise empty string
    cropID = sys.argv[2] if len(sys.argv) > 2 else ""

    output_dir = os.path.join(os.path.dirname(zarr_path), "optical_flow_output")
    os.makedirs(output_dir, exist_ok=True)
    
    log_path = os.path.join(output_dir, "opticalFlow_out.txt")  # set log file path

    with open(log_path, 'w') as f:  # open log file for writing
        print("Zarr path:", zarr_path, file=f)  # log zarr file path
        print("Crop ID:", cropID, file=f)  # log crop ID
        print("Output directory:", output_dir, file=f)  # log output directory
        print("Optical flow calculation started at", datetime.datetime.now(), "\n", file=f)  # log start time

        compute_farneback_optical_flow(zarr_path, cropID, output_dir, f)  # compute optical flow

        print("Optical flow calculation completed at", datetime.datetime.now(), "\n", file=f)  # log completion time
        print("Now generating movie...", file=f)  # log movie generation start

    make_movie(output_dir)  # generate movie from flow images

# entry point of the script
if __name__ == "__main__":
    main()