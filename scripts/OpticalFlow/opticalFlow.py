import os
import sys
import re
import cv2
import numpy as np
import imageio
import datetime

# function to compute farneback optical flow for a given video
def compute_farneback_optical_flow(video_path, output_dir, log_file):
    cap = cv2.VideoCapture(video_path)  # open the video file
    ret, first_frame = cap.read()  # read the first frame

    if not ret:  # check if the video was successfully read
        print(f"Error: Could not read video from {video_path}", file=log_file)
        return

    prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)  # convert the first frame to grayscale
    hsv = np.zeros_like(first_frame)  # initialize hsv image
    hsv[..., 1] = 255  # set saturation to maximum

    frame_index = 0  # initialize frame index
    flow_list = []  # list to store optical flow data

    while True:
        ret, frame = cap.read()  # read the next frame
        if not ret:  # break if no more frames
            break

        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert current frame to grayscale

        # compute optical flow using farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev=prev, next=curr, flow=None,
            pyr_scale=0.5, levels=3, winsize=10,
            iterations=5, poly_n=7, poly_sigma=1.5, flags=0
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # calculate magnitude and angle
        hsv[..., 0] = ang * 180 / np.pi / 2  # set hue based on angle
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # normalize magnitude

        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # convert hsv to rgb
        imageio.imwrite(os.path.join(output_dir, f"flow_{frame_index:04d}.png"), rgb_flow)  # save flow image

        flow_list.append(flow)  # append flow data to list
        prev = curr  # update previous frame
        frame_index += 1  # increment frame index

    cap.release()  # release video capture
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

    # create a key overlay
    key_height = 100
    key_width = 300
    key = np.zeros((key_height, key_width, 3), dtype=np.uint8)
    for i in range(key_width):
        hue = (i / key_width) * 180  # map width to hue
        key[:, i, 0] = hue  # set hue
        key[:, i, 1] = 255  # set saturation
        key[:, i, 2] = 255  # set value
    key = cv2.cvtColor(key, cv2.COLOR_HSV2BGR)  # convert HSV to BGR

    # add text to the key
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(key, "Direction (Hue)", (10, 30), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(key, "Magnitude (Brightness)", (10, 70), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

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
        print("Usage: python AVIOpticalFlow.py <path_to_avi_file>")
        sys.exit(1)

    video_path = sys.argv[1]  # get the video path from command-line arguments
    if not os.path.isfile(video_path):  # check if the video file exists
        print(f"Error: File not found or invalid path: {video_path}")
        sys.exit(1)

    #hardcoded path if needed 
    #output_dir = "/groups/sgro/sgrolab/Ankit/Data/optical_flow_output"
    #os.makedirs(output_dir, exist_ok=True)

    output_dir = os.path.join(os.path.dirname(video_path), "optical_flow_output")  # set output directory
    os.makedirs(output_dir, exist_ok=True)  # create output directory if it doesn't exist

    log_path = os.path.join(output_dir, "opticalFlow_out.txt")  # set log file path

    with open(log_path, 'w') as f:  # open log file for writing
        print("Video file:", video_path, file=f)  # log video file path
        print("Output directory:", output_dir, file=f)  # log output directory
        print("Optical flow calculation started at", datetime.datetime.now(), "\n", file=f)  # log start time

        compute_farneback_optical_flow(video_path, output_dir, f)  # compute optical flow

        print("Optical flow calculation completed at", datetime.datetime.now(), "\n", file=f)  # log completion time
        print("Now generating movie...", file=f)  # log movie generation start

    make_movie(output_dir)  # generate movie from flow images

# entry point of the script
if __name__ == "__main__":
    main()