import sys
import os
import datetime
import numpy as np
import cv2
import imageio
from tkinter import Tk, filedialog

def compute_farneback_optical_flow(video_path, output_dir, log_file):
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()

    if not ret:
        print(f"Error: Could not read video from {video_path}", file=log_file)
        return

    prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(first_frame)
    hsv[..., 1] = 255

    os.makedirs(output_dir, exist_ok=True)

    frame_index = 0
    flow_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        output_path = os.path.join(output_dir, f"flow_{frame_index:04d}.png")
        imageio.imwrite(output_path, rgb_flow)
        flow_list.append(flow)

        prev = curr
        frame_index += 1

    cap.release()
    flow_stack = np.stack(flow_list)
    np.save(os.path.join(output_dir, "flow_raw.npy"), flow_stack)

    print(f"Saved {frame_index} flow frames to {output_dir}", file=log_file)
    print(f"Saved raw flow data to {os.path.join(output_dir, 'flow_raw.npy')}", file=log_file)

def main(videoFile=None):
    if videoFile is None:
        Tk().withdraw()
        videoFile = filedialog.askopenfilename(
            title='Select a video file',
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if not videoFile or not os.path.isfile(videoFile):
            print(f"Error: The selected path '{videoFile}' is not a valid file.")
            sys.exit(1)

    print("Video File:", videoFile)

    parentDir = os.path.dirname(videoFile)
    os.chdir(parentDir)

    outputDir = os.path.join(parentDir, 'optical_flow_output')
    logPath = os.path.join(parentDir, 'opticalFlow_out.txt')

    with open(logPath, 'w') as f:
        print('Video file:', videoFile, '\n', file=f)
        print('Output directory:', outputDir, '\n', file=f)
        print('Optical flow calculation started at', datetime.datetime.now(), '\n', file=f)

        compute_farneback_optical_flow(videoFile, outputDir, f)

        print('Optical flow calculation completed at', datetime.datetime.now(), '\n', file=f)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        videoFile = sys.argv[1]
        if not os.path.isfile(videoFile):
            print(f"Error: The provided path '{videoFile}' is not a valid file.")
            sys.exit(1)
    else:
        videoFile = None

    main(videoFile)