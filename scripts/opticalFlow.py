import cv2
import numpy as np
import os
import argparse
import imageio

def compute_farneback_optical_flow(video_path, output_dir):
    # opens the video file
    cap = cv2.VideoCapture(video_path)

    # reads the first frame
    ret, first_frame = cap.read()

    # converts to grayscale
    prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # creates HSV image for flow visualization
    hsv = np.zeros_like(first_frame)
    hsv[..., 1] = 255  # saturation to max

    # prepares output directory
    os.makedirs(output_dir, exist_ok=True)

    frame_index = 0
    flow_list = []  # to store raw flow arrays for better analysis if needed 


    while True:
        # reads the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # converts current frame to grayscale
        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculates dense optical flow using Farnebäck’s algorithm
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2 , 0) #parameter values might have to be changed

        # converts flow from Cartesian to polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # uses angle (direction) as hue, magnitude (speed) as value
        hsv[..., 0] = ang * 180 / np.pi / 2      # Hue: flow direction
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value: flow speed

        # converts HSV to RGB for display
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # saves the result
        output_path = os.path.join(output_dir, f"flow_{frame_index:04d}.png")
        imageio.imwrite(output_path, rgb_flow)
        
        flow_list.append(flow)  # stores raw flow for curr frame

        # updates previous frame
        prev = curr
        frame_index += 1

    cap.release()
    
    # saves all raw flow arrays as one numpy file
    flow_stack = np.stack(flow_list)
    npy_path = os.path.join(output_dir, "flow_raw.npy")
    np.save(npy_path, flow_stack)

    print(f"Saved {frame_index} flow visualizations to: {output_dir}")
    print(f"Saved raw flow array (.npy) to: {npy_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optical Flow")
    parser.add_argument('--input', type=str, required=True, help="Input")
    parser.add_argument('--output', type=str, required=True, help="Output")
    args = parser.parse_args()

    compute_farneback_optical_flow(args.input, args.output)