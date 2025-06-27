import os
import sys
import cv2
import numpy as np
import imageio
import datetime
import zarr

# function to compute farneback optical flow
def compute_farneback_optical_flow(zarr_path, cropID, output_dir, log_file):
    parent_dir = os.path.dirname(zarr_path)
    maxProjectionsRoot = zarr.open(os.path.join(parent_dir, 'analysis', 'max_projections_' + cropID), mode='r+')   
    
    maxZ = maxProjectionsRoot['maxz']

    num_frames = maxZ.shape[0]
    height = maxZ.shape[3]
    width = maxZ.shape[4]

    hsv = np.zeros((height, width, 3), dtype=np.uint8)  # initialize hsv image
    hsv[..., 1] = 255  # set saturation to maximum
    flow_list = []  # list to store optical flow data

    def enhance_cell_contrast(frame):
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(frame)
    
    prev_frame_raw = maxZ[0,0,0,:,:]
    prev_frame = cv2.normalize(prev_frame_raw, None, 0, 255, cv2.NORM_MINMAX)
    prev_frame = prev_frame.astype(np.uint8)
    prev_frame = enhance_cell_contrast(prev_frame)

    prev_frame = cv2.bilateralFilter(prev_frame, d=5, sigmaColor=35, sigmaSpace=5)

    prev_flow = None

    for frame_index in range (1, num_frames):
        curr_frame_raw = maxZ[frame_index, 0, 0, :, :]
        curr_frame = cv2.normalize(curr_frame_raw, None, 0, 255, cv2.NORM_MINMAX)
        curr_frame = curr_frame.astype(np.uint8)
        curr_frame = enhance_cell_contrast(curr_frame)

        curr_frame = cv2.bilateralFilter(curr_frame, d=5, sigmaColor=35, sigmaSpace=5)

        if frame_index > num_frames * 0.6:  # in the last 40% of frames
            # use parameters optimized for larger movements
            flow = cv2.calcOpticalFlowFarneback(
                prev=prev_frame, next=curr_frame, flow=None,
                pyr_scale=0.5, levels=7,      # Fewer levels to focus on larger structures
                winsize=15,                   # Larger window for capturing group movements
                iterations=10,                # More iterations for accuracy
                poly_n=7,                     # Larger neighborhood for group behavior
                poly_sigma=1.5,               # Higher sigma for smoother group flow
                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
            )
        else:
            # standard parameters for early frames
            flow = cv2.calcOpticalFlowFarneback(
                prev=prev_frame, next=curr_frame, flow=None,
                pyr_scale=0.5, levels=10, winsize=7,
                iterations=8, poly_n=5, poly_sigma=1.1, 
                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
            )

        # adds temporal smoothing to help reduce flickering between frames
        if prev_flow is not None:
            alpha = 0.7  # weight for current flow (0.7 current + 0.3 previous)
            flow = alpha * flow + (1 - alpha) * prev_flow

        # apply targeted filtering for problem areas
        flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        questionable_flow = flow_magnitude > np.percentile(flow_magnitude, 95)  # Top 5% strongest flows

        # apply spatial median filtering only to areas with very strong flow
        # this helps correct outlier directions in cells with problematic motion
        if np.any(questionable_flow):
            flow_x_fixed = flow[..., 0].copy()
            flow_y_fixed = flow[..., 1].copy()

            # create a dilated mask to include surrounding areas
            kernel = np.ones((5, 5), np.uint8)
            expanded_mask = cv2.dilate(questionable_flow.astype(np.uint8), kernel)
            
            # apply median filter only to those areas
            flow_x_median = cv2.medianBlur((flow[..., 0] * expanded_mask).astype(np.float32), 5)
            flow_y_median = cv2.medianBlur((flow[..., 1] * expanded_mask).astype(np.float32), 5)
            
            # replace values in the expanded mask area
            flow_x_fixed[expanded_mask > 0] = flow_x_median[expanded_mask > 0]
            flow_y_fixed[expanded_mask > 0] = flow_y_median[expanded_mask > 0]
            
            # reconstruct flow
            flow = np.dstack((flow_x_fixed, flow_y_fixed))
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # calculate magnitude and angle
        hsv[..., 0] = ang * 180 / np.pi / 2  # set hue based on angle
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # normalize magnitude

        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # convert hsv to bgr
        imageio.imwrite(os.path.join(output_dir, f"flow_{frame_index:04d}.png"), rgb_flow)  # save flow image

        flow_list.append(flow)  # append flow data to list
        prev_frame = curr_frame  # update previous frame

        prev_flow = flow.copy() 

    np.save(os.path.join(output_dir, "flow_raw.npy"), np.stack(flow_list))  # save raw flow data
    print(f"Saved {frame_index} flow frames and raw data to: {output_dir}", file=log_file)

# function to create a movie from optical flow images
def make_movie(output_dir, output_filename="optical_flow_movie.mp4", fps=10):
    frames = sorted([f for f in os.listdir(output_dir) if f.startswith("flow_") and f.endswith(".png")])
    
    if not frames:
        print(f"no flow png frames found in: {output_dir}")
        return

    first_frame_path = os.path.join(output_dir, frames[0])
    frame = cv2.imread(first_frame_path)
    if frame is None:
        print(f"error reading first frame: {first_frame_path}")
        return

    height, width, _ = frame.shape
    output_path = os.path.join(output_dir, output_filename)
    
    # use h.264 codec for better compression and compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for fname in frames:
        img = cv2.imread(os.path.join(output_dir, fname))
        if img is not None:
            writer.write(img)
        else:
            print(f"warning: skipping unreadable frame {fname}")

    writer.release()
    print(f"movie saved to: {output_path}")

# main function
def main():
    if len(sys.argv) < 2:
        print("usage: python opticalFlow.py <path_to_zarr> [cropID]")
        sys.exit(1)

    zarr_path = sys.argv[1]
    if not os.path.exists(zarr_path):
        print(f"error: path not found: {zarr_path}")
        sys.exit(1)
    
    cropID = sys.argv[2] if len(sys.argv) > 2 else ""

    output_dir = os.path.join(os.path.dirname(zarr_path), "optical_flow_output")
    os.makedirs(output_dir, exist_ok=True)
    
    log_path = os.path.join(output_dir, "opticalFlow_out.txt")

    with open(log_path, 'w') as f:
        print("zarr path:", zarr_path, file=f)
        print("crop id:", cropID, file=f)
        print("output directory:", output_dir, file=f)
        print("optical flow calculation started at", datetime.datetime.now(), "\n", file=f)

        compute_farneback_optical_flow(zarr_path, cropID, output_dir, f)

        print("optical flow calculation completed at", datetime.datetime.now(), "\n", file=f)
        print("generating movie...", file=f)

    make_movie(output_dir)

# entry point
if __name__ == "__main__":
    main()