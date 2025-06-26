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
    try:
        parent_dir = os.path.dirname(zarr_path)
        max_proj_path = os.path.join(parent_dir, 'analysis', 'max_projections_' + cropID)
        
        try:
            maxProjectionsRoot = zarr.open(max_proj_path, mode='r')
            # Try lowercase key first
            try:
                maxZ = maxProjectionsRoot['maxz']
            except KeyError:
                # Try uppercase key next
                maxZ = maxProjectionsRoot['maxZ']
        except Exception as e:
            print(f"Error opening zarr: {str(e)}", file=log_file)
            raise
            
        print(f"Dataset shape: {maxZ.shape}", file=log_file)
        
        num_frames = maxZ.shape[0]
        height = maxZ.shape[3]
        width = maxZ.shape[4]
        
        print(f"Processing {num_frames} frames of size {width}x{height}", file=log_file)

        hsv = np.zeros((height, width, 3), dtype=np.uint8)
        hsv[..., 1] = 255  # set saturation to maximum
        flow_list = []
        
        # Enhanced pre-processing for first frame
        prev_frame_raw = maxZ[0, 0, 0, :, :]
        # Normalize to full 8-bit range for better contrast
        prev_frame = cv2.normalize(prev_frame_raw, None, 0, 255, cv2.NORM_MINMAX)
        prev_frame = prev_frame.astype(np.uint8)
        
        # Apply Gaussian blur to reduce noise (optional)
        prev_frame = cv2.GaussianBlur(prev_frame, (5, 5), 0)

        for frame_index in range(1, num_frames):
            # Enhanced pre-processing for current frame
            curr_frame_raw = maxZ[frame_index, 0, 0, :, :]
            curr_frame = cv2.normalize(curr_frame_raw, None, 0, 255, cv2.NORM_MINMAX)
            curr_frame = curr_frame.astype(np.uint8)
            curr_frame = cv2.GaussianBlur(curr_frame, (5, 5), 0)
            
            # Improved Farneback parameters for biological data
            flow = cv2.calcOpticalFlowFarneback(
                prev=prev_frame, 
                next=curr_frame, 
                flow=None,
                pyr_scale=0.5,     # Pyramid scale
                levels=5,          # Pyramid levels - increased for better multi-scale analysis
                winsize=25,        # Window size - increased for more stable flow
                iterations=3,      # Iterations at each pyramid level
                poly_n=5,          # Polynomial expansion neighborhood size
                poly_sigma=1.2,    # Gaussian std for polynomial expansion
                flags=0
            )

            # Calculate flow magnitude and angle
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Apply scaling to improve visualization
            max_mag = np.max(mag)
            print(f"Frame {frame_index}: Max flow magnitude = {max_mag:.2f}", file=log_file)
            
            # Set hue based on angle
            hsv[..., 0] = ang * 180 / np.pi / 2
            
            # Adaptive scaling of magnitude for visualization
            if max_mag > 0:
                # Improve visibility with non-linear scaling
                scaled_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                # Apply gamma correction to enhance subtle movements
                gamma = 0.7
                scaled_mag = np.power(scaled_mag / 255, gamma) * 255
                hsv[..., 2] = scaled_mag.astype(np.uint8)
            else:
                hsv[..., 2] = 0  # No movement detected
            
            # Convert HSV to RGB for visualization
            rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Add key/legend to explain the visualization
            if frame_index == 1:
                # Create a key showing the color wheel
                key_size = 100
                key = np.zeros((key_size, key_size, 3), dtype=np.uint8)
                key[..., 1] = 255  # Full saturation
                key[..., 2] = 255  # Full value
                
                # Create color wheel
                y, x = np.ogrid[:key_size, :key_size]
                center = key_size // 2
                radius = center - 10
                dist = np.sqrt((x - center)**2 + (y - center)**2)
                mask = dist <= radius
                
                # Set hue based on angle
                ang_key = np.arctan2(y - center, x - center) + np.pi
                key[..., 0][mask] = ang_key[mask] * 180 / np.pi / 2
                
                # Convert to BGR
                key_bgr = cv2.cvtColor(key, cv2.COLOR_HSV2BGR)
                
                # Overlay key at bottom-right corner
                key_y_offset = rgb_flow.shape[0] - key_size - 10
                key_x_offset = rgb_flow.shape[1] - key_size - 10
                
                # Add the key to the first frame and save separately
                key_frame = rgb_flow.copy()
                key_frame[key_y_offset:key_y_offset+key_size, key_x_offset:key_x_offset+key_size] = key_bgr
                
                # Add text labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(key_frame, "Direction", (key_x_offset-70, key_y_offset+key_size//2), 
                            font, 0.5, (255, 255, 255), 1)
                cv2.putText(key_frame, "Speed", (key_x_offset+key_size//2-20, key_y_offset-10), 
                            font, 0.5, (255, 255, 255), 1)
                
                # Save the keyed first frame
                imageio.imwrite(os.path.join(output_dir, "flow_key.png"), key_frame)
            
            # Save the frame
            imageio.imwrite(os.path.join(output_dir, f"flow_{frame_index:04d}.png"), rgb_flow)
            
            # Save flow data
            flow_list.append(flow)
            
            # Update previous frame
            prev_frame = curr_frame

        # Save raw flow data
        np.save(os.path.join(output_dir, "flow_raw.npy"), np.stack(flow_list))
        print(f"Saved {len(flow_list)} flow frames and raw data to: {output_dir}", file=log_file)
        
    except Exception as e:
        print(f"Error in optical flow computation: {str(e)}", file=log_file)
        import traceback
        traceback.print_exc(file=log_file)
        raise

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