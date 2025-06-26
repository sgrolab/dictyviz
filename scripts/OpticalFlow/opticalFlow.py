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
    max_proj_path = os.path.join(parent_dir, 'analysis', 'max_projections_' + cropID)
    
    # open zarr dataset directly with 'maxz' key
    maxProjectionsRoot = zarr.open(max_proj_path, mode='r')
    maxZ = maxProjectionsRoot['maxz']
    
    print(f"dataset shape: {maxZ.shape}", file=log_file)
    
    num_frames = maxZ.shape[0]
    height = maxZ.shape[3]
    width = maxZ.shape[4]
    
    print(f"processing {num_frames} frames of size {width}x{height}", file=log_file)

    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[..., 1] = 255  # set saturation to maximum
    flow_list = []
    
    # process first frame - properly handle 16-bit data
    prev_frame_raw = maxZ[0, 0, 0, :, :]
    
    # check data type and convert accordingly
    data_max = np.max(prev_frame_raw)
    if data_max > 255 or prev_frame_raw.dtype == np.uint16:
        print(f"detected 16-bit data (max value: {data_max})", file=log_file)
        # convert 16-bit to 8-bit properly
        prev_frame = cv2.normalize(prev_frame_raw, None, 0, 255, cv2.NORM_MINMAX)
    else:
        # already 8-bit or similar range
        prev_frame = prev_frame_raw
        
    prev_frame = prev_frame.astype(np.uint8)
    prev_frame = cv2.GaussianBlur(prev_frame, (3, 3), 0)  # lighter blur

    for frame_index in range(1, num_frames):
        # handle current frame - properly convert bit depth
        curr_frame_raw = maxZ[frame_index, 0, 0, :, :]
        
        # convert to 8-bit properly
        if data_max > 255 or curr_frame_raw.dtype == np.uint16:
            curr_frame = cv2.normalize(curr_frame_raw, None, 0, 255, cv2.NORM_MINMAX)
        else:
            curr_frame = curr_frame_raw
            
        curr_frame = curr_frame.astype(np.uint8)
        curr_frame = cv2.GaussianBlur(curr_frame, (3, 3), 0)  # lighter blur
        
        # calculate optical flow with parameters optimized for biological data
        flow = cv2.calcOpticalFlowFarneback(
            prev=prev_frame, 
            next=curr_frame, 
            flow=None,
            pyr_scale=0.5,     # pyramid scale
            levels=6,          # pyramid levels - higher for more detail
            winsize=15,        # window size - balanced for biological motion
            iterations=3,      # iterations at each pyramid level
            poly_n=5,          # polynomial expansion neighborhood size
            poly_sigma=1.2,    # gaussian std for polynomial expansion
            flags=0
        )

        # calculate flow magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # log max magnitude
        max_mag = np.max(mag)
        print(f"frame {frame_index}: max flow magnitude = {max_mag:.2f}", file=log_file)
        
        # set hue based on angle
        hsv[..., 0] = ang * 180 / np.pi / 2
        
        # scale magnitude for visualization
        if max_mag > 0:
            # normalize magnitude
            scaled_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            # apply gamma correction for better visualization
            gamma = 0.6  # lower gamma = more visible subtle movements
            scaled_mag = np.power(scaled_mag / 255, gamma) * 255
            hsv[..., 2] = scaled_mag.astype(np.uint8)
        else:
            hsv[..., 2] = 0  # no movement detected
        
        # convert hsv to bgr
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # create color wheel legend on first frame
        if frame_index == 1:
            key_size = 100
            key = np.zeros((key_size, key_size, 3), dtype=np.uint8)
            key[..., 1] = 255  # full saturation
            key[..., 2] = 255  # full value
            
            # create color wheel
            y, x = np.ogrid[:key_size, :key_size]
            center = key_size // 2
            radius = center - 10
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            mask = dist <= radius
            
            # set hue based on angle
            ang_key = np.arctan2(y - center, x - center) + np.pi
            key[..., 0][mask] = ang_key[mask] * 180 / np.pi / 2
            
            # convert to bgr
            key_bgr = cv2.cvtColor(key, cv2.COLOR_HSV2BGR)
            
            # position at bottom-right corner
            key_y_offset = rgb_flow.shape[0] - key_size - 10
            key_x_offset = rgb_flow.shape[1] - key_size - 10
            
            # add key to frame
            key_frame = rgb_flow.copy()
            key_frame[key_y_offset:key_y_offset+key_size, key_x_offset:key_x_offset+key_size] = key_bgr
            
            # add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(key_frame, "direction", (key_x_offset-70, key_y_offset+key_size//2), 
                        font, 0.5, (255, 255, 255), 1)
            cv2.putText(key_frame, "speed", (key_x_offset+key_size//2-20, key_y_offset-10), 
                        font, 0.5, (255, 255, 255), 1)
            
            # save frame with key
            imageio.imwrite(os.path.join(output_dir, "flow_key.png"), key_frame)
        
        # save the frame
        imageio.imwrite(os.path.join(output_dir, f"flow_{frame_index:04d}.png"), rgb_flow)
        
        # save flow data
        flow_list.append(flow)
        
        # update previous frame
        prev_frame = curr_frame

    # save raw flow data
    np.save(os.path.join(output_dir, "flow_raw.npy"), np.stack(flow_list))
    print(f"saved {len(flow_list)} flow frames and raw data to: {output_dir}", file=log_file)

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