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

    # with underscore 
    path_with_underscore = os.path.join(parent_dir, 'analysis', 'max_projections_' + cropID)
    # without underscore
    path_without_underscore = os.path.join(parent_dir, 'analysis', 'max_projections' + cropID)
    
    # check which path exists
    if os.path.exists(path_with_underscore):
        max_proj_path = path_with_underscore
    elif os.path.exists(path_without_underscore):
        max_proj_path = path_without_underscore
    else:
        print(f"Error: Neither {path_with_underscore} nor {path_without_underscore} exists", file=log_file)
        return False
        
    maxProjectionsRoot = zarr.open(max_proj_path, mode='r+') 
    
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

        # create and add the legend to each frame
        legend = create_flow_legend(width, height)
        legend_h, legend_w = legend.shape[:2]

        # position in bottom right with padding
        pad = 10
        pos_x = width - legend_w - pad
        pos_y = height - legend_h - pad

        # create a copy of the flow visualization and overlay the legend
        final_frame = rgb_flow.copy()
        final_frame[pos_y:pos_y+legend_h, pos_x:pos_x+legend_w] = legend

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

    # saves a standalone legend image for reference
    standalone_legend = create_flow_legend(400, 300)  # Fixed size for reference image
    imageio.imwrite(os.path.join(output_dir, "flow_legend.png"), standalone_legend)

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


# Add this function after your existing functions

def create_flow_legend(width, height):
    legend_width = int(width * 0.25)  # 25% of frame width
    legend_height = int(height * 0.25)  # 25% of frame height

    # create the legend image with transparent background
    legend = np.zeros((legend_height, legend_width, 4), dtype=np.uint8)
    
    # create circular color wheel for direction
    center_x = legend_width // 3
    center_y = legend_height // 2
    radius = min(center_x, center_y) - 10
    
    # draw direction color wheel
    for y in range(legend_height):
        for x in range(center_x * 2):
            # calc distance from center
            dx = x - center_x
            dy = y - center_y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < radius:
                # Calculate angle and map to hue
                angle = np.arctan2(dy, dx)
                hue = ((angle + np.pi) / (2 * np.pi)) * 180
                
                # Create HSV color and convert to RGBA
                color = np.zeros((1, 1, 3), dtype=np.uint8)
                color[0, 0, 0] = hue
                color[0, 0, 1] = 255
                color[0, 0, 2] = 255
                bgr_color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0, 0]
                
                # Set pixel color and alpha
                legend[y, x, 0:3] = bgr_color
                legend[y, x, 3] = 255  # Fully opaque
    
    # Create magnitude gradient (brightness)
    bar_width = legend_width // 3
    for y in range(legend_height):
        brightness = int((1 - y / legend_height) * 255)  # Top is brightest
        for x in range(center_x * 2, center_x * 2 + bar_width):
            legend[y, x, 0:3] = [brightness, brightness, brightness]  # Grayscale
            legend[y, x, 3] = 255  # Fully opaque
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = legend_width / 300  # Scale font based on legend size
    
    # Direction label
    cv2.putText(legend, "Direction", (center_x-radius//2, legend_height-10), 
                font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Add direction arrows with labels
    arrow_length = radius // 2
    directions = [
        (0, "→", (center_x+arrow_length, center_y)),
        (np.pi/2, "↓", (center_x, center_y+arrow_length)),
        (np.pi, "←", (center_x-arrow_length, center_y)),
        (3*np.pi/2, "↑", (center_x, center_y-arrow_length))
    ]
    
    for angle, symbol, (end_x, end_y) in directions:
        dx = int(arrow_length * np.cos(angle))
        dy = int(arrow_length * np.sin(angle))
        cv2.putText(legend, symbol, (end_x-5, end_y+5), font, font_scale*1.5, 
                    (255, 255, 255), 1, cv2.LINE_AA)
   
    # Magnitude label
    cv2.putText(legend, "Speed", (center_x*2 + 5, 20), 
                font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(legend, "Fast", (center_x*2 + 5, 40), 
                font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(legend, "Slow", (center_x*2 + 5, legend_height-20), 
                font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Add semi-transparent background
    bg = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    alpha_bg = np.zeros((legend_height, legend_width), dtype=np.uint8) + 180
    
    # Convert to BGR for OpenCV compatibility, with proper alpha blending
    result_bgr = legend[:, :, 0:3].copy()
    
    return result_bgr


# entry point
if __name__ == "__main__":
    main()