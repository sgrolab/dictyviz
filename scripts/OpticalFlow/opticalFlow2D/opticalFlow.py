import os
import sys
import cv2
import numpy as np
import imageio
import datetime
import zarr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def enhance_cell_contrast(frame):
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances local contrast
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(frame)

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

        flow = cv2.calcOpticalFlowFarneback(
            prev=prev_frame, next=curr_frame, flow=None,
            
            # pyr_scale: Image pyramid scaling factor (0.1 to 0.99)
            # - Higher values (0.7-0.9): Faster computation, captures large motions better, less detail
            # - Lower values (0.3-0.5): Slower computation, captures fine details better, may miss large motions
            # - 0.5 means each pyramid level is half the size of the previous level
            pyr_scale=0.2,
            
            # levels: Number of pyramid levels (1-20+)
            # - More levels: Better handling of large motions, slower computation
            # - Fewer levels: Faster computation, may miss large displacements
            # - Each level processes the image at different resolutions
            levels=3,
            
            # winsize: Averaging window size in pixels (odd numbers: 5, 7, 9, 11, 15, etc.)
            # - Larger windows (11-15): More robust to noise, smoother flow, less spatial detail
            # - Smaller windows (5-7): More spatial detail, more sensitive to noise
            # - Must be odd number, represents neighborhood size for flow calculation
            winsize=3,
            
            # iterations: Number of iterations at each pyramid level (3-15)
            # - More iterations: More accurate flow estimation, slower computation
            # - Fewer iterations: Faster computation, potentially less accurate
            # - Algorithm refines flow estimate this many times per level
            iterations=8,
            
            # poly_n: Size of pixel neighborhood for polynomial expansion (3, 5, or 7)
            # - 3: Fastest, least accurate, good for small motions
            # - 5: Balanced speed/accuracy (most common choice)
            # - 7: Slowest, most accurate, good for complex motions
            # - Determines complexity of motion model fitted to each pixel
            poly_n=7,
            
            # poly_sigma: Standard deviation of Gaussian used to smooth derivatives (0.8-2.0)
            # - Lower values (0.8-1.0): Preserves sharp edges, more noise sensitive
            # - Higher values (1.2-1.5): Smoother derivatives, more noise robust
            # - Controls smoothing applied before polynomial fitting
            poly_sigma=1.5,
            
            # flags: Algorithm behavior flags
            # - OPTFLOW_FARNEBACK_GAUSSIAN: Use Gaussian weighting (recommended)
            # - Can combine with other flags using bitwise OR (|)
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
        hsv[..., 2] = np.clip(mag * (255/15), 0, 255).astype(np.uint8) # takes raw magnitude values and will scale anything above a magntitude of 15 to a brightness of 255

        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # convert hsv to bgr

        # Create histogram visualization
        #hist_image = create_flow_histogram(mag, width, height)
        #hist_h, hist_w = hist_image.shape[:2]

        # Position histogram in top-left corner with padding
        #hist_pad = 10
        #hist_pos_x = hist_pad
        #hist_pos_y = hist_pad

        # Create a copy of the flow visualization
        final_frame = rgb_flow.copy()

        # Add histogram to the frame
        #final_frame[hist_pos_y:hist_pos_y+hist_h, hist_pos_x:hist_pos_x+hist_w] = hist_image

        # create and add the legend to each frame 
        legend = create_flow_color_wheel(width, height)
        legend_h, legend_w = legend.shape[:2]

        # position in bottom right with padding
        pad = 10
        pos_x = width - legend_w - pad
        pos_y = height - legend_h - pad

        # create a copy of the flow visualization and overlay the legend
        final_frame[pos_y:pos_y+legend_h, pos_x:pos_x+legend_w] = legend

        cv2.imwrite(os.path.join(output_dir, f"flow_{frame_index:04d}.png"), final_frame)  # save flow image

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

def create_flow_color_wheel(width, height):
    # make a square legend with padding
    size = min(width, height)
    legend_size = int(size * 0.15) # Legend size as 15% of frame dimension
    min_size = 100
    legend_size = max(legend_size, min_size)
    legend = np.zeros((legend_size, legend_size, 3), dtype=np.uint8) + 20  # dark grey background
    
    # calculate center and radius
    center_x, center_y = legend_size // 2, legend_size // 2
    max_radius = (legend_size // 2) - 2 
    
    # create the color wheel
    for y in range(legend_size):
        for x in range(legend_size):
            # calculate distance from center
            dx, dy = x - center_x, y - center_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # skip pixels outside the circle
            if distance > max_radius:
                continue
            
            # calculate angle and normalize to 0-360 degrees
            angle = (np.degrees(np.arctan2(-dy, dx))) % 360 
            
            # normalize distance to 0-1 range for brightness
            normalized_distance = distance / max_radius
            
            # set HSV values based on angle and distance
            # OpenCV uses 0-180 for hue (represents 0-360 degrees)
            hue = angle / 2
            saturation = 255
            
            # makes the center dimmer, edges brighter
            value = int(normalized_distance * 255)
            
            # convert HSV to BGR for this pixel
            color = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0][0]
            legend[y, x] = color
    
    # add a thin border around the wheel
    cv2.circle(legend, (center_x, center_y), max_radius, (200, 200, 200), 1)

    # add magnitude labels of 0 and 15 on the circle 
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1

    text_size = cv2.getTextSize("0", font, font_scale, font_thickness)[0]
    text_x = center_x - text_size[0] // 2
    text_y = center_y + text_size[1] // 2
    cv2.putText(legend, "0", (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    text_size = cv2.getTextSize("15", font, font_scale, font_thickness)[0]
    edge_x = center_x + max_radius - text_size[0] - 2  
    edge_y = center_y
    cv2.putText(legend, "15", (edge_x, edge_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return legend

def create_flow_histogram(mag, width, height):
    fig = plt.figure(figsize=(4, 2), dpi=100)
    
    # flatten magnitude array for histogram
    flat_mag = mag.flatten()
    
    # create histogram with appropriate bins
    max_mag = np.max(flat_mag)
    if max_mag > 0:
        # Create histogram with 30 bins
        plt.hist(flat_mag, bins=30, color='cyan', edgecolor='blue', alpha=0.7)
        plt.title('Flow Magnitude Distribution', color='black', fontsize=10)
        plt.xlabel('Magnitude', color='black', fontsize=8)
        plt.ylabel('Pixel Count', color='black', fontsize=8)
        
        # Set max x value to show the full range
        plt.xlim(0, max_mag * 1.1)
        
        # Style the plot for visibility on dark background
        plt.grid(alpha=0.3)
        plt.tick_params(colors='black', which='both')
        for spine in plt.gca().spines.values():
            spine.set_edgecolor('black')
    else:
        plt.text(0.5, 0.5, 'No motion detected', 
                horizontalalignment='center', color='black')
    
    # Make background dark for better visibility
    plt.gca().set_facecolor('#303030')
    
    # Save the figure to a numpy array
    fig.canvas.draw()
    buffer = fig.canvas.buffer_rgba()
    hist_image = np.asarray(buffer)[:,:,:3] 
    
    plt.close(fig)

    return hist_image

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

    output_dir = os.path.join(os.path.dirname(zarr_path), "optical_flow_output2")
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