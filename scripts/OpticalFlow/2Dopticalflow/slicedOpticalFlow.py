import os
import sys
import cv2
import torch
import numpy as np
import imageio
import datetime
import zarr
import tifffile as tiff
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def enhance_cell_contrast(frame):
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances local contrast
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(frame)

def compute_farneback_optical_flow(zarr_path, dim, cropID, output_dir, log_file):
    
    parent_dir = os.path.dirname(zarr_path)

    # with underscore 
    path_with_underscore = os.path.join(parent_dir, 'analysis', 'sliced_max_projections_' + cropID)
    # without underscore
    path_without_underscore = os.path.join(parent_dir, 'analysis', 'sliced_max_projections' + cropID)
    
    # check which path exists
    if os.path.exists(path_with_underscore):
        max_proj_path = path_with_underscore
    elif os.path.exists(path_without_underscore):
        max_proj_path = path_without_underscore
    else:
        print(f"Error: Neither {path_with_underscore} nor {path_without_underscore} exists", file=log_file)
        return False
        
    slicedMaxProjectionsRoot = zarr.open(max_proj_path, mode='r+') 
    
    slicedMax = slicedMaxProjectionsRoot[f"sliced_max{dim}"]
    slicedMax = np.array(slicedMax)
    #slicedMax = np.flip(slicedMax, axis=3)  

    num_frames = slicedMax.shape[0]
    num_slices = slicedMax.shape[2]
    height = slicedMax.shape[3] 
    width = slicedMax.shape[4]

    flow_list = np.zeros((num_frames, num_slices, height, width, 2), dtype=np.float32)  # initialize flow list

    prev_frame_raw = slicedMax[0,0,:,:,:]
    prev_frame_raw = np.concatenate(prev_frame_raw, axis=0) #concatenate slices into a single frame
    prev_frame_raw = np.flip(prev_frame_raw, axis=0)  # flip vertically to match original orientation

    prev_frame = cv2.normalize(prev_frame_raw, None, 0, 255, cv2.NORM_MINMAX)
    prev_frame = prev_frame.astype(np.uint8)
    prev_frame = enhance_cell_contrast(prev_frame)

    prev_frame = cv2.bilateralFilter(prev_frame, d=5, sigmaColor=35, sigmaSpace=5)

    prev_flow = None

    for frame_index in range (1, num_frames):

        curr_frame_raw = slicedMax[frame_index, 0, :, :, :]
        curr_frame_raw = np.concatenate(curr_frame_raw, axis=0)  # concatenate slices into a single frame
        curr_frame_raw = np.flip(curr_frame_raw, axis=0)  # flip vertically to match original orientation

        curr_frame = cv2.normalize(curr_frame_raw, None, 0, 255, cv2.NORM_MINMAX)
        curr_frame = curr_frame.astype(np.uint8)
        curr_frame = enhance_cell_contrast(curr_frame)

        curr_frame = cv2.bilateralFilter(curr_frame, d=5, sigmaColor=35, sigmaSpace=5)

        print(prev_frame.shape, curr_frame.shape)

        if frame_index > num_frames * 0.6:  # in the last 40% of frames
            # use parameters optimized for larger movements
            flow = cv2.calcOpticalFlowFarneback(
                prev=prev_frame, next=curr_frame, flow=None,
                pyr_scale=0.4, levels=7,      # Fewer levels to focus on larger structures
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

        prev_frame = curr_frame  # update previous frame

        prev_flow = flow.copy() 

        # unpack slices and append to flow list
        flow = np.split(flow, num_slices, axis=0)  # split flow into slices
        
        flow_list[frame_index] = flow 

    # after processing all slices, save the flow images and raw data
    np.save(os.path.join(output_dir, f"flow_raw.npy"), np.stack(flow_list))  # save raw flow data
    print(f"Saved {frame_index} flow frames and raw data to: {output_dir}", file=log_file)

# function to create a movie from optical flow images
def make_movie(output_dir, fps=10):

    output_filename="optical_flow_movie.mp4"
    output_path = os.path.join(output_dir, output_filename)

    # load raw flow data
    flow_raw = np.load(os.path.join(output_dir, "flow_raw.npy"))

    num_frames, num_slices, height, width, _ = flow_raw.shape

    hsv = np.zeros((height*num_slices, width, 3), dtype=np.uint8)  # initialize hsv image
    hsv[..., 1] = 255  # set saturation to maximum
  
    # use h.264 codec for better compression and compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height*num_slices))

    for frame_index in range(num_frames):
        # process each frame of optical flow
        curr_frame_flow = flow_raw[frame_index]
        curr_frame_flow = np.concatenate(curr_frame_flow, axis=0)
        curr_frame_flow = np.flip(curr_frame_flow, axis=1)  # flip vertically to match original orientation

        mag, ang = cv2.cartToPolar(curr_frame_flow[..., 0], curr_frame_flow[..., 1])  # calculate magnitude and angle
        hsv[..., 0] = ang * 180 / np.pi / 2  # set hue based on angle
        hsv[..., 2] = np.clip(mag * (255/15), 0, 255).astype(np.uint8) # takes raw magnitude values and will scale anything above a magntitude of 15 to a brightness of 255

        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # convert hsv to bgr

        # Create a copy of the flow visualization
        final_frame = rgb_flow.copy()

        # create and add the legend to each frame 
        legend = create_flow_color_wheel(width, height)
        legend_h, legend_w = legend.shape[:2]

        # position in bottom right with padding
        pad = 10
        pos_x = width - legend_w - pad
        pos_y = height - legend_h - pad

        # create a copy of the flow visualization and overlay the legend
        final_frame[pos_y:pos_y+legend_h, pos_x:pos_x+legend_w] = legend

        imageio.imwrite(os.path.join(output_dir, f"flow_{frame_index:04d}.png"), final_frame)  # save flow image

        writer.write(final_frame)  # write frame to video

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
            angle = (np.degrees(np.arctan2(-dy, dx)) + 250) % 360 
            
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

# calculate the magnitude and variance of the window 
def calculate_mag_var(vx, vy, log_file, window_size=40):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", file=log_file)
    
    # Convert to torch tensors
    vx_torch = torch.from_numpy(vx).float().to(device)
    vy_torch = torch.from_numpy(vy).float().to(device)
    
    magnitude = torch.sqrt(vx_torch**2 + vy_torch**2)
    
    # Use unfold to create sliding windows
    # unfold(dim, size, step) creates sliding windows
    vx_windows = vx_torch.unfold(2, window_size, 1).unfold(3, window_size, 1)  # shape: (frames, slices, out_h, out_w, win_h, win_w)
    vy_windows = vy_torch.unfold(2, window_size, 1).unfold(3, window_size, 1)
    mag_windows = magnitude.unfold(2, window_size, 1).unfold(3, window_size, 1)
    
    # Calculate variance and mean over the window dimensions (last 2 dims)
    var_x = torch.var(vx_windows, dim=(-2, -1))
    var_y = torch.var(vy_windows, dim=(-2, -1))
    total_variance = var_x + var_y
    mean_magnitude = torch.mean(mag_windows, dim=(-2, -1))
    
    # Move back to CPU and convert to numpy
    magnitude_map = mean_magnitude.cpu().numpy()
    variance_map = total_variance.cpu().numpy()
    
    return magnitude_map, variance_map

def find_optimal_regions(magnitude_map, variance_map, top_k=5, suppression_radius = 35):
    """
    Find optimal regions with high magnitude and low variance
    Returns list of (row, col, magnitude, variance, score) tuples
    """
    # normalize maps to 0-1 range
    norm_magnitude = (magnitude_map - magnitude_map.min()) / (magnitude_map.max() - magnitude_map.min())
    norm_variance = (variance_map - variance_map.min()) / (variance_map.max() - variance_map.min())

    # score: high magnitude, low variance
    score = norm_magnitude - norm_variance
    results = []

    used_masks = np.zeros_like(score[0], dtype=bool)

    for i in range(top_k):

        used_masks_broadcast = np.broadcast_to(used_masks, score.shape)
        masked_score = np.ma.array(score, mask=used_masks_broadcast)
        if masked_score.count() == 0:
            break
    
        # Find highest remaining score
        max_idx = np.unravel_index(masked_score.argmax(), score.shape)
        frame, slice_nb, row, col = max_idx
        results.append((frame, slice_nb, row, col, 
                        magnitude_map[frame, slice_nb, row, col], 
                        variance_map[frame, slice_nb, row, col], 
                        norm_magnitude[frame, slice_nb, row, col], 
                        norm_variance[frame, slice_nb, row, col], 
                        score[frame, slice_nb, row, col]))

        # Apply suppression mask around selected point and the two adjacent slices
        rr, cc = np.ogrid[:score.shape[2], :score.shape[3]]
        suppression_mask = (rr - row)**2 + (cc - col)**2 <= suppression_radius**2
        suppression_mask = np.broadcast_to(suppression_mask, (3, score.shape[2], score.shape[3]))
        used_masks[slice_nb-1:slice_nb+2][suppression_mask] = True
    
    return results

def save_analysis_results(optimal_regions, output_dir):
    """
    Save magnitude/variance analysis results to the frame directory
    """
    # Save optimal regions as text file
    regions_file = os.path.join(output_dir, "optimal_regions.txt")
    with open(regions_file, 'w') as f:
        f.write(f"Optimal Flow Regions Analysis\n")
        f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Top regions (frame, slice, row, col, raw_magnitude, raw_variance, norm_magnitude, norm_variance, score):\n")

        for i, (frame, slice_nb, row, col, raw_mag, raw_var, norm_magnitude, norm_variance, score) in enumerate(optimal_regions):
            f.write(f"{i+1}. Frame: {frame:3d}, Slice: {slice_nb:3d}, Row: {row:3d}, Col: {col:3d}, "
                   f"Raw_Mag: {raw_mag:.4f}, Raw_Var: {raw_var:.4f}, "
                   f"Norm_Mag: {norm_magnitude:.4f}, Norm_Var: {norm_variance:.4f}, "
                   f"Score: {score:.4f}\n")
    
    print(f"✓ Analysis results saved:")
    print(f"  - magnitude_map.npy") 
    print(f"  - variance_map.npy") 
    print(f"  - optimal_regions.txt")
    
    return regions_file

def crop_regions_from_zarr(zarr_path, dim, nb_slices, output_dir, optimal_regions, f, crop_size=256):
    """
    Crop regions from zarr based on optimal regions found
    """

    # create a directory for cropped regions
    cropped_dir = os.path.join(output_dir, "cropped_regions")
    os.makedirs(cropped_dir, exist_ok=True)

    if not os.path.exists(zarr_path):
        print(f"error: zarr path not found: {zarr_path}", file=f)
        return

    print(f"cropping regions from zarr: {zarr_path}", file=f)
    zarr_root = zarr.open(zarr_path, mode='r+')
    res_array = zarr_root[0][0]

    _, _, _, lenY, lenX = res_array.shape
    
    for i, (frame, slice_nb, row, col, *_) in enumerate(optimal_regions):
        if dim == 'x':
            slice_depth = lenY//(nb_slices-1)
            x_min = slice_nb * slice_depth - (crop_size // 2)
            x_max = slice_nb * slice_depth + (crop_size // 2)
            y_min = col - (crop_size // 2)
            y_max = col + (crop_size // 2)
        elif dim == 'y':
            slice_depth = lenX//(nb_slices-1)
            x_min = col - (crop_size // 2)
            x_max = col + (crop_size // 2)
            y_min = slice_nb * slice_depth - (crop_size // 2)
            y_max = slice_nb * slice_depth + (crop_size // 2)

        # ensure cropping indices are within bounds
        x_min = max(0, x_min)
        x_max = min(lenX, x_max)
        y_min = max(0, y_min)
        y_max = min(lenY, y_max)

        cropped_region = res_array[:, :, :, y_min:y_max, x_min:x_max]

        # save the cropped region as a TIFF file
        cropped_filename = os.path.join(cropped_dir, f"cropped_region_{dim}_{i+1}_frame{frame:03d}_row{row:03d}_col{col:03d}.ome.tif")
        tiff.imwrite(cropped_filename, cropped_region, metadata={'axes': 'TCZYX'})

        print(f"✓ Cropped region {i+1} saved to: {cropped_filename}", file=f)

# main function
def main():
    if len(sys.argv) < 2:
        print("usage: python slicedOpticalFlow.py <path_to_zarr> [cropID]")
        sys.exit(1)

    zarr_path = sys.argv[1]
    if not os.path.exists(zarr_path):
        print(f"error: path not found: {zarr_path}")
        sys.exit(1)
    
    cropID = sys.argv[2] if len(sys.argv) > 2 else ""

    

    dims = ['x', 'y']
    output_dirs = [os.path.join(os.path.dirname(zarr_path), "sliced_X_optical_flow_output"), os.path.join(os.path.dirname(zarr_path), "sliced_Y_optical_flow_output")]
    for output_dir, dim in zip(output_dirs, dims):
        os.makedirs(output_dir, exist_ok=True)

        log_path = os.path.join(output_dir, "slicedOpticalFlow_out.txt")

        with open(log_path, 'w') as f:
            print("zarr path:", zarr_path, file=f)
            print("crop id:", cropID, file=f)
            print("output directory:", output_dir, "\n", file=f)
            # check if flow_raw.npy already exists
            if os.path.exists(os.path.join(output_dir, "flow_raw.npy")):
                print("flow_raw.npy already exists in", output_dir, file=f)
                print("skipping optical flow calculation for", dim, "\n", file=f)
            else:
                print("optical flow calculation started at", datetime.datetime.now(), file=f)
                compute_farneback_optical_flow(zarr_path, dim, cropID, output_dir, f)
                print("optical flow calculation completed at", datetime.datetime.now(), "\n", file=f)

            # check if movie already exists
            if os.path.exists(os.path.join(output_dir, "optical_flow_movie.mp4")):
                print("optical flow movie already exists in", output_dir, file=f)
                print("skipping movie generation for", dim, "\n", file=f)
            else:
                print("generating movie...", file=f)
                make_movie(output_dir)
                print("movie generation completed at", datetime.datetime.now(), "\n", file=f)

            # check if magnitude/variance analysis already exists
            if os.path.exists(os.path.join(output_dir, "magnitude_map.npy")) and os.path.exists(os.path.join(output_dir, "variance_map.npy")):
                # load existing analysis results
                magnitude_map = np.load(os.path.join(output_dir, "magnitude_map.npy"))
                variance_map = np.load(os.path.join(output_dir, "variance_map.npy"))

                print("magnitude/variance analysis already exists in", output_dir, file=f)
                print("skipping analysis for", dim, "\n", file=f)
            else:
                print("performing magnitude/variance analysis...", file=f)

                # calculate magnitude and variance maps for determining optimal regions
                flow_raw = np.load(os.path.join(output_dir, "flow_raw.npy"))
                vx = flow_raw[..., 0]
                vy = flow_raw[..., 1]

                magnitude_map, variance_map = calculate_mag_var(vx, vy, f, window_size=40)

                # save the magnitude and variance maps
                np.save(os.path.join(output_dir, "magnitude_map.npy"), magnitude_map)
                np.save(os.path.join(output_dir, "variance_map.npy"), variance_map)

            print("finding optimal regions...", file=f)
            optimal_regions  = find_optimal_regions(magnitude_map, variance_map, top_k=5, suppression_radius=35)
            # Save analysis results to frame directory
            print("saving analysis results...", file=f)
            regions_file = save_analysis_results(optimal_regions, output_dir)
            print(f"✓ Analysis results saved to: {regions_file}", file=f)

            # crop regions from zarr
            print("cropping regions from zarr...", file=f)
            nb_slices = magnitude_map.shape[1]
            crop_regions_from_zarr(zarr_path, dim, nb_slices, output_dir, optimal_regions, f)
            print("cropping completed", file=f)

# entry point
if __name__ == "__main__":
    main()