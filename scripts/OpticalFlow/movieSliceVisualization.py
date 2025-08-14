import os
import sys
import numpy as np
import cv2
import matplotlib
import json
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from OpticalFlow.helpers import flowLoader
from scipy.ndimage import gaussian_filter
from OpticalFlow.optical2Dflow import opticalFlow
import glob

def getCellChannelFromJSON(jsonFile):
    with open(jsonFile) as f:
        channelSpecs = json.load(f)["channels"]
    cells_found = False
    for i, channelInfo in enumerate(channelSpecs):
        if channelInfo["name"].startswith("cells"):
            cells = i
            if cells_found:
                print(f"Warning: Multiple channels starting with 'cells' found. Multiple cell channels is not supported. Using channel {i}.")
            print(f"Found cell channel: {i}")
            cells_found = True
    if not cells_found:
        print("Error: No channel starting with 'cells' found in parameters.json.")
        return None
    return cells

def create_hsv_flow(vx, vy, max_flow=None):
    """Create HSV flow visualization"""
    # Calculate magnitude and angle
    mag, ang = cv2.cartToPolar(vx, vy)
    
    # Determine scaling factor
    if max_flow is None:
        max_flow = np.percentile(mag, 99)  # Use 99th percentile to avoid outliers

    # Create HSV image
    height, width = vx.shape
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue based on angle
    hsv[..., 1] = 255  # Set saturation to maximum
    hsv[..., 2] = np.clip(mag * (255/max_flow), 0, 255).astype(np.uint8)  # Value based on magnitude
    
    # Convert HSV to BGR
    bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return bgr_flow, mag, max_flow

def normalize_vz(vz):
    """Normalize vz with tile-based approach"""
    # Apply Gaussian smoothing to reduce noise
    vz_smoothed = gaussian_filter(vz, sigma=3)
    
    # Get approximate tile dimensions (3x4 grid = 12 tiles)
    height, width = vz_smoothed.shape
    tile_height = height // 4
    tile_width = width // 3
    
    # Create a normalized version with zero mean per tile
    vz_normalized = np.zeros_like(vz_smoothed)
    
    # Process each tile independently
    for i in range(4):  # 4 rows of tiles
        for j in range(3):  # 3 columns of tiles
            # Define tile boundaries
            y_start = i * tile_height
            y_end = min((i + 1) * tile_height, height)
            x_start = j * tile_width
            x_end = min((j + 1) * tile_width, width)
            
            # Extract tile
            tile = vz_smoothed[y_start:y_end, x_start:x_end]
            
            # Normalize tile to have zero mean
            tile_mean = np.mean(tile)
            
            if np.abs(tile_mean) > 1e-6:  # Avoid division by zero or normalization of uniform tiles
                # Normalize to have zero mean
                normalized_tile = (tile - tile_mean)
            else:
                normalized_tile = np.zeros_like(tile)
            
            # Store normalized tile
            vz_normalized[y_start:y_end, x_start:x_end] = normalized_tile
    
    # Apply a small blur at tile boundaries to reduce edge artifacts
    vz_normalized = gaussian_filter(vz_normalized, sigma=1)
    
    return vz_normalized

def create_xy_flow_frame(vx, vy, raw_slice, max_flow=None, arrow_step=10):
    """Create a frame with raw image and XY flow visualization"""
    
    # Set up figure with two panels side by side
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])
    
    # Raw data panel
    ax_raw = fig.add_subplot(gs[0, 0])
    if raw_slice is not None:
        # Normalize raw data for display
        vmin = np.percentile(raw_slice, 5)
        vmax = np.percentile(raw_slice, 95)
        raw_display = np.clip(raw_slice, vmin, vmax)
        raw_display = (raw_display - vmin) / (vmax - vmin)
        ax_raw.imshow(raw_display, cmap='gray', origin='lower')
    else:
        ax_raw.text(0.5, 0.5, 'No Raw Data\nAvailable', ha='center', va='center', 
                  transform=ax_raw.transAxes, fontsize=12)
    ax_raw.set_title("Raw Image Data", fontsize=14)
    
    # Flow visualization panel
    ax_flow = fig.add_subplot(gs[0, 1])
    
    # Create HSV flow visualization
    bgr, magnitude, max_flow = create_hsv_flow(vx, vy, max_flow)
    
    # Generate color wheel legend
    legend = opticalFlow.create_flow_color_wheel(100, 100)
    legend_rgb = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
    
    # Overlay legend in bottom right corner of the flow image
    h, w = bgr.shape[:2]
    lh, lw = legend.shape[:2] 
    pad = 20
    pos_x = w - lw - pad
    pos_y = h - lh - pad
    
    # Convert to RGB for matplotlib display
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_overlay = rgb.copy()
    rgb_overlay[pos_y:pos_y+lh, pos_x:pos_x+lw] = legend_rgb  # Use RGB legend
    
    # Show the overlay in the subplot
    ax_flow.imshow(rgb_overlay, origin='lower')
    ax_flow.set_title("HSV Flow (Hue=Direction, Saturation=Speed)", fontsize=14)
    
    # Add quiver arrows
    y, x = np.mgrid[0:vx.shape[0]:arrow_step, 0:vx.shape[1]:arrow_step]
    if max_flow > 0:
        ax_flow.quiver(x, y, vx[::arrow_step, ::arrow_step]/max_flow*10, 
                     vy[::arrow_step, ::arrow_step]/max_flow*10,
                     color='white', scale=1, scale_units='xy', 
                     angles='xy', alpha=0.7, width=0.002)
    
    # Clean up figure
    plt.tight_layout()
    
    # Convert figure to image
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    
    plt.close(fig)
    
    return img, max_flow

def create_vz_flow_frame(vz, raw_slice):
    """Create a frame with raw image and vz flow visualization"""
    
    # Set up figure with two panels side by side
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])
    
    # Normalize vz with tile-based approach
    vz_normalized = normalize_vz(vz)
    
    # vz flow panel
    ax_vz = fig.add_subplot(gs[0, 1])
    im = ax_vz.imshow(vz_normalized, cmap='RdBu_r', origin='lower', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax_vz, shrink=0.8)
    ax_vz.set_title('Out-of-plane Flow (vz)', fontsize=14)
    
    # Raw data panel
    ax_raw = fig.add_subplot(gs[0, 0])
    if raw_slice is not None:
        # Normalize raw data for display
        vmin = np.percentile(raw_slice, 5)
        vmax = np.percentile(raw_slice, 95)
        raw_display = np.clip(raw_slice, vmin, vmax)
        raw_display = (raw_display - vmin) / (vmax - vmin)
        ax_raw.imshow(raw_display, cmap='gray', origin='lower')
    else:
        ax_raw.text(0.5, 0.5, 'No Raw Data\nAvailable', ha='center', va='center', 
                  transform=ax_raw.transAxes, fontsize=12)
    ax_raw.set_title("Raw Image Data", fontsize=14)
    
    # Clean up figure
    plt.tight_layout()
    
    # Convert figure to image
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]

    plt.close(fig)
    
    return img

def create_video_from_frames(frames_dir, output_path, fps=10):
    """Create video from existing frame images using OpenCV"""
    
    # Get all frame files and sort them
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    
    if not frame_files:
        print(f"No frame files found in {frames_dir}")
        return False
    
    print(f"Found {len(frame_files)} frames in {frames_dir}")
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print(f"Error: Could not read first frame {frame_files[0]}")
        return False
    
    height, width, channels = first_frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return False
    
    # Write frames to video
    for frame_file in tqdm(frame_files, desc=f"Creating video {os.path.basename(output_path)}"):
        frame = cv2.imread(frame_file)
        if frame is not None:
            out.write(frame)
        else:
            print(f"Warning: Could not read frame {frame_file}")
    
    # Release everything
    out.release()
    
    print(f"✓ Video created: {output_path}")
    return True

def create_videos(results_dir, slice_index=None, frame_avg=False, arrow_step=10, fps=10):
    """Process all frames and create two videos"""
    
    # Define paths
    frames_dir = os.path.join(results_dir, "slice_visualization_frames")
    xy_frames_dir = os.path.join(frames_dir, "xy_frames")
    vz_frames_dir = os.path.join(frames_dir, "vz_frames")
    
    # Check if frame directories exist
    if not os.path.exists(xy_frames_dir) or not os.path.exists(vz_frames_dir):
        print("Frame directories not found. Generating frames from flow data...")
        
        # Get all available frame numbers
        available_frames = []
        for item in os.listdir(results_dir):
            if item.isdigit():
                frame_path = os.path.join(results_dir, item)
                if os.path.isdir(frame_path):
                    vx_file = os.path.join(frame_path, "optical_flow_vx.npy")
                    vy_file = os.path.join(frame_path, "optical_flow_vy.npy")
                    if os.path.exists(vx_file) and os.path.exists(vy_file):
                        available_frames.append(int(item))
        
        available_frames.sort()
        if not available_frames:
            print(f"No optical flow data found in {results_dir}")
            return
        
        print(f"Processing {len(available_frames)} frames: {min(available_frames)} to {max(available_frames)}")
        
        # Create output directory for frames
        os.makedirs(xy_frames_dir, exist_ok=True)
        os.makedirs(vz_frames_dir, exist_ok=True)
        
        # Process each frame
        first_frame = True
        max_flow = None  # Will be determined from first frame
        actual_slice_idx = None  # Store the actual slice index used

        # Find the channel index for cells
        parent_dir = os.path.dirname(results_dir)
        cells_channel = getCellChannelFromJSON(os.path.join(parent_dir, 'parameters.json'))
        
        for frame_number in tqdm(available_frames, desc="Processing frames"):
            try:
                # Check if frames already exist
                xy_frame_path = os.path.join(xy_frames_dir, f"frame_{frame_number:04d}.png")
                vz_frame_path = os.path.join(vz_frames_dir, f"frame_{frame_number:04d}.png")
                
                if os.path.exists(xy_frame_path) and os.path.exists(vz_frame_path):
                    if first_frame:
                        print(f"Frame {frame_number} already exists, skipping generation...")
                    continue
                
                # Load flow data
                if frame_avg:
                    flow_data = flowLoader.load_average_flow_frame(results_dir, frame_number)
                else:
                    flow_data = flowLoader.load_flow_frame(results_dir, frame_number)
                
                if not flow_data:
                    print(f"Warning: No flow data for frame {frame_number}, skipping")
                    continue
                    
                # Load raw data for comparison
                raw_data = flowLoader.load_raw_data(parent_dir, frame_number, cells_channel)
                
                # Extract slice (using same slice index for all frames)
                vx, vy, vz, conf, raw_slice, idx = flowLoader.extract_slice(
                    flow_data, raw_data, idx=slice_index
                )
                
                # Keep track of the slice index used
                if first_frame:
                    actual_slice_idx = idx
                    print(f"Using slice index {idx} for all frames")
                    first_frame = False
                
                # Create XY flow frame if it doesn't exist
                if not os.path.exists(xy_frame_path):
                    xy_frame, frame_max_flow = create_xy_flow_frame(vx, vy, raw_slice, max_flow, arrow_step)
                    
                    # Update max_flow for consistent scaling across frames
                    if max_flow is None:
                        max_flow = frame_max_flow
                    
                    cv2.imwrite(xy_frame_path, cv2.cvtColor(xy_frame, cv2.COLOR_RGB2BGR))
                
                # Create vz flow frame if it doesn't exist
                if not os.path.exists(vz_frame_path):
                    vz_frame = create_vz_flow_frame(vz, raw_slice)
                    cv2.imwrite(vz_frame_path, cv2.cvtColor(vz_frame, cv2.COLOR_RGB2BGR))
                
            except Exception as e:
                print(f"Error processing frame {frame_number}: {str(e)}")
                import traceback
                traceback.print_exc()
    else:
        print("Frame directories found. Skipping frame generation...")
    
    # Create videos from existing frames using OpenCV
    if slice_index is not None:
        if frame_avg:
            xy_video_path = os.path.join(results_dir, f"xy_flow_movie_slice{slice_index}_frame_avg.mp4")
            vz_video_path = os.path.join(results_dir, f"vz_flow_movie_slice{slice_index}_frame_avg.mp4")
        else:
            xy_video_path = os.path.join(results_dir, f"xy_flow_movie_slice{slice_index}.mp4")
            vz_video_path = os.path.join(results_dir, f"vz_flow_movie_slice{slice_index}.mp4")
    else:
        if frame_avg:
            xy_video_path = os.path.join(results_dir, "xy_flow_movie_frame_avg.mp4")
            vz_video_path = os.path.join(results_dir, "vz_flow_movie_frame_avg.mp4")
        else:
            xy_video_path = os.path.join(results_dir, "xy_flow_movie.mp4")
            vz_video_path = os.path.join(results_dir, "vz_flow_movie.mp4")
            
    # Create XY flow video
    success_xy = create_video_from_frames(xy_frames_dir, xy_video_path, fps)
    
    # Create VZ flow video  
    success_vz = create_video_from_frames(vz_frames_dir, vz_video_path, fps)
    
    if success_xy and success_vz:
        print(f"✓ Both videos created successfully:")
        print(f"  - XY flow video: {xy_video_path}")
        print(f"  - VZ flow video: {vz_video_path}")
    elif success_xy:
        print(f"✓ XY flow video created: {xy_video_path}")
        print(f"✗ VZ flow video failed")
    elif success_vz:
        print(f"✗ XY flow video failed")
        print(f"✓ VZ flow video created: {vz_video_path}")
    else:
        print("✗ Both videos failed to create")

def main():
    results_dir = sys.argv[1]
    slice_index = int(sys.argv[2])
    frame_avg = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
    
    # Validate results directory
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' does not exist.")
        sys.exit(1)
    
    print(f"Creating optical flow visualization movies from {results_dir}")
    print(f"Slice index: {slice_index if slice_index is not None else 'middle (auto)'}")

    create_videos(results_dir, slice_index, frame_avg)
    
if __name__ == "__main__":
    main()