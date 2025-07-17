"""
Z-Stack Movie Generation Script using OpenCV
Creates an MP4 video showing Z-slice progression at a given time point
"""
import os
import sys
import numpy as np
from datetime import datetime
import zarr
import cv2

def load_zarr_data(zarr_path):
    """Load zarr data from the specified path (inside 0/0)."""
    try:
        zarr_folder = zarr.open(zarr_path, mode='r')
        res_array = zarr_folder['0']['0']
        print(f"Loaded zarr data with shape: {res_array.shape}")
        return res_array
    except Exception as e:
        print(f"Error loading zarr data: {e}")
        return None

def normalize_image(img, percentile_range=(5, 95)):
    """Normalize image using percentile clipping for display purposes."""
    vmin = np.percentile(img, percentile_range[0])
    vmax = np.percentile(img, percentile_range[1])
    img_clipped = np.clip(img, vmin, vmax)
    normalized = (img_clipped - vmin) / (vmax - vmin + 1e-8)  # avoid divide-by-zero
    return normalized

def create_frame_with_info(slice_img, output_size=(1200, 900)):
    """Create a single frame from a 2D Z-slice, properly normalized and resized."""
    normalized = normalize_image(slice_img)
    normalized_uint8 = (normalized * 255).astype(np.uint8)
    resized = cv2.resize(normalized_uint8, output_size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)  # convert to 3-channel BGR

def save_movie(frames, output_path, fps=8, output_size=(1200, 900)):
    """Save movie using OpenCV VideoWriter."""
    if not frames:
        print("No frames to save")
        return False
    
    print(f"Saving {len(frames)} frames to video...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, output_size)
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        output_path_avi = output_path.replace('.mp4', '.avi')
        out = cv2.VideoWriter(output_path_avi, cv2.VideoWriter_fourcc(*'XVID'), fps, output_size)
        if not out.isOpened():
            print("Error: Could not open alternative video writer either")
            return False
        else:
            output_path = output_path_avi
            print(f"Using AVI fallback: {output_path}")
    
    for i, frame in enumerate(frames):
        print(f"Writing frame {i+1}/{len(frames)}")
        out.write(frame)
    
    out.release()
    print(f"✓ Video saved: {output_path}")
    return True

def main():

    # Construct movie output path in z_movies/<time_point>/...
    zarr_name = os.path.basename(zarr_path).replace('.zarr', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("z_movies", f"time_{time_point:03d}")

    # create the directory if it doesn't exist
    print("creating output folder")
    os.makedirs(output_dir, exist_ok=True)

    if len(sys.argv) < 3:
        print("Usage: python3 makeMovie.py <zarr_path> <time_point>")
        sys.exit(1)

    zarr_path = sys.argv[1]
    time_point = int(sys.argv[2])
    
    print(f"Z-Stack Movie Generator")
    print(f"Zarr path: {zarr_path}")
    print(f"Time point: {time_point}")
    print("=" * 50)
    
    if not os.path.exists(zarr_path):
        print(f"Error: Zarr path not found: {zarr_path}")
        sys.exit(1)
    
    data = load_zarr_data(zarr_path)
    if data is None:
        sys.exit(1)
    
    if time_point >= data.shape[0]:
        print(f"Error: Time point {time_point} is out of range (0–{data.shape[0]-1})")
        sys.exit(1)

    # Extract the desired time point
    time_data = data[time_point, 0, :, :, :]  # [Z, Y, X]
    z_slices = time_data.shape[0]
    
    print(f"Processing {z_slices} Z-slices...")

    # Generate video frames
    output_size = (1200, 900)
    fps = 8
    frames = []
    for z in range(z_slices):
        print(f"Processing Z-slice {z+1}/{z_slices}")
        slice_img = time_data[z, :, :]
        frame = create_frame_with_info(slice_img, output_size)
        frames.append(frame)

    output_filename = f"zstack_{zarr_name}_time{time_point:03d}_{timestamp}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    print(f"Saving movie to: {output_path}")
    success = save_movie(frames, output_path, fps, output_size)

    if success:
        print(f"\n{'='*50}")
        print(f"✓ Movie generation complete!")
        print(f"✓ Output: {output_path}")
    else:
        print("✗ Failed to save movie")
        sys.exit(1)

if __name__ == '__main__':
    main()