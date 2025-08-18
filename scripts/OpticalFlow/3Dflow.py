import os

# set environment variables for PyTorch CUDA compatibility
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;7.0;7.5;8.0;8.6"  # supports multiple GPU architectures
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # prevents memory fragmentation

import opticalflow3D
import sys
import numpy as np
import zarr
import torch
import logging

def cutoff_at_threshold(frame, threshold):
    """
    Set all values in the frame to zero if they are below the threshold.
    """
    frame[frame < threshold] = 0
    return frame

def compute_3D_opticalflow(zarr_path):

    parent_dir = os.path.dirname(zarr_path)
    
    # Extract zarr file/directory name for naming convention
    zarr_name = os.path.basename(zarr_path)
    # Remove .zarr extension if present
    if zarr_name.endswith('.zarr'):
        zarr_name = zarr_name[:-5]
    
    # Create output directory with zarr name included
    output_dir = os.path.join(parent_dir, "optical_flow_3Dresults")

    # create main output directory
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "gpu_usage.log")

    #setting up logging
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

    zarrFolder = zarr.open(zarr_path, mode='r+') 
    
    resArray = zarrFolder['0']['0']
    
    # get the total number of frames
    num_frames = resArray.shape[0]

    z_dim, y_dim, x_dim = resArray.shape[2], resArray.shape[3], resArray.shape[4]

    # Determine volume size category
    max_dim = max(z_dim, y_dim, x_dim)
    is_small = max_dim < 400
    is_medium = 400 <= max_dim < 800
    is_large = max_dim >= 800

    # Adaptive iterations: gradually increase with data size
    if is_small:
        iters = 3
    elif is_medium:
        iters = 5
    else:  # large
        iters = 8

    # Adaptive pyramid levels: more levels for larger data
    if is_small:
        num_levels = 2
    elif is_medium:
        num_levels = 3
    else:  # large
        num_levels = 4

    # Adaptive scale: finer for small data, coarser for large
    if is_small:
        scale = 0.7  # Less downsampling for small volumes
    elif is_medium:
        scale = 0.5
    else:  # large
        scale = 0.3  # More aggressive downsampling for large volumes

    # Adaptive spatial size: larger for bigger data
    if is_small:
        spatial_size = 2
    elif is_medium:
        spatial_size = 3
    else:  # large
        spatial_size = 4

    # Adaptive presmoothing: less for small data, more for noisy/large
    if is_small:
        presmoothing = 2
    elif is_medium:
        presmoothing = 4
    else:  # large
        presmoothing = 6

    sigma_k = 1.5  # Standard deviation for Gaussian filter, can be adjusted based on noise level

    # Filter type: gaussian for quality, box for speed with very large volumes
    filter_type = "box" if is_large else "gaussian"

    # Log selected parameters
    print(f"Volume max dimension: {max_dim} pixels")
    print(f"Size category: {'Small' if is_small else ('Medium' if is_medium else 'Large')}")
    print(f"Parameters: iters={iters}, levels={num_levels}, scale={scale}, spatial_size={spatial_size}, sigma_k={sigma_k}, presmoothing={presmoothing}, filter={filter_type}")

    successful_frames = [] 

    current_frame = resArray[0, 0, :, :, :]  # Get the first frame to determine dimensions
    
    # Cutoff at a threshold to remove noise
    threshold = 3000 # Hard coded for now
    current_frame = cutoff_at_threshold(current_frame, threshold)

    current_frame_np = np.asarray(current_frame, dtype=np.float32)

    # Loop through consecutive frame pairs
    #for i in range(num_frames - 1):
    for i in range(111, 125):  # Example range for testing

        print(f"\n--- Processing frame pair {i} -> {i+1} ---")

        # Check if output .npy files already exist
        frame_dir = os.path.join(output_dir, str(i))
        output_vz_path = os.path.join(frame_dir, "optical_flow_vz.npy")
        output_vy_path = os.path.join(frame_dir, "optical_flow_vy.npy")
        output_vx_path = os.path.join(frame_dir, "optical_flow_vx.npy")
        output_confidence_path = os.path.join(frame_dir, "optical_flow_confidence.npy")
        if os.path.exists(output_vz_path) and os.path.exists(output_vy_path) and os.path.exists(output_vx_path) and os.path.exists(output_confidence_path):
            print(f"Output files for frame pair {i}->{i+1} already exist. Skipping...")
            successful_frames.append(i)
            continue
        
        # Create output directory for this frame pair
        os.makedirs(frame_dir, exist_ok=True)
        
        # Monitor GPU memory
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logging.info(f'GPU is available: {device_name}')
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        else:
            logging.info("GPU is not available")

        # get next frame
        next_frame = resArray[i+1, 0, :, :, :]

        next_frame = cutoff_at_threshold(next_frame, threshold)

        # Convert to float32 NumPy arrays first (the library expects NumPy, not PyTorch tensors)
        next_frame_np = np.asarray(next_frame, dtype=np.float32)

        print(f"Input arrays - current_frame_np: {current_frame_np.shape}, next_frame_np: {next_frame_np.shape}")
        
        # Dynamically set total_vol, sub_volume, and overlap

        total_vol = current_frame_np.shape
        sub_volume = tuple(max(1, s // 2) for s in total_vol)  # 1/2 of each dimension, at least 1
        overlap = tuple(max(1, int(sv * 0.6)) for sv in sub_volume)  # 60% of sub_volume, at least 1
    
        # FIXED: Set filter size to be safe for padding operations
        # Filter size should be small enough that padding doesn't exceed input dimensions
        # For a dimension of size N, padding of filter_size should satisfy: 2*filter_size < N
        min_dim = min(total_vol)
        filter_size = max(1, min(min_dim // 3, min(sub_volume) // 3))  
        filter_size = max(3, (filter_size // 2) * 2 + 1)  # Ensures filter_size is odd and at least 3
        print(f"Total volume: {total_vol}, Sub-volume: {sub_volume}, Overlap: {overlap}, Filter size: {filter_size}")

         # Initialize the farneback object
        farneback = opticalflow3D.Farneback3D(
            iters = iters,
            num_levels = num_levels,
            scale = scale,
            spatial_size = spatial_size,
            sigma_k = sigma_k,
            presmoothing = presmoothing,
            filter_type = filter_type,
            filter_size = filter_size,
        )

        try:
            # Calculate optical flow between consecutive frames
            output_vz, output_vy, output_vx, output_confidence = farneback.calculate_flow(
                current_frame_np, next_frame_np, 
                start_point=(0, 0, 0),
                total_vol=total_vol,
                sub_volume=sub_volume,
                overlap=overlap,
            )   

            print(f"Output tensors - vz: {output_vz.shape}, vy: {output_vy.shape}, vx: {output_vx.shape}, conf: {output_confidence.shape}")
            print(f"Output types - vz: {type(output_vz)}, vy: {type(output_vy)}, vx: {type(output_vx)}, conf: {type(output_confidence)}")
            
            # Save individual frame results directly
            if isinstance(output_vz, torch.Tensor):
                np.save(output_vz_path, output_vz.detach().cpu().numpy())
                np.save(output_vy_path, output_vy.detach().cpu().numpy())
                np.save(output_vx_path, output_vx.detach().cpu().numpy())
                np.save(output_confidence_path, output_confidence.detach().cpu().numpy())
            else:
                np.save(output_vz_path, np.asarray(output_vz))
                np.save(output_vy_path, np.asarray(output_vy))
                np.save(output_vx_path, np.asarray(output_vx))
                np.save(output_confidence_path, np.asarray(output_confidence))

            successful_frames.append(i)
            print(f"Frame {i}->{i+1} completed. Saved to: {frame_dir}")

            current_frame_np = next_frame_np  # Update current frame for next iteration
            
            # Force GPU memory cleanup after each frame
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
        except Exception as e:
            print(f"Error processing frames {i}->{i+1}: {str(e)}")
            # Clear GPU memory on error too
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Continue with next frame pair instead of stopping
            continue
    
    if len(successful_frames) == 0:
        raise ValueError("No optical flow results were calculated successfully")

    print("Optical flow calculation completed successfully!")
    print(f"Processed {len(successful_frames)} frame pairs")
    print(f"Successfully processed frames: {successful_frames}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")
    
    print(f"Individual frame results saved in subfolders: {output_dir}")
    
    return successful_frames
    
def main():
    # check if zarr_path is provided as command line argument
    if len(sys.argv) < 2:
        print("Error: No zarr path provided.")
        print("Usage: python 3Dopticalflow.py <path_to_zarr_file_or_directory>")
        sys.exit(1)
        
    zarr_path = sys.argv[1]
    
    # validate that zarr_path exists
    if not os.path.exists(zarr_path):
        print(f"Error: Zarr path '{zarr_path}' does not exist.")
        print("Usage: python 3Dopticalflow.py <path_to_zarr_file_or_directory>")
        sys.exit(1)
    
    try:
        print(f"Computing 3D optical flow for: {zarr_path}")

        successful_frames = compute_3D_opticalflow(zarr_path)

        print("3D optical flow computation completed successfully!")
        print(f"Successfully processed {len(successful_frames)} frame pairs")

    except Exception as e:
        print(f"Error during optical flow computation: {str(e)}")
        
        sys.exit(1)

if __name__ == "__main__":
    main()