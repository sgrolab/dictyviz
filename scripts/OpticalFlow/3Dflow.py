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

def compute_3D_opticalflow(zarr_path):

    parent_dir = os.path.dirname(zarr_path)
    
    # Extract zarr file/directory name for naming convention
    zarr_name = os.path.basename(zarr_path)
    # Remove .zarr extension if present
    if zarr_name.endswith('.zarr'):
        zarr_name = zarr_name[:-5]
    
    zarr_name_short = zarr_name[-13:]
    
    # Create output directory with zarr name included
    output_dir = os.path.join(parent_dir, f"{zarr_name_short}_optical_flow_3Dresults")

    # create main output directory
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "gpu_usage.log")

    #setting up logging
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

    zarrFolder = zarr.open(zarr_path, mode='r+') 
    
    resArray = zarrFolder['0']['0']
    
    # get the total number of frames
    num_frames = resArray.shape[0]

    # Initialize the farneback object
    farneback = opticalflow3D.Farneback3D(
        iters = 5,
        num_levels = 3,
        scale = 0.6,
        spatial_size = 5,
        presmoothing = 7,
        filter_type = "box",
        filter_size = 21,
    )

    successful_frames = []

    # Loop through consecutive frame pairs
    for i in range(num_frames - 1):
        print(f"\n--- Processing frame pair {i} -> {i+1} ---")
        
        # Monitor GPU memory
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logging.info(f'GPU is available: {device_name}')
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        else:
            logging.info("GPU is not available")

        # get consecutive frames
        frame1 = resArray[i, 0, :, :, :]
        frame2 = resArray[i+1, 0, :, :, :]

        # Convert to float32 NumPy arrays first (the library expects NumPy, not PyTorch tensors)
        frame1_np = np.asarray(frame1, dtype=np.float32)
        frame2_np = np.asarray(frame2, dtype=np.float32)
        
        print(f"Input arrays - frame1_np: {frame1_np.shape}, frame2_np: {frame2_np.shape}")
        
        try:
            # Calculate optical flow between consecutive frames
            output_vz, output_vy, output_vx, output_confidence = farneback.calculate_flow(
                frame1_np, frame2_np, 
                start_point=(0, 0, 0),
                total_vol=(217, 1906, 1440),
                sub_volume=(96, 192, 192),
                overlap=(48, 96, 96),
            )

            print(f"Output tensors - vz: {output_vz.shape}, vy: {output_vy.shape}, vx: {output_vx.shape}, conf: {output_confidence.shape}")
            print(f"Output types - vz: {type(output_vz)}, vy: {type(output_vy)}, vx: {type(output_vx)}, conf: {type(output_confidence)}")
            
            # Create folder for this time point
            frame_dir = os.path.join(output_dir, str(i))
            os.makedirs(frame_dir, exist_ok=True)
            
            # Save individual frame results directly
            if isinstance(output_vz, torch.Tensor):
                np.save(os.path.join(frame_dir, "optical_flow_vz.npy"), output_vz.detach().cpu().numpy())
                np.save(os.path.join(frame_dir, "optical_flow_vy.npy"), output_vy.detach().cpu().numpy())
                np.save(os.path.join(frame_dir, "optical_flow_vx.npy"), output_vx.detach().cpu().numpy())
                np.save(os.path.join(frame_dir, "optical_flow_confidence.npy"), output_confidence.detach().cpu().numpy())
            else:
                np.save(os.path.join(frame_dir, "optical_flow_vz.npy"), np.asarray(output_vz))
                np.save(os.path.join(frame_dir, "optical_flow_vy.npy"), np.asarray(output_vy))
                np.save(os.path.join(frame_dir, "optical_flow_vx.npy"), np.asarray(output_vx))
                np.save(os.path.join(frame_dir, "optical_flow_confidence.npy"), np.asarray(output_confidence))
            
            successful_frames.append(i)
            print(f"Frame {i}->{i+1} completed. Saved to: {frame_dir}")
            
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