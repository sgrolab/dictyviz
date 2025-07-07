import os

# Set environment variables for PyTorch CUDA compatibility
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;7.0;7.5;8.0;8.6"  # Support multiple GPU architectures
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Prevent memory fragmentation

import opticalflow3D
import sys
import numpy as np
import datetime
import zarr
import torch
import inspect

def compute_3D_opticalflow(zarr_path):

    zarrFolder = zarr.open(zarr_path, mode='r+') 
    
    resArray = zarrFolder['0']['0']
    
    frame1 = resArray[123,0,:,:,:]
    frame2 = resArray[124,0,:,:,:]

    # Convert to float32 NumPy arrays first (the library expects NumPy, not PyTorch tensors)
    frame1_np = np.asarray(frame1, dtype=np.float32)
    frame2_np = np.asarray(frame2, dtype=np.float32)

    farneback = opticalflow3D.Farneback3D(
        iters = 5,
        num_levels = 5,
        scale = 0.5,
        spatial_size = 7,
        presmoothing = 7,
        filter_type = "box",
        filter_size = 21,
    )
    
    output_vz, output_vy, output_vx, output_confidence = farneback.calculate_flow(
        frame1_np, frame2_np, 
        start_point=(0, 300, 300),
        total_vol=(512, 512, 512),
        sub_volume=(350, 350, 350),
        overlap=(64, 64, 64),
    )
    
    
    print("Optical flow calculation completed successfully!")
    
    print(f"Output types: vz={type(output_vz)}, vy={type(output_vy)}, vx={type(output_vx)}, conf={type(output_confidence)}")
    
    # Convert PyTorch tensors back to NumPy arrays for saving
    if isinstance(output_vz, torch.Tensor):
        print("Converting PyTorch tensors to NumPy arrays...")
        output_vz = output_vz.detach().cpu().numpy()
        output_vy = output_vy.detach().cpu().numpy()
        output_vx = output_vx.detach().cpu().numpy()
        output_confidence = output_confidence.detach().cpu().numpy()
    else:
        print("Converting to NumPy arrays...")
        output_vz = np.asarray(output_vz)
        output_vy = np.asarray(output_vy)
        output_vx = np.asarray(output_vx)
        output_confidence = np.asarray(output_confidence)
    
    print(f"Final output shapes: vz={output_vz.shape}, vy={output_vy.shape}, vx={output_vx.shape}, conf={output_confidence.shape}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")
    
    # save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_dir = os.path.dirname(zarr_path)
    output_dir = os.path.join(parent_dir, "optical_flow_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # save flow components as numpy arrays
    np.save(os.path.join(output_dir, f"optical_flow_vz_{timestamp}.npy"), output_vz)
    np.save(os.path.join(output_dir, f"optical_flow_vy_{timestamp}.npy"), output_vy)
    np.save(os.path.join(output_dir, f"optical_flow_vx_{timestamp}.npy"), output_vx)
    np.save(os.path.join(output_dir, f"optical_flow_confidence_{timestamp}.npy"), output_confidence)
    
    print(f"Optical flow results saved to: {output_dir}")
    
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

        output_vz, output_vy, output_vx, output_confidence = compute_3D_opticalflow(zarr_path)

    except Exception as e:
        print(f"Error during optical flow computation: {str(e)}")
        
        sys.exit(1)

if __name__ == "__main__":
    main()