import opticalflow3D
import zarr
import os
import sys
import cv2
import numpy as np
import imageio
import datetime

# Import cupy to handle GPU-CPU memory transfers
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("Warning: CuPy not available")

os.environ["CUPY_DUMP_CUDA_SOURCE_ON_ERROR"] = "1"
os.environ["CUPY_CACHE_IN_MEMORY"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CCCL_IGNORE_DEPRECATED_CPP_DIALECT"] = "1"
os.environ["CUPY_NVRTC_EXTRA_FLAGS"] = "--std=c++17"

def cupy_to_numpy(array):
    """Convert CuPy array to NumPy array if needed"""
    if CUPY_AVAILABLE and hasattr(array, 'get'):
        # This is a CuPy array, convert to NumPy
        return array.get()
    else:
        # Already a NumPy array or CuPy not available
        return array

def compute_3D_opticalflow(zarr_path):

    zarrFolder = zarr.open(zarr_path, mode='r+') 
    
    resArray = zarrFolder['0']['0']
    
    frame1 = resArray[0,0,:,:,:]
    frame2 = resArray[1,0,:,:,:]
    
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
        frame1, frame2, 
        start_point=(0, 300, 300),
        total_vol=(512, 512, 512),
        sub_volume=(350, 350, 350),
        overlap=(64, 64, 64),
        threadsperblock=(8, 8, 8),
    )
    
    # Convert CuPy arrays to NumPy arrays for CPU operations
    print("Converting GPU results to CPU memory...")
    output_vz = cupy_to_numpy(output_vz)
    output_vy = cupy_to_numpy(output_vy)
    output_vx = cupy_to_numpy(output_vx)
    output_confidence = cupy_to_numpy(output_confidence)
    
    print(f"Converted arrays - vz: {output_vz.shape}, vy: {output_vy.shape}, vx: {output_vx.shape}, confidence: {output_confidence.shape}")
    
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
    
    return output_vz, output_vy, output_vx, output_confidence

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

        print("3D optical flow computation completed successfully!")
        print(f"Output shapes - vz: {output_vz.shape}, vy: {output_vy.shape}, vx: {output_vx.shape}, confidence: {output_confidence.shape}")
        
    except Exception as e:
        print(f"Error during optical flow computation: {str(e)}")
        
        # Check if it's a CUDA compilation error
        if "CUDA" in str(e) or "cupy" in str(e).lower() or "nvcc" in str(e).lower():
            print("This appears to be a CUDA compilation error.")
            print("Possible solutions:")
            print("1. Update CuPy: pip install --upgrade cupy-cuda11x")
            print("2. Clear cache: rm -rf ~/.cupy/kernel_cache")
            print("3. Check CUDA/CuPy version compatibility")
            print("4. Consider using CPU-based optical flow instead")
        
        sys.exit(1)

if __name__ == "__main__":
    main()

