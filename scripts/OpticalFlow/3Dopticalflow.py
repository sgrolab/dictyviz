import os

# set environment variables for PyTorch CUDA compatibility
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;7.0;7.5;8.0;8.6"  # supports multiple GPU architectures
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # prevents memory fragmentation

import opticalflow3D
import sys
import numpy as np
import datetime
import zarr
import torch

def compute_3D_opticalflow(zarr_path):

    zarrFolder = zarr.open(zarr_path, mode='r+') 
    
    resArray = zarrFolder['0']['0']
    
    # get the total number of frames
    num_frames = resArray.shape[0]

    # Initialize the farneback object
    farneback = opticalflow3D.Farneback3D(
        iters = 5,
        num_levels = 5,
        scale = 0.5,
        spatial_size = 7,
        presmoothing = 7,
        filter_type = "box",
        filter_size = 21,
    )
    
    # Initialize lists to store all optical flow results
    all_vz = []
    all_vy = []
    all_vx = []
    all_confidence = []
    
    # Loop through consecutive frame pairs
    for i in range(num_frames - 1):

        # get consecutive frames
        frame1 = resArray[i, 0, :, :, :]
        frame2 = resArray[i+1, 0, :, :, :]

        # Convert to float32 NumPy arrays first (the library expects NumPy, not PyTorch tensors)
        frame1_np = np.asarray(frame1, dtype=np.float32)
        frame2_np = np.asarray(frame2, dtype=np.float32)
        
        try:
            # Calculate optical flow between consecutive frames
            output_vz, output_vy, output_vx, output_confidence = farneback.calculate_flow(
                frame1_np, frame2_np, 
                start_point=(0, 0, 0),
                total_vol=(217, 286, 286),
                sub_volume=(185, 254, 254),
                overlap=(23, 31, 31),
            )
            
            # Convert PyTorch tensors back to NumPy arrays if needed
            if isinstance(output_vz, torch.Tensor):
                output_vz = output_vz.detach().cpu().numpy()
                output_vy = output_vy.detach().cpu().numpy()
                output_vx = output_vx.detach().cpu().numpy()
                output_confidence = output_confidence.detach().cpu().numpy()
            else:
                output_vz = np.asarray(output_vz)
                output_vy = np.asarray(output_vy)
                output_vx = np.asarray(output_vx)
                output_confidence = np.asarray(output_confidence)
            
            # Store results
            all_vz.append(output_vz)
            all_vy.append(output_vy)
            all_vx.append(output_vx)
            all_confidence.append(output_confidence)
            
            print(f"Flow shapes: vz={output_vz.shape}, vy={output_vy.shape}, vx={output_vx.shape}, conf={output_confidence.shape}")
            
        except Exception as e:
            print(f"Error processing frames {i}->{i+1}: {str(e)}")
            # Continue with next frame pair instead of stopping
            continue
    
    # Convert lists to numpy arrays
    print(f"\nCombining {len(all_vz)} optical flow results...")
    output_vz = np.stack(all_vz, axis=0) if all_vz else np.array([])
    output_vy = np.stack(all_vy, axis=0) if all_vy else np.array([])
    output_vx = np.stack(all_vx, axis=0) if all_vx else np.array([])
    output_confidence = np.stack(all_confidence, axis=0) if all_confidence else np.array([])
    
    if len(all_vz) == 0:
        raise ValueError("No optical flow results were calculated successfully")

    print("Optical flow calculation completed successfully!")
    print(f"Processed {len(all_vz)} frame pairs")
    print(f"Output types: vz={type(output_vz)}, vy={type(output_vy)}, vx={type(output_vx)}, conf={type(output_confidence)}")
    print(f"Final output shapes: vz={output_vz.shape}, vy={output_vy.shape}, vx={output_vx.shape}, conf={output_confidence.shape}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")
    
    # save results
    timestamp = datetime.datetime.now()
    parent_dir = os.path.dirname(zarr_path)
    output_dir = os.path.join(parent_dir, "optical_flow_3Dresults")
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
        print(f"Final output shapes - vz: {output_vz.shape}, vy: {output_vy.shape}, vx: {output_vx.shape}, confidence: {output_confidence.shape}")

    except Exception as e:
        print(f"Error during optical flow computation: {str(e)}")
        
        sys.exit(1)

if __name__ == "__main__":
    main()