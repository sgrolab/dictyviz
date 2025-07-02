import opticalflow3D
import zarr
import os
import sys
import cv2
import numpy as np
import imageio
import datetime

def compute_3D_opticalflow(zarr_path):

    # open zarr file
    print(f"Opening zarr file: {zarr_path}")
    zarrFolder = zarr.open(zarr_path, mode='r+') 
        
    # access the data array
    resArray = zarrFolder['0']['0']
    print(f"Data array shape: {resArray.shape}")
        
    # extract two consecutive frames for optical flow
    frame1 = resArray[0,0,:,:,:]
    frame2 = resArray[1,0,:,:,:]
    
    print(f"Frame1 shape: {frame1.shape}, Frame2 shape: {frame2.shape}")
    
    # Debug: Check what's available in opticalflow3D module
    print("Available attributes in opticalflow3D:", dir(opticalflow3D))
    
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
        sys.exit(1)

if __name__ == "__main__":
    main()

