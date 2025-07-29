
# Purpose: Load optical flow data and raw image data (including zarr support)

import os
import numpy as np
import zarr
from scipy import ndimage

def load_flow_frame(results_dir, frame_number):
    """Load optical flow data for a specific frame"""
    frame_dir = os.path.join(results_dir, str(frame_number))
    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"Frame directory {frame_dir} does not exist")
    
    components = ['vx', 'vy', 'vz', 'confidence']
    flow_data = {}
    
    for comp in components:
        path = os.path.join(frame_dir, f"optical_flow_{comp}.npy")
        if os.path.exists(path):
            flow_data[comp] = np.load(path)
        else:
            print(f"Warning: {comp} missing at {path}")
    
    return flow_data

def load_raw_data(results_dir, frame_number):
    """Load raw image data for comparison"""

    # Try to find zarr file in parent directory
    parent_dir = os.path.dirname(results_dir)
    zarr_files = [f for f in os.listdir(parent_dir) if f.endswith('.zarr')]
    
    if not zarr_files:
        print("Warning: No zarr file found for raw data")
        return None
    
    try:
        zarr_path = os.path.join(parent_dir, zarr_files[0])
        zarr_folder = zarr.open(zarr_path, mode='r')
        res_array = zarr_folder['0']['0']
        
        if frame_number < res_array.shape[0]:
            print(f"Loading raw data: frame {frame_number} from {zarr_path}")
            raw_frame = np.asarray(res_array[frame_number, 0, :, :, :])
            return raw_frame
        else:
            print(f"Warning: Frame {frame_number} out of bounds in raw data (max index = {res_array.shape[0]-1})")
    except Exception as e:
        print(f"Warning: Could not load raw data: {e}")
    
    return None

def extract_slice(flow_data, raw_data=None, idx=None):
    """Extract full 3D flow"""
    
    shape = flow_data['vx'].shape
    print(f"Data shape (Z, Y, X): {shape}")
    
    if idx is None:
        idx = shape[0] // 2  # Use middle slice as default
        print(f"Using middle slice: {idx}")
    else:
        print(f"Using user-specified slice: {idx}")
    
    # Validate slice index
    if idx >= shape[0] or idx < 0:
        print(f"Warning: Slice {idx} out of bounds [0, {shape[0]-1}]. Using middle slice.")
        idx = shape[0] // 2

    vx = flow_data['vx'][idx, :, :]
    vy = flow_data['vy'][idx, :, :]
    vz = flow_data.get('vz', None)
    
    if vz is not None:
        vz_raw = vz[idx, :, :]
        vz = ndimage.gaussian_filter(vz_raw, sigma=1.5)
        
    conf = flow_data.get('confidence', None)
    if conf is not None:
        conf = conf[idx, :, :]
    
    # Extract raw data slice if available
    raw_slice = None
    if raw_data is not None:
        raw_slice = raw_data[idx, :, :]
    
    return vx, vy, vz, conf, raw_slice, idx