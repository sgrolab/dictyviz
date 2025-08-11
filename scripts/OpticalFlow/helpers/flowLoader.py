
# Purpose: Load optical flow data and raw image data (including zarr support)

import os
import numpy as np
import zarr
from scipy import ndimage
from tqdm import tqdm


def load_flow_frame(results_dir, frame_number):
    """Load optical flow data for a specific frame"""
    frame_dir = os.path.join(results_dir, str(frame_number))
    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"Frame directory {frame_dir} does not exist")
    
    components = ['vx', 'vy', 'vz']
    flow_data = {}
    
    for comp in components:
        path = os.path.join(frame_dir, f"optical_flow_{comp}.npy")
        if os.path.exists(path):
            flow_data[comp] = np.load(path)
        else:
            print(f"Warning: {comp} missing at {path}")
    
    return flow_data

def load_average_flow_frame(results_dir, frame_number):
    """Load the frame averaged flow data for a specific frame"""
    frame_dir = os.path.join(results_dir, str(frame_number))
    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"Frame directory {frame_dir} does not exist")
    
    avg_flow_path = os.path.join(frame_dir, f"avg_flow_frame_{frame_number}.npy")
    if not os.path.exists(avg_flow_path):
        return None
    
    avg_flow = np.load(avg_flow_path)
    flow_data = {
        'vx': avg_flow[0],
        'vy': avg_flow[1],
        'vz': avg_flow[2] if avg_flow.shape[0] > 2 else None,
        'confidence': avg_flow[3] if avg_flow.shape[0] > 3 else None
    }
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

def load_first_frames(results_dir, nb_frames, log_file=None):
    """
    Load the first nb_frames of flow data from the results directory.
    """

    flow_frame_0 = load_flow_frame(results_dir, 0)
    lenZ, lenY, lenX = flow_frame_0['vx'].shape
    flow_data = np.zeros((3, 5, lenZ, lenY, lenX), dtype=np.float32)  # initialize flow data array

    for frame in range(nb_frames):
        if log_file:
            log_file.write(f"Loading flow data for frame {frame}...\n")
        else:
            print(f"Loading flow data for frame {frame}...")
        flow_frame = load_flow_frame(results_dir, frame)
        
        flow_frame_vx = flow_frame['vx']
        flow_frame_vy = flow_frame['vy']
        flow_frame_vz = flow_frame['vz']

        flow_data[0, frame] = flow_frame_vx
        flow_data[1, frame] = flow_frame_vy
        flow_data[2, frame] = flow_frame_vz
    
    return flow_data

def load_next_frame(flow_data, results_dir, frame_index, log_file=None):
    """
    Load the next frame of flow data and update the flow_data array.
    """
    frame_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    if frame_index < 0 or frame_index >= len(frame_dirs):
        print(f"Error: frame_index {frame_index} is out of bounds for flow_data with shape {flow_data.shape}.")
        return

    if log_file:
        log_file.write(f"Loading next flow frame for index {frame_index}...\n")
    else:
        print(f"Loading next flow frame for index {frame_index}...")

    # drop the first frame and add the next frame
    flow_data = flow_data[:, 1:]

    next_frame = load_flow_frame(results_dir, frame_index)

    flow_frame_vx = next_frame['vx']
    flow_frame_vy = next_frame['vy']
    flow_frame_vz = next_frame['vz']

    flow_data[0, -1] = flow_frame_vx
    flow_data[1, -1] = flow_frame_vy
    flow_data[2, -1] = flow_frame_vz