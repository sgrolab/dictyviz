import numpy as np
import sys
import os
import helpers.flowLoader as flowLoader
from matplotlib.colors import hsv_to_rgb
import tifffile

def main():

    results_dir = sys.argv[1]
    frame_number = int(sys.argv[2])

    flow_data = flowLoader.load_flow_frame(results_dir, frame_number)
    vx = flow_data['vx']
    vy = flow_data['vy'] 
    vz = flow_data.get('vz')
    if vz is None:
        vz = np.zeros_like(vx)

    mag = np.sqrt(vx**2 + vy**2 + vz**2)
    norm_mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

    epsilon = 1e-6
    vx_norm = vx / (mag + epsilon)
    vy_norm = vy / (mag + epsilon)
    vz_norm = vz / (mag + epsilon)

    azimuth = np.arctan2(vy_norm, vx_norm)
    azimuth_norm = (azimuth + np.pi) / (2 * np.pi)

    elevation = np.arccos(np.clip(vz_norm, -1, 1))
    elevation_norm = elevation / np.pi

    hsv = np.stack([azimuth_norm, elevation_norm, norm_mag], axis=-1)
    rgb = hsv_to_rgb(hsv).astype(np.float32)

    print(f"RGB shape: {rgb.shape}")
    print("Saving RGB volume to optical_flow_rgb.tif...")

    frame_dir = os.path.join(results_dir, str(frame_number))
    os.makedirs(frame_dir, exist_ok=True)  # create folder if it doesn't exist
    file_path = os.path.join(frame_dir, "optical_flow_rgb.tiff")
    tifffile.imwrite(file_path, rgb, photometric='rgb')

if __name__ == "__main__":
    main()