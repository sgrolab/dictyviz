import numpy as np
import sys
import helpers.flowLoader as flowLoader
from matplotlib.colors import hsv_to_rgb
import napari 

def main():
    # Get parameters 
    results_dir = sys.argv[1]
    frame_number = int(sys.argv[2])

    # Get flow data 
    flow_data = flowLoader.load_flow_frame(results_dir, frame_number)
    vx = flow_data['vx']
    vy = flow_data['vy']   
    vz = flow_data.get('vz')

    if vz is None:
        vz = np.zeros_like(vx)  # Assume no z-motion

    print(f"Flow data shape: vx={vx.shape}, vy={vy.shape}, vz={vz.shape}")

    # Compute flow magnitude and normalize
    mag = np.sqrt(vx**2 + vy**2 + vz**2)
    norm_mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

    # Normalize direction components
    epsilon = 1e-6
    vx_norm = vx / (mag + epsilon)
    vy_norm = vy / (mag + epsilon)
    vz_norm = vz / (mag + epsilon)

    # Convert to spherical coordinates
    # Azimuth = angle in XY plane (horizontal)
    azimuth = np.arctan2(vy_norm, vx_norm)
    azimuth_norm = (azimuth + np.pi) / (2 * np.pi)  # Normalize to [0, 1]

    # Elevation = angle from Z-axis (vertical)
    elevation = np.arccos(np.clip(vz_norm, -1, 1))  # [0, π]
    elevation_norm = elevation / np.pi  # Normalize to [0, 1]

    # Create HSV: Hue=azimuth, Saturation=elevation, Value=magnitude
    hsv = np.stack([azimuth_norm, elevation_norm, norm_mag], axis=-1)
    print(f"HSV shape: {hsv.shape}")

    # Convert to RGB
    rgb = hsv_to_rgb(hsv).astype(np.float32)
    print(f"RGB shape after conversion: {rgb.shape}")

    # Visualize in Napari
    viewer = napari.Viewer()
    
    viewer.add_image(rgb, 
                    name='3D Optical Flow RGB',
                    rgb=True,
                    colormap=None)
    
    napari.run()

if __name__ == "__main__":
    main()