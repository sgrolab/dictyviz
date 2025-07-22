import numpy as np
import sys
import os
import helpers.flowLoader as flowLoader
from tqdm import tqdm

def main():
    results_dir = sys.argv[1]
    
    # Discover all available frames in the results directory
    available_frames = []
    for item in os.listdir(results_dir):
        if item.isdigit():  # Check if directory name is a number
            frame_path = os.path.join(results_dir, item)
            if os.path.isdir(frame_path):
                # Check if this frame has flow data files
                vx_file = os.path.join(frame_path, "vx.npy")
                vy_file = os.path.join(frame_path, "vy.npy")
                if os.path.exists(vx_file) and os.path.exists(vy_file):
                    available_frames.append(int(item))
    
    available_frames.sort()
    lenT = len(available_frames)
    
    if lenT == 0:
        print(f"No optical flow data found in {results_dir}")
        return
    
    print(f"Processing {lenT} frames: {min(available_frames)} to {max(available_frames)}")
    
    # Process all frames
    for i, frame_number in enumerate(tqdm(available_frames, desc="Processing optical flow frames")):
        try:
            # Load flow data for this frame
            flow_data = flowLoader.load_flow_frame(results_dir, frame_number)
            vx = flow_data['vx']
            vy = flow_data['vy'] 
            vz = flow_data.get('vz')
            if vz is None:
                vz = np.zeros_like(vx)

            # Compute magnitude and normalize
            mag = np.sqrt(vx**2 + vy**2 + vz**2)
            norm_mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

            epsilon = 1e-6

            # Gives a unit direction vector per voxel, this isolates the direction of flow
            vx_norm = vx / (mag + epsilon)
            vy_norm = vy / (mag + epsilon)
            vz_norm = vz / (mag + epsilon)

            # Transforms values from [-1, 1] to [0, 1] for RGB representation
            r = (vx_norm + 1.0) / 2.0
            g = (vy_norm + 1.0) / 2.0
            b = (vz_norm + 1.0) / 2.0

            # Scale by magnitude for brightness 
            r_scaled = r * norm_mag
            g_scaled = g * norm_mag
            b_scaled = b * norm_mag

            # Clip values to [0, 1] range
            r_scaled = np.clip(r_scaled, 0.0, 1.0)
            g_scaled = np.clip(g_scaled, 0.0, 1.0)
            b_scaled = np.clip(b_scaled, 0.0, 1.0)

            # Creates 4 dimensional array with the rgb contents and the magnitude 
            rgbm_array = np.stack([r_scaled, g_scaled, b_scaled, norm_mag], axis=-1)

            # Print progress every 10 frames or for the last frame
            if i % 10 == 0 or i == lenT - 1:
                print(f"\nFrame {frame_number} ({i+1}/{lenT}):")
                print(f"  Channel shapes: R={r_scaled.shape}, G={g_scaled.shape}, B={b_scaled.shape}")
                print(f"  Value ranges: R=[{r_scaled.min():.3f}, {r_scaled.max():.3f}], "
                      f"G=[{g_scaled.min():.3f}, {g_scaled.max():.3f}], "
                      f"B=[{b_scaled.min():.3f}, {b_scaled.max():.3f}], "
                      f"M=[{norm_mag.min():.3f}, {norm_mag.max():.3f}]")

            # Save individual channels - each frame gets its own directory
            frame_dir = os.path.join(results_dir, str(frame_number))
            os.makedirs(frame_dir, exist_ok=True)
            
            # Save all components for this frame
            np.save(os.path.join(frame_dir, "optical_flow_red_channel.npy"), r_scaled)
            np.save(os.path.join(frame_dir, "optical_flow_green_channel.npy"), g_scaled)
            np.save(os.path.join(frame_dir, "optical_flow_blue_channel.npy"), b_scaled)
            np.save(os.path.join(frame_dir, "optical_flow_magnitude_channel.npy"), norm_mag)
            np.save(os.path.join(frame_dir, "optical_flow_rgbm.npy"), rgbm_array)

        except Exception as e:
            print(f"Error processing frame {frame_number}: {str(e)}")
            continue

    print(f"\n✓ Successfully processed {lenT} frames")
    print(f"✓ Saved RGBM data for frames: {min(available_frames)} to {max(available_frames)}")
    print(f"✓ Each frame directory contains:")
    print("  - optical_flow_red_channel.npy")
    print("  - optical_flow_green_channel.npy")
    print("  - optical_flow_blue_channel.npy")
    print("  - optical_flow_magnitude_channel.npy")
    print("  - optical_flow_rgbm.npy (combined 4-channel array)")

if __name__ == "__main__":
    main()