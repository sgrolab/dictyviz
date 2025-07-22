import numpy as np
import sys
import os
import helpers.flowLoader as flowLoader

def main():
    results_dir = sys.argv[1]
    frame_number = int(sys.argv[2])

    # Load flow
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

    print(f"Individual channel shapes: R={r_scaled.shape}, G={g_scaled.shape}, B={b_scaled.shape}")
    print(f"Value ranges: R=[{r_scaled.min():.3f}, {r_scaled.max():.3f}], "
          f"G=[{g_scaled.min():.3f}, {g_scaled.max():.3f}], "
          f"B=[{b_scaled.min():.3f}, {b_scaled.max():.3f}]")

    # Save individual channels
    frame_dir = os.path.join(results_dir, str(frame_number))
    os.makedirs(frame_dir, exist_ok=True)
    
    np.save(os.path.join(frame_dir, "optical_flow_red_channel.npy"), r_scaled)
    np.save(os.path.join(frame_dir, "optical_flow_green_channel.npy"), g_scaled)
    np.save(os.path.join(frame_dir, "optical_flow_blue_channel.npy"), b_scaled)

    print(f"✓ Saved individual RGB channels to: {frame_dir}")

if __name__ == "__main__":
    main()