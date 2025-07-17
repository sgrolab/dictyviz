import os
import numpy as np
import datetime


def calculate_mag_var(vx, vy, vz, window_size=40):
    """
    Compute 3D magnitude and variance over sliding window for each (vx, vy, vz) cube.
    Returns 3D arrays: mean magnitude map and total variance map.
    """
    depth, height, width = vx.shape
    output_height = height - window_size + 1
    output_width = width - window_size + 1
    output_depth = depth - window_size + 1

    magnitude_map = np.zeros((output_depth, output_height, output_width))
    variance_map = np.zeros((output_depth, output_height, output_width))  

    print(f"Computing magnitude and variance maps with window size {window_size}...")
    print(f"Output map size: {output_depth} x {output_height} x {output_width}")

    for i in range(output_depth):
        for j in range(output_height):
            for k in range(output_width):
                window_vx = vx[i:i+window_size , j:j+window_size, k:k+window_size]
                window_vy = vy[i:i+window_size, j:j+window_size, k:k+window_size]
                window_vz = vz[i:i+window_size, j:j+window_size, k:k+window_size]

                magnitude = np.sqrt(window_vx**2 + window_vy**2 + window_vz**2)

                var_x = np.var(window_vx)
                var_y = np.var(window_vy)
                var_z = np.var(window_vz)
                total_variance = var_x + var_y + var_z

                magnitude_map[i, j, k] = np.mean(magnitude)  
                variance_map[i, j, k] = total_variance      

    return magnitude_map, variance_map


def find_optimal_regions(magnitude_map, variance_map, top_k=3, suppression_radius=35):
    """
    Find top_k regions with high flow magnitude and low variance.
    Uses non-maximum suppression to avoid spatial overlap.
    """
    norm_magnitude = (magnitude_map - magnitude_map.min()) / (magnitude_map.max() - magnitude_map.min())
    norm_variance = (variance_map - variance_map.min()) / (variance_map.max() - variance_map.min())
    score = norm_magnitude - norm_variance

    results = []
    used_mask = np.zeros_like(score, dtype=bool)

    for _ in range(top_k):
        masked_score = np.ma.array(score, mask=used_mask)
        if masked_score.count() == 0:
            break

        max_idx = np.unravel_index(masked_score.argmax(), score.shape)
        depth, row, col = max_idx

        results.append((
            depth, row, col, 
            magnitude_map[depth, row, col], 
            variance_map[depth, row, col], 
            norm_magnitude[depth, row, col], 
            norm_variance[depth, row, col], 
            score[depth, row, col]
        ))

        # Suppress a sphere around this region
        dd, rr, cc = np.ogrid[:score.shape[0], :score.shape[1], :score.shape[2]]
        suppression_mask = ((dd - depth)**2 + (rr - row)**2 + (cc - col)**2) <= suppression_radius**2
        used_mask[suppression_mask] = True

    return results


def save_analysis_results(magnitude_map, variance_map, optimal_regions, frame_dir, frame_number, slice_idx):
    """
    Save analysis outputs including .npy maps and a human-readable .txt summary.
    """
    os.makedirs(frame_dir, exist_ok=True)
    np.save(os.path.join(frame_dir, "magnitude_map.npy"), magnitude_map)
    np.save(os.path.join(frame_dir, "variance_map.npy"), variance_map)

    regions_file = os.path.join(frame_dir, "optimal_regions.txt")
    with open(regions_file, 'w') as f:
        f.write(f"Optimal Flow Regions Analysis\n")
        f.write(f"Frame: {frame_number}, Z-slice: {slice_idx}\n")
        f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Top regions (z, y, x, raw_magnitude, raw_variance, norm_magnitude, norm_variance, score):\n")

        for i, (depth, row, col, raw_mag, raw_var, norm_mag, norm_var, score) in enumerate(optimal_regions):
            f.write(f"{i+1}. z: {depth:3d}, y: {row:3d}, x: {col:3d}, "
                    f"Raw_Mag: {raw_mag:.4f}, Raw_Var: {raw_var:.4f}, "
                    f"Norm_Mag: {norm_mag:.4f}, Norm_Var: {norm_var:.4f}, "
                    f"Score: {score:.4f}\n")

    print(f"✓ Analysis results saved:")
    print(f"  - magnitude_map.npy") 
    print(f"  - variance_map.npy") 
    print(f"  - optimal_regions.txt")

    return regions_file