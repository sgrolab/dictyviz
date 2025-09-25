import os
import numpy as np
import datetime
import torch 
import torch.nn.functional as F 

def calculate_mag_var(vx, vy, vz, window_size=40):
    
    # Convert to torch tensors and move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vx = torch.from_numpy(vx).float().to(device)
    vy = torch.from_numpy(vy).float().to(device)
    vz = torch.from_numpy(vz).float().to(device)

    # Compute magnitude volume
    magnitude = torch.sqrt(vx**2 + vy**2 + vz**2)

    # Add batch and channel dims
    vx = vx.unsqueeze(0).unsqueeze(0)
    vy = vy.unsqueeze(0).unsqueeze(0)
    vz = vz.unsqueeze(0).unsqueeze(0)
    magnitude = magnitude.unsqueeze(0).unsqueeze(0)

    # Use average pooling to compute local mean
    kernel = window_size
    stride = 1  # Set higher (e.g., 5 or 10) to speed up

    mean_mag = F.avg_pool3d(magnitude, kernel_size=kernel, stride=stride).squeeze()
    
    var_x = F.avg_pool3d(vx**2, kernel_size=kernel, stride=stride).squeeze() - \
            (F.avg_pool3d(vx, kernel_size=kernel, stride=stride).squeeze())**2
    var_y = F.avg_pool3d(vy**2, kernel_size=kernel, stride=stride).squeeze() - \
            (F.avg_pool3d(vy, kernel_size=kernel, stride=stride).squeeze())**2
    var_z = F.avg_pool3d(vz**2, kernel_size=kernel, stride=stride).squeeze() - \
            (F.avg_pool3d(vz, kernel_size=kernel, stride=stride).squeeze())**2

    total_variance = var_x + var_y + var_z

    return mean_mag.cpu().numpy(), total_variance.cpu().numpy()

def find_optimal_regions(magnitude_map, variance_map, top_k=3, suppression_radius=35):
    """
    Find top_k regions with high flow magnitude and low variance.
    Uses non-maximum suppression to avoid spatial overlap.
    """
    norm_magnitude = (magnitude_map - magnitude_map.min()) / (magnitude_map.max() - magnitude_map.min())
    norm_variance = (variance_map - variance_map.min()) / (variance_map.max() - variance_map.min())
    score_map = norm_magnitude - norm_variance

    results = []
    used_mask = np.zeros_like(score_map, dtype=bool)

    for _ in range(top_k):
        masked_score = np.ma.array(score_map, mask=used_mask)
        if masked_score.count() == 0:
            break

        max_idx = np.unravel_index(masked_score.argmax(), score_map.shape)
        depth, row, col = max_idx

        results.append((
            depth, row, col, 
            magnitude_map[depth, row, col], 
            variance_map[depth, row, col], 
            norm_magnitude[depth, row, col], 
            norm_variance[depth, row, col], 
            score_map[depth, row, col]
        ))

        # Suppress a sphere around this region
        dd, rr, cc = np.ogrid[:score_map.shape[0], :score_map.shape[1], :score_map.shape[2]]
        suppression_mask = ((dd - depth)**2 + (rr - row)**2 + (cc - col)**2) <= suppression_radius**2
        used_mask[suppression_mask] = True

    return results, score_map


def save_analysis_results(magnitude_map, variance_map, optimal_regions, frame_dir, frame_number):
    """
    Save analysis outputs including .npy maps and a human-readable .txt summary.
    """
    os.makedirs(frame_dir, exist_ok=True)
    np.save(os.path.join(frame_dir, "magnitude_map.npy"), magnitude_map)
    np.save(os.path.join(frame_dir, "variance_map.npy"), variance_map)

    regions_file = os.path.join(frame_dir, "optimal_regions.txt")
    with open(regions_file, 'w') as f:
        f.write(f"Optimal Flow Regions Analysis\n")
        f.write(f"Frame: {frame_number}")
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