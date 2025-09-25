import os
import sys 
import cv2
import cmapy
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from OpticalFlow.helpers import analyzeRegions
from OpticalFlow.helpers import flowLoader

def main():
    results_dir = sys.argv[1]
    frame_avg = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
    z_slice = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    # Get all frame directories (numeric names)
    frame_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if frame_avg:
        # find frame directories that contain a avg_flow_frame_*.npy file
        frame_dirs = [d for d in frame_dirs if any(f.startswith('avg_flow_frame_') for f in os.listdir(os.path.join(results_dir, d)))]
    # remove non-numeric directories
    frame_dirs = [d for d in frame_dirs if d.isdigit()]
    frame_dirs.sort(key=int)  # Sort numerically by frame number
    print(f"Found {len(frame_dirs)} frames in directory: {results_dir}")
    if not frame_dirs:
        print(f"No frame directories found in: {results_dir}")
        sys.exit(1)

    all_regions = []

    for frame_str in frame_dirs:
        frame_number = int(frame_str)
        frame_dir = os.path.join(results_dir, frame_str)
        print(f"\nAnalyzing frame {frame_number}...")

        # Check if score map already exists
        score_map_file = os.path.join(frame_dir, f"flow_score_map_frame_{frame_number}.npy")
        if os.path.exists(score_map_file):
            print(f"Score map already exists for frame {frame_number}. Skipping analysis.")
            continue

        # Load 3D flow data from directory
        if frame_avg:
            flow_data = flowLoader.load_average_flow_frame(results_dir, frame_number)
        else:
            flow_data = flowLoader.load_flow_frame(results_dir, frame_number)

        vx_3d = flow_data['vx']
        vy_3d = flow_data['vy']   
        vz_3d = flow_data.get('vz')

        # Calculate magnitude and variance
        magnitude_map, variance_map = analyzeRegions.calculate_mag_var(vx_3d, vy_3d, vz_3d, window_size=40)

        # Find optimal flow regions for this frame
        optimal_regions, score_map = analyzeRegions.find_optimal_regions(magnitude_map, variance_map, top_k=3)

        # Save results for this frame
        analyzeRegions.save_analysis_results(
            magnitude_map, variance_map, optimal_regions, frame_dir, frame_number
        )

        # Add frame number to each region and collect
        for region in optimal_regions:
            all_regions.append((frame_number,) + region)

        # Save score map for this frame
        np.save(score_map_file, score_map)

        del score_map

    # Find top 3 regions across all frames by score
    all_regions_sorted = sorted(all_regions, key=lambda x: x[-1], reverse=True)
    top3_overall = all_regions_sorted[:3]

    # Save summary in the main results directory
    summary_file = os.path.join(results_dir, "top3_flow_regions_overall.txt")
    with open(summary_file, 'w') as f:
        f.write("Top 3 Flow Regions Across All Frames\n")
        f.write("Frame | z | y | x | Raw_Mag | Raw_Var | Norm_Mag | Norm_Var | Score\n")
        for i, region in enumerate(top3_overall):
            frame_number, depth, row, col, raw_mag, raw_var, norm_mag, norm_var, score = region
            f.write(f"{i+1}. Frame: {frame_number}, z: {depth}, y: {row}, x: {col}, "
                    f"Raw_Mag: {raw_mag:.4f}, Raw_Var: {raw_var:.4f}, "
                    f"Norm_Mag: {norm_mag:.4f}, Norm_Var: {norm_var:.4f}, "
                    f"Score: {score:.4f}\n")
    print(f"\n✓ Top 3 flow regions across all frames saved to: {summary_file}")

    # Generate movie of score maps for each frame at a single z slice
    score_map = np.load(score_map_file)
    width = score_map.shape[2]
    height = score_map.shape[1]

    movie_filename = os.path.join(results_dir, f"flow_score_movie_{z_slice}.mp4")
    vid = cv2.VideoWriter(movie_filename, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for frame_str in frame_dirs:
        frame_number = int(frame_str)
        frame_dir = os.path.join(results_dir, frame_str)
        print(f"\nPlotting frame {frame_number}...")

        score_map_file = os.path.join(frame_dir, f"flow_score_map_frame_{frame_number}.npy")
        score_map = np.load(score_map_file)
        score_map = score_map[z_slice]

        # Convert to 8-bit grayscale
        score_map_frame = cv2.normalize(score_map, None, 0, 255, cv2.NORM_MINMAX)
        score_map_frame = cv2.applyColorMap(score_map_frame.astype('uint8'), cmapy.cmap('viridis'))

        vid.write(score_map_frame)
        del score_map
        del score_map_frame

    vid.release()
    print(f"✓ Flow score movie saved to: {movie_filename}")

if __name__ == "__main__":
    main()