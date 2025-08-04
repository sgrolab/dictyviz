import os
import sys 
from .helpers import analyzeRegions
from .helpers import flowLoader

def main():
    results_dir = sys.argv[1]

    # Get all frame directories (numeric names)
    frame_list = sorted([d for d in os.listdir(results_dir) if d.isdigit()])
    if not frame_list:
        print(f"No frame directories found in: {results_dir}")
        sys.exit(1)

    all_regions = []

    for frame_str in frame_list:
        frame_number = int(frame_str)
        frame_dir = os.path.join(results_dir, frame_str)
        print(f"\nAnalyzing frame {frame_number}...")

        # Load 3D flow data from directory
        flow_data = flowLoader.load_flow_frame(results_dir, frame_number)
        vx_3d = flow_data['vx']
        vy_3d = flow_data['vy']   
        vz_3d = flow_data.get('vz')

        # Calculate magnitude and variance
        magnitude_map, variance_map = analyzeRegions.calculate_mag_var(vx_3d, vy_3d, vz_3d, window_size=40)

        # Find optimal flow regions for this frame
        optimal_regions = analyzeRegions.find_optimal_regions(magnitude_map, variance_map, top_k=3)

        # Save results for this frame
        analyzeRegions.save_analysis_results(
            magnitude_map, variance_map, optimal_regions, frame_dir, frame_number
        )

        # Add frame number to each region and collect
        for region in optimal_regions:
            all_regions.append((frame_number,) + region)

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

if __name__ == "__main__":
    main()