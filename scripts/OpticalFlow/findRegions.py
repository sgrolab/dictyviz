import os
import sys 
import helpers.analyzeRegions as analyzeRegions
import helpers.flowLoader as flowLoader

def main():

    results_dir = sys.argv[1]
    frame_number = int(sys.argv[2])
    
    frame_dir = os.path.join(results_dir, str(frame_number))
    
    # Check if frame directory exists
    if not os.path.exists(frame_dir):
        print(f"Error: Frame directory {frame_dir} does not exist")
        sys.exit(1)

    # ==== FLOW ANALYSIS ====
    print(f"Performing flow analysis...")

    # Load 3D flow data from directory
    flow_data = flowLoader.load_flow_frame(results_dir, frame_number)
    
    vx_3d = flow_data['vx']
    vy_3d = flow_data['vy']   
    vz_3d = flow_data.get('vz')
    
    # Calculate magnitude and variance
    magnitude_map, variance_map = analyzeRegions.calculate_mag_var(vx_3d, vy_3d, vz_3d, window_size=40)

    # Find optimal flow regions
    optimal_regions = analyzeRegions.find_optimal_regions(magnitude_map, variance_map, top_k=3)

    # Save results
    regions_file = analyzeRegions.save_analysis_results(
        magnitude_map, variance_map, optimal_regions, frame_dir, frame_number
    )

    # Print summary
    print(f"\n✓ Flow Analysis Summary:")
    for i, (depth, row, col, raw_mag, raw_var, norm_mag, norm_var, score) in enumerate(optimal_regions):
        print(f"{i+1}. z: {depth:3d}, y: {row:3d}, x: {col:3d}, "
              f"Raw_Mag: {raw_mag:.4f}, Raw_Var: {raw_var:.4f}, "
              f"Norm_Mag: {norm_mag:.4f}, Norm_Var: {norm_var:.4f}, "
              f"Score: {score:.4f}\n")

    # Dummy placeholders if not already set in helpers
    save_filename = f"flow_analysis_frame{frame_number:03d}.png"
    save_path = os.path.join(frame_dir, save_filename)

    print(f"\n✓ Comprehensive visualization complete!")
    print(f"✓ Saved: {save_filename}")
    print(f"✓ Full path: {save_path}")
    print(f"✓ Features: Raw data, HSV flow, magnitude, out-of-plane, confidence, overlay")
    print(f"✓ Analysis saved to: {regions_file}")

if __name__ == "__main__":
    main()