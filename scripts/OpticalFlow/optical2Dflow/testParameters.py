
import os
import sys
from opticalFlow import compute_farneback_optical_flow, make_movie
import datetime

def run_parameters(zarr_path, cropID, base_output_dir):
    """Test different parameters for the 2d optical flow to see which parameters """
    
    # Define parameter variations to test
    parameter_sets = [
        # Test pyr_scale 
        {"name": "pyr_scale_0.3", "pyr_scale": 0.3, "levels": 10, "winsize": 7, "iterations": 8, "poly_n": 5, "poly_sigma": 1.1},
        {"name": "pyr_scale_0.5", "pyr_scale": 0.5, "levels": 10, "winsize": 7, "iterations": 8, "poly_n": 5, "poly_sigma": 1.1},
        {"name": "pyr_scale_0.7", "pyr_scale": 0.7, "levels": 10, "winsize": 7, "iterations": 8, "poly_n": 5, "poly_sigma": 1.1},
        
        # Test levels
        {"name": "levels_5", "pyr_scale": 0.5, "levels": 5, "winsize": 7, "iterations": 8, "poly_n": 5, "poly_sigma": 1.1},
        {"name": "levels_10", "pyr_scale": 0.5, "levels": 10, "winsize": 7, "iterations": 8, "poly_n": 5, "poly_sigma": 1.1},
        {"name": "levels_15", "pyr_scale": 0.5, "levels": 15, "winsize": 7, "iterations": 8, "poly_n": 5, "poly_sigma": 1.1},
        
        # Test winsize
        {"name": "winsize_5", "pyr_scale": 0.5, "levels": 10, "winsize": 5, "iterations": 8, "poly_n": 5, "poly_sigma": 1.1},
        {"name": "winsize_7", "pyr_scale": 0.5, "levels": 10, "winsize": 7, "iterations": 8, "poly_n": 5, "poly_sigma": 1.1},
        {"name": "winsize_11", "pyr_scale": 0.5, "levels": 10, "winsize": 11, "iterations": 8, "poly_n": 5, "poly_sigma": 1.1},
        {"name": "winsize_15", "pyr_scale": 0.5, "levels": 10, "winsize": 15, "iterations": 8, "poly_n": 5, "poly_sigma": 1.1},
        
        # Test iterations
        {"name": "iterations_5", "pyr_scale": 0.5, "levels": 10, "winsize": 7, "iterations": 5, "poly_n": 5, "poly_sigma": 1.1},
        {"name": "iterations_8", "pyr_scale": 0.5, "levels": 10, "winsize": 7, "iterations": 8, "poly_n": 5, "poly_sigma": 1.1},
        {"name": "iterations_12", "pyr_scale": 0.5, "levels": 10, "winsize": 7, "iterations": 12, "poly_n": 5, "poly_sigma": 1.1},
        
        # Test poly_n
        {"name": "poly_n_3", "pyr_scale": 0.5, "levels": 10, "winsize": 7, "iterations": 8, "poly_n": 3, "poly_sigma": 1.1},
        {"name": "poly_n_5", "pyr_scale": 0.5, "levels": 10, "winsize": 7, "iterations": 8, "poly_n": 5, "poly_sigma": 1.1},
        {"name": "poly_n_7", "pyr_scale": 0.5, "levels": 10, "winsize": 7, "iterations": 8, "poly_n": 7, "poly_sigma": 1.1},
        
        # Test poly_sigma
        {"name": "poly_sigma_0.8", "pyr_scale": 0.5, "levels": 10, "winsize": 7, "iterations": 8, "poly_n": 5, "poly_sigma": 0.8},
        {"name": "poly_sigma_1.1", "pyr_scale": 0.5, "levels": 10, "winsize": 7, "iterations": 8, "poly_n": 5, "poly_sigma": 1.1},
        {"name": "poly_sigma_1.5", "pyr_scale": 0.5, "levels": 10, "winsize": 7, "iterations": 8, "poly_n": 5, "poly_sigma": 1.5},
    ]
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, params in enumerate(parameter_sets):
        print(f"\n{'='*60}")
        print(f"Running parameter set {i+1}/{len(parameter_sets)}: {params['name']}")
        print(f"{'='*60}")
        
        # Create output directory with parameter name
        param_output_dir = os.path.join(base_output_dir, f"sweep_{timestamp}", params['name'])
        os.makedirs(param_output_dir, exist_ok=True)
        
        # Create log file for this parameter set
        log_path = os.path.join(param_output_dir, f"opticalFlow_{params['name']}.txt")
        
        with open(log_path, 'w') as f:
            print(f"Parameter Sweep - {params['name']}", file=f)
            print(f"Timestamp: {datetime.datetime.now()}", file=f)
            for key, value in params.items():
                if key != 'name':
                    print(f"{key}: {value}", file=f)
            print(f"zarr path: {zarr_path}", file=f)
            print(f"crop id: {cropID}", file=f)
            print("="*50, file=f)
            
            # Run optical flow with this parameter set
            compute_farneback_optical_flow(zarr_path, cropID, param_output_dir, f, params)
            
            print(f"✓ Completed: {params['name']}")
            
            # Generate movie with parameter name in filename
            movie_filename = f"optical_flow_{params['name']}.mp4"
            make_movie(param_output_dir, movie_filename)
    
    print(f"\n{'='*60}")
    print(f"Parameter sweep completed!")
    print(f"Results in: {os.path.join(base_output_dir, f'sweep_{timestamp}')}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python testParameters.py <path_to_zarr> [cropID]")
        sys.exit(1)

    zarr_path = sys.argv[1]
    cropID = sys.argv[2] if len(sys.argv) > 2 else ""
    
    base_output_dir = os.path.join(os.path.dirname(zarr_path), "optical_flow_parameter_sweep")
    os.makedirs(base_output_dir, exist_ok=True)
    
    run_parameters(zarr_path, cropID, base_output_dir)

if __name__ == "__main__":
    main()