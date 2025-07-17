import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import matplotlib.pyplot as plt
import argparse
import datetime
import zarr 
import helpers.analyzeRegions as analyzeRegions
import helpers.flowLoader as flowLoader

def create_hsv_flow(vx, vy, max_flow=None):

    # Calculate magnitude and angle like in opticalFlow.py
    mag, ang = cv2.cartToPolar(vx, vy)
    
    # Determine scaling factor
    if max_flow is None:
        max_flow = np.percentile(mag, 99)  # Use 99th percentile to avoid outliers

    # Create HSV image (same approach as opticalFlow.py)
    height, width = vx.shape
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue based on angle
    hsv[..., 1] = 255  # Set saturation to maximum
    hsv[..., 2] = np.clip(mag * (255/max_flow), 0, 255).astype(np.uint8)  # Value based on magnitude
    
    # Convert HSV to RGB
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb_flow, mag, max_flow

def plot_flow(vx, vy, vz, conf, raw_slice, axis, slice_idx, frame_number, save_path=None, show_arrows=True, arrow_step=10):
    """Create comprehensive flow visualization with raw data"""
    
    # Create HSV flow visualization using OpenCV
    rgb, magnitude, max_flow = create_hsv_flow(vx, vy)

    # Create 2x3 subplot layout to include raw data
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'3D Optical Flow: Frame {frame_number}, {axis.upper()} Slice @ {slice_idx}', fontsize=16, fontweight='bold')
    
    # Plot 1: Raw data image
    if raw_slice is not None:
        # Normalize raw data for display
        vmin = np.percentile(raw_slice, 5)
        vmax = np.percentile(raw_slice, 95)
        raw_display = np.clip(raw_slice, vmin, vmax)
        raw_display = (raw_display - vmin) / (vmax - vmin)  

        axs[0, 0].imshow(raw_display, cmap='gray', origin='lower')
        axs[0, 0].set_title("Raw Image Data", fontsize=14)
    else:
        axs[0, 0].text(0.5, 0.5, 'No Raw Data\nAvailable', ha='center', va='center', 
                      transform=axs[0, 0].transAxes, fontsize=12)
        axs[0, 0].set_title("Raw Image Data", fontsize=14)
    
    # Plot 2: HSV flow visualization
    axs[0, 1].imshow(rgb, origin='lower')
    axs[0, 1].set_title("HSV Flow (Hue=Direction, Saturation=Speed)", fontsize=14)
    if show_arrows:
        y, x = np.mgrid[0:vx.shape[0]:arrow_step, 0:vx.shape[1]:arrow_step]
        if max_flow > 0:
            axs[0, 1].quiver(x, y, vx[::arrow_step, ::arrow_step]/max_flow*10, vy[::arrow_step, ::arrow_step]/max_flow*10,
                           color='white', scale=1, scale_units='xy', angles='xy', alpha=0.7, width=0.002)


    # Plot 3: Flow magnitude
    im2 = axs[0, 2].imshow(magnitude, cmap='viridis', origin='lower')
    axs[0, 2].set_title(f'Magnitude (max={max_flow:.3f})', fontsize=14)
    plt.colorbar(im2, ax=axs[0, 2], shrink=0.8)

    # Plot 4: vz plot
    if vz is not None:
        vz_max = np.max(np.abs(vz))
        im3 = axs[1, 0].imshow(vz, cmap='RdBu_r', origin='lower', vmin=-vz_max, vmax=vz_max)
        axs[1, 0].set_title(f'Out-of-plane Flow (v{axis})', fontsize=14)
        plt.colorbar(im3, ax=axs[1, 0], shrink=0.8)
    else:
        axs[1, 0].text(0.5, 0.5, 'No vz data', ha='center', va='center', 
                      transform=axs[1, 0].transAxes, fontsize=12)
        axs[1, 0].set_title(f'Out-of-plane Flow (v{axis})', fontsize=14)
    
    # Plot 5: Confidence
    if conf is not None:

        vmin = np.percentile(conf, 5)
        vmax = np.percentile(conf, 95)

        im4 = axs[1, 1].imshow(conf, cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)

        axs[1, 1].set_title('Flow Confidence', fontsize=14)
        plt.colorbar(im4, ax=axs[1, 1], shrink=0.8)
    else:
        axs[1, 1].text(0.5, 0.5, 'No confidence data', ha='center', va='center',
                      transform=axs[1, 1].transAxes, fontsize=12)
        axs[1, 1].set_title('Flow Confidence', fontsize=14)
    
    # Plot 6: Enhanced Flow overlay on raw data
    if raw_slice is not None:
        # Enhanced background: use contrast-enhanced raw data
        raw_enhanced = np.power(raw_display, 0.7)  # Gamma correction for better contrast
        axs[1, 2].imshow(raw_enhanced, cmap='gray', origin='lower', alpha=0.8)
        
        # Create a selective overlay: only show significant flow
        flow_threshold = np.percentile(magnitude, 75)  # Only show top 25% of flow
        significant_flow_mask = magnitude > flow_threshold
        
        # Apply mask to HSV flow for selective overlay
        rgb_selective = rgb.copy()
        rgb_selective[~significant_flow_mask] = [0, 0, 0]  # Make low-flow areas transparent
        
        # Create alpha channel based on flow magnitude for smooth blending
        alpha_flow = np.zeros_like(magnitude)
        alpha_flow[significant_flow_mask] = 0.6 * (magnitude[significant_flow_mask] / np.max(magnitude))
        
        # Overlay significant flow with variable transparency
        axs[1, 2].imshow(rgb_selective, origin='lower', alpha=0.4)
        
        if show_arrows and max_flow > 0:
            # Adaptive arrow density based on local flow magnitude
            arrow_step_adaptive = arrow_step * 3  # Start with sparser grid
            y, x = np.mgrid[0:vx.shape[0]:arrow_step_adaptive, 0:vx.shape[1]:arrow_step_adaptive]
            
            # Sample flow at arrow positions
            vx_arrows = vx[::arrow_step_adaptive, ::arrow_step_adaptive]
            vy_arrows = vy[::arrow_step_adaptive, ::arrow_step_adaptive]
            mag_arrows = magnitude[::arrow_step_adaptive, ::arrow_step_adaptive]
            
            # Only show arrows where flow is significant
            significant_arrows = mag_arrows > flow_threshold
            
            if np.any(significant_arrows):
                # Scale arrows based on local magnitude and make them more visible
                scale_factor = 15.0 / max_flow
                arrow_colors = []
                arrow_widths = []
                
                for i in range(len(y.flat)):
                    if significant_arrows.flat[i]:
                        # Color arrows based on magnitude: cyan for medium, yellow for high
                        mag_norm = mag_arrows.flat[i] / max_flow
                        if mag_norm > 0.8:
                            arrow_colors.append('yellow')
                            arrow_widths.append(0.004)
                        elif mag_norm > 0.5:
                            arrow_colors.append('cyan')
                            arrow_widths.append(0.003)
                        else:
                            arrow_colors.append('white')
                            arrow_widths.append(0.002)
                    else:
                        arrow_colors.append('none')
                        arrow_widths.append(0.001)
                
                # Plot arrows with adaptive colors and sizes
                for i in range(len(y.flat)):
                    if significant_arrows.flat[i] and arrow_colors[i] != 'none':
                        yi, xi = y.flat[i], x.flat[i]
                        vxi, vyi = vx_arrows.flat[i] * scale_factor, vy_arrows.flat[i] * scale_factor
                        axs[1, 2].arrow(xi, yi, vxi, vyi, head_width=2, head_length=2,
                                      fc=arrow_colors[i], ec=arrow_colors[i], alpha=0.9,
                                      linewidth=arrow_widths[i]*1000)
        
        # Add flow statistics overlay
        flow_stats = f'Significant Flow: {np.sum(significant_flow_mask)}/{magnitude.size} pixels\nThreshold: {flow_threshold:.3f}\nMax: {max_flow:.3f}'
        axs[1, 2].text(0.02, 0.98, flow_stats, transform=axs[1, 2].transAxes, 
                      fontsize=9, verticalalignment='top', color='white',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        axs[1, 2].set_title('Flow Overlay' , fontsize=14)
    else:
        axs[1, 2].text(0.5, 0.5, 'No Raw Data\nfor Overlay', ha='center', va='center',
                      transform=axs[1, 2].transAxes, fontsize=12)
        axs[1, 2].set_title('Flow Overlay', fontsize=14)
    
    # Set axis labels for all plots
    for i, ax in enumerate(axs.flat):
        ax.set_xlabel('X axis (pixels)', fontsize=12)
        ax.set_ylabel('Y axis (pixels)', fontsize=12)
        ax.tick_params(labelsize=10)
    
    # Add information text
    info_text = f"Data: {vx.shape[1]}×{vx.shape[0]} pixels | Slice: {slice_idx} | Arrow step: {arrow_step}"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=11, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.07)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved to {save_path}")
    
    plt.close()  # Close figure to free memory

def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize_3D_flow.py <results_directory> <frame_number> [slice_index]")
        print("  slice_index: optional Z-slice index (default: middle slice)")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    frame_number = int(sys.argv[2])
    
    # Parse optional slice index
    slice_index = None
    if len(sys.argv) >= 4:
        slice_index = int(sys.argv[3])
        print(f"Using user-specified slice index: {slice_index}")
    
    # Default settings
    arrow_step = 5
    
    print(f"Visualizing frame {frame_number} from {results_dir}")
    print(f"Using Z-axis slice, Arrow step: {arrow_step}")
    
    # Validate inputs
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' does not exist.")
        sys.exit(1)
    
    try:
        # Load flow data
        print("Loading optical flow data...")
        flow_data = flowLoader.load_flow_frame(results_dir, frame_number)
        
        if not flow_data:
            raise ValueError("No flow data loaded")
        
        # Load raw data for comparison
        print("Loading raw image data...")
        raw_data = flowLoader.load_raw_data(results_dir, frame_number)
        
        # Extract slice
        print(f"Extracting Z slice...")
        vx, vy, vz, conf, raw_slice, idx = flowLoader.extract_slice(
            flow_data, raw_data, idx=slice_index
        )
        
        # Create output filename in the frame directory
        frame_dir = os.path.join(results_dir, str(frame_number))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        slice_type = "custom" if slice_index is not None else "middle"
        save_filename = f"flow_visualization_frame{frame_number}_z{idx}_{slice_type}_{timestamp}.png"
        save_path = os.path.join(frame_dir, save_filename)
        
        print(f"Generating comprehensive visualization...")
        print(f"Frame: {frame_number}")
        print(f"Slice: Z-axis at index {idx} ({slice_type})")
        print(f"Data shape: {vx.shape}")
        print(f"Raw data: {'Available' if raw_slice is not None else 'Not available'}")
        print(f"Output: {save_path}")
        
        # Create visualization
        plot_flow(vx, vy, vz, conf, raw_slice, 'z', idx, frame_number,
                  save_path=save_path, show_arrows=True, arrow_step=arrow_step)
        
        # Perform magnitude/variance analysis
        print(f"Performing flow analysis...")
       
        #vx_3d = flow_data['vx']s
        #vy_3d = flow_data['vy']   
        #vz_3d = flow_data.get('vz')

        magnitude_map, variance_map = analyzeRegions.calculate_mag_var(vx, vy, vz, window_size=40)
        optimal_regions = analyzeRegions.find_optimal_regions(magnitude_map, variance_map, top_k=3)
        
        # Save analysis results to frame directory
        regions_file = analyzeRegions.save_analysis_results(magnitude_map, variance_map, optimal_regions, 
                                           frame_dir, frame_number, idx)
        
        # Print summary of optimal regions
        print(f"\n✓ Flow Analysis Summary:")
        for i, (depth, row, col, raw_mag, raw_var, norm_magnitude, norm_variance, score) in enumerate(optimal_regions):
            print(f"{i+1}. z: {depth:3d}, y: {row:3d}, x: {col:3d}, "
                   f"Raw_Mag: {raw_mag:.4f}, Raw_Var: {raw_var:.4f}, "
                   f"Norm_Mag: {norm_magnitude:.4f}, Norm_Var: {norm_variance:.4f}, "
                   f"Score: {score:.4f}\n")
        
        print(f"\n✓ Comprehensive visualization complete!")
        print(f"✓ Saved: {save_filename}")
        print(f"✓ Full path: {save_path}")
        print(f"✓ Features: Raw data, HSV flow, magnitude, out-of-plane, confidence, overlay")
        print(f"✓ Analysis saved to: {regions_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()