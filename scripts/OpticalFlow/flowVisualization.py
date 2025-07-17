#!/usr/bin/env python3
"""
Simple example for single slice spherical flow visualization
Based on your existing optical flow results
"""

import numpy as np
import cv2
import tifffile
import os
from typing import Tuple

def simple_spherical_flow_viz(results_dir: str, frame_number: int, slice_index: int):
    """
    Simple implementation of your original plan for single slice visualization
    
    Your plan steps:
    1. Load position and flow for one frame and one slice ✓
    2. Calculate the magnitude of the flow vector ✓
    3. Calculate spherical coordinates on a UNIT sphere ✓
    4. Scale magnitude values from 0-255 ✓
    5. Find the RGB value by multiplying scaled magnitude with spherical coordinates ✓
    6. Save as a TIFF file to open in Napari ✓
    """
    
    # Step 1: Load flow data for one frame and extract one slice
    frame_dir = os.path.join(results_dir, str(frame_number))
    print(f"Loading flow data from: {frame_dir}")
    
    # Load the 3D flow components
    vx_3d = np.load(os.path.join(frame_dir, "optical_flow_vx.npy"))
    vy_3d = np.load(os.path.join(frame_dir, "optical_flow_vy.npy"))
    vz_3d = np.load(os.path.join(frame_dir, "optical_flow_vz.npy"))
    
    print(f"3D flow shape: {vx_3d.shape}")
    
    # Extract single slice
    vx = vx_3d[slice_index]
    vy = vy_3d[slice_index]
    vz = vz_3d[slice_index]
    
    print(f"Slice {slice_index} shape: {vx.shape}")
    
    # Step 2: Calculate magnitude of flow vector
    magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
    print(f"Magnitude range: {magnitude.min():.4f} - {magnitude.max():.4f}")
    
    # Step 3: Calculate spherical coordinates on UNIT sphere
    # Normalize flow vectors to unit sphere (avoid division by zero)
    magnitude_threshold = 0.001
    significant_flow = magnitude > magnitude_threshold
    
    # Initialize normalized components
    vx_norm = np.zeros_like(vx)
    vy_norm = np.zeros_like(vy) 
    vz_norm = np.zeros_like(vz)
    
    # Normalize only where we have significant flow
    vx_norm[significant_flow] = vx[significant_flow] / magnitude[significant_flow]
    vy_norm[significant_flow] = vy[significant_flow] / magnitude[significant_flow]
    vz_norm[significant_flow] = vz[significant_flow] / magnitude[significant_flow]
    
    print(f"Normalized coordinates range: x[{vx_norm.min():.3f}, {vx_norm.max():.3f}], "
          f"y[{vy_norm.min():.3f}, {vy_norm.max():.3f}], z[{vz_norm.min():.3f}, {vz_norm.max():.3f}]")
    
    # Step 4: Scale magnitude values from 0-255
    magnitude_max = np.percentile(magnitude[significant_flow], 99) if np.any(significant_flow) else 1.0
    magnitude_scaled = np.clip(magnitude / magnitude_max, 0, 1) * 255
    
    print(f"Scaled magnitude range: {magnitude_scaled.min():.1f} - {magnitude_scaled.max():.1f}")
    
    # Step 5: Find RGB value by multiplying scaled magnitude with spherical coordinates
    # Convert normalized coordinates [-1,1] to [0,255] range
    r_base = ((vx_norm + 1) * 127.5).astype(np.uint8)
    g_base = ((vy_norm + 1) * 127.5).astype(np.uint8)
    b_base = ((vz_norm + 1) * 127.5).astype(np.uint8)
    
    # Multiply by scaled magnitude to get final RGB
    magnitude_factor = magnitude_scaled / 255.0
    
    r_channel = (r_base * magnitude_factor).astype(np.uint8)
    g_channel = (g_base * magnitude_factor).astype(np.uint8)
    b_channel = (b_base * magnitude_factor).astype(np.uint8)
    
    # Combine into RGB image
    rgb_image = np.stack([r_channel, g_channel, b_channel], axis=-1)
    
    print(f"RGB image shape: {rgb_image.shape}")
    print(f"RGB ranges: R[{r_channel.min()}-{r_channel.max()}], "
          f"G[{g_channel.min()}-{g_channel.max()}], B[{b_channel.min()}-{b_channel.max()}]")
    
    # Step 6: Save as TIFF file for Napari
    output_filename = f"spherical_flow_frame{frame_number}_slice{slice_index}.tiff"
    output_path = os.path.join(frame_dir, output_filename)
    
    tifffile.imwrite(output_path, rgb_image, photometric='rgb')
    print(f"✓ Saved TIFF: {output_path}")
    
    # Also save magnitude for reference
    magnitude_filename = f"magnitude_frame{frame_number}_slice{slice_index}.tiff"
    magnitude_path = os.path.join(frame_dir, magnitude_filename)
    tifffile.imwrite(magnitude_path, magnitude.astype(np.float32))
    print(f"✓ Saved magnitude: {magnitude_path}")
    
    # Create a simple color interpretation guide
    create_color_guide(frame_dir, frame_number, slice_index)
    
    return rgb_image, magnitude

def create_color_guide(frame_dir: str, frame_number: int, slice_index: int):
    """Create a simple color interpretation guide"""
    guide_size = 300
    guide = np.zeros((guide_size, guide_size, 3), dtype=np.uint8)
    
    center = guide_size // 2
    
    # Create a grid showing color mapping
    for y in range(guide_size):
        for x in range(guide_size):
            # Map pixel position to normalized coordinates
            vx_norm = (x - center) / center  # -1 to 1
            vy_norm = (y - center) / center  # -1 to 1
            vz_norm = 0.5  # Fixed z component for this demo
            
            # Apply the same mapping as in main function
            r = int((vx_norm + 1) * 127.5)
            g = int((vy_norm + 1) * 127.5)
            b = int((vz_norm + 1) * 127.5)
            
            # Clamp values
            r = np.clip(r, 0, 255)
            g = np.clip(g, 0, 255)
            b = np.clip(b, 0, 255)
            
            guide[y, x] = [r, g, b]
    
    # Add a circle boundary
    cv2.circle(guide, (center, center), center-5, (255, 255, 255), 2)
    
    # Save guide
    guide_filename = f"color_guide_frame{frame_number}_slice{slice_index}.png"
    guide_path = os.path.join(frame_dir, guide_filename)
    cv2.imwrite(guide_path, cv2.cvtColor(guide, cv2.COLOR_RGB2BGR))
    print(f"✓ Saved color guide: {guide_path}")

def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python spherical_flow.py <results_directory> <frame_number> <slice_index>")
        print("Example: python spherical_flow.py ./optical_flow_3Dresults 141 100")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    frame_number = int(sys.argv[2])
    slice_index = int(sys.argv[3])
    
    # Validate inputs
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' does not exist")
        sys.exit(1)
    
    frame_dir = os.path.join(results_dir, str(frame_number))
    if not os.path.exists(frame_dir):
        print(f"Error: Frame directory '{frame_dir}' does not exist")
        sys.exit(1)
    
    try:
        print(f"Processing spherical flow visualization...")
        print(f"Results directory: {results_dir}")
        print(f"Frame number: {frame_number}")
        print(f"Slice index: {slice_index}")
        print("-" * 50)
        
        rgb_image, magnitude = simple_spherical_flow_viz(results_dir, frame_number, slice_index)
        
        print("-" * 50)
        print("✓ Processing complete!")
        print("\nTo view in Napari:")
        print("1. Open Napari")
        print("2. Load the generated TIFF file")
        print("3. Use the color guide to interpret flow directions")
        
        print(f"\nFlow statistics for slice {slice_index}:")
        print(f"- Total pixels: {magnitude.size}")
        print(f"- Significant flow pixels: {np.sum(magnitude > 0.001)}")
        print(f"- Mean magnitude: {magnitude.mean():.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()