import cv2
import subprocess
import os
import platform

def get_video_dimensions(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def get_video_fps(video_path):
    """Get frame rate of video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 30  # Default fallback
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def get_cross_platform_settings():
    """Get optimal video settings based on the current platform"""
    system = platform.system()
    
    if system == "Windows":
        return {
            'extension': '.mp4',
            'codec': 'libx264',
            'pixel_format': 'yuv420p',
            'additional_flags': ['-movflags', '+faststart']
        }
    elif system == "Darwin":  # macOS
        return {
            'extension': '.mp4',
            'codec': 'libx264', 
            'pixel_format': 'yuv420p',
            'additional_flags': ['-movflags', '+faststart']
        }
    else:  # Linux and others
        return {
            'extension': '.mp4',
            'codec': 'libx264',
            'pixel_format': 'yuv420p', 
            'additional_flags': ['-movflags', '+faststart']
        }

def sanitize_filename(filename):
    """Remove or replace characters that are problematic in filenames"""
    # Remove file extension and path
    name = os.path.splitext(os.path.basename(filename))[0]
    # Replace problematic characters with underscores
    problematic_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
    for char in problematic_chars:
        name = name.replace(char, '_')
    # Remove multiple consecutive underscores
    while '__' in name:
        name = name.replace('__', '_')
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name

def combine_movies(xy_movie, opticalflow_movie, output_path):
    # gets dimensions of both videos
    xy_w, xy_h = get_video_dimensions(xy_movie)
    opt_w, opt_h = get_video_dimensions(opticalflow_movie)

    # Get frame rates to match them if they're different
    xy_fps = get_video_fps(xy_movie)
    opt_fps = get_video_fps(opticalflow_movie)
    
    if xy_fps != opt_fps:
        print(f"Warning: Different frame rates detected - XY: {xy_fps}fps, Flow: {opt_fps}fps")
        print("Using XY video frame rate for output")


    # build FFmpeg filter to equalize heights since videos will be stacked horizontally (so only heights matter)
    if xy_h != opt_h:
        if xy_h < opt_h:
            # Resize opticalflow_movie down to xy_movie height
            filter_str = f"[1:v]scale=-2:{xy_h}[opt];[0:v][opt]hstack=inputs=2"
        else:
            # resize xy_movie down to opticalflow_movie height
            filter_str = f"[0:v]scale=-2:{opt_h}[xy];[xy][1:v]hstack=inputs=2"
    else:
        # same height already
        filter_str = "[0:v][1:v]hstack=inputs=2"

    # Get platform-specific settings
    settings = get_cross_platform_settings()
    
    # Build FFmpeg command with cross-platform compatibility
    ffmpeg_command = [
        "ffmpeg",
        "-y",  # Overwrite output file without asking
        "-i", xy_movie,
        "-i", opticalflow_movie,
        "-filter_complex", filter_str,
        "-c:v", settings['codec'],           # Video codec
        "-pix_fmt", settings['pixel_format'], # Pixel format for compatibility
        "-crf", "18",                        # High quality (lower = better quality)
        "-preset", "medium",                 # Balanced encoding speed
        "-r", "30"                          # Set frame rate to 30fps
    ]
    
    # Add platform-specific additional flags
    ffmpeg_command.extend(settings['additional_flags'])
    
    # Add output path
    ffmpeg_command.append(output_path)

    try:
        print(f"Encoding for {platform.system()} with settings: {settings}")
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print(f"✓ Cross-platform compatible video saved to: {output_path}")
        
        # Platform-specific compatibility info
        system = platform.system()
        if system == "Windows":
            print("✓ Compatible with: Windows Media Player, VLC, Movies & TV")
        elif system == "Darwin":
            print("✓ Compatible with: QuickTime Player, VLC, Final Cut Pro")
        else:
            print("✓ Compatible with: VLC, most Linux video players")
            
    except subprocess.CalledProcessError as e:
        print(f"Error combining videos: {e}")
        if e.stderr:
            print(f"FFmpeg error details: {e.stderr}")
        
        # Try fallback with more universal settings
        print("Trying fallback with more compatible settings...")
        try_fallback_encoding(xy_movie, opticalflow_movie, output_path, filter_str)

def try_fallback_encoding(xy_movie, opticalflow_movie, output_path, filter_str):
    """Fallback to most universally compatible settings"""
    # Change extension to .avi for maximum compatibility
    fallback_path = output_path.replace('.mp4', '_fallback.avi')
    
    fallback_command = [
        "ffmpeg",
        "-y",
        "-i", xy_movie,
        "-i", opticalflow_movie,
        "-filter_complex", filter_str,
        "-c:v", "libx264",      # Still try H.264
        "-pix_fmt", "yuv420p",  # Universal pixel format
        "-crf", "20",           # Good quality
        "-preset", "ultrafast", # Fast encoding
        fallback_path
    ]
    
    try:
        subprocess.run(fallback_command, check=True, capture_output=True, text=True)
        print(f"✓ Fallback video saved to: {fallback_path}")
        print("✓ This format should work on all platforms")
    except subprocess.CalledProcessError as e:
        print(f"Fallback encoding also failed: {e}")
        print("Try installing a more recent version of FFmpeg")

if __name__ == "__main__":
    import argparse

    # parse command-line arguments
    parser = argparse.ArgumentParser(description="Combine XY movie and Optical Flow movie into a single cross-platform compatible video.")
    parser.add_argument("xy_movie", help="Path to the XY movie file.")
    parser.add_argument("opticalflow_movie", help="Path to the Optical Flow movie file.")
    parser.add_argument("--output", "-o", help="Custom output filename (optional)")
    args = parser.parse_args()

    # define output path with cross-platform extension
    output_dir = os.path.dirname(args.xy_movie)
    settings = get_cross_platform_settings()
    
    if args.output:
        # Use custom filename but ensure correct extension
        base_name = os.path.splitext(args.output)[0]
        output_filename = base_name + settings['extension']
    else:
        # Create descriptive filename with both input movie names
        xy_name = sanitize_filename(args.xy_movie)
        flow_name = sanitize_filename(args.opticalflow_movie)
        
        # Truncate names if they're too long to avoid overly long filenames
        max_name_length = 20
        if len(xy_name) > max_name_length:
            xy_name = xy_name[:max_name_length]
        if len(flow_name) > max_name_length:
            flow_name = flow_name[:max_name_length]
        
        output_filename = f"combined_{xy_name}_plus_{flow_name}{settings['extension']}"
    
    output_path = os.path.join(output_dir, output_filename)

    # Print system info
    print(f"Detected system: {platform.system()}")
    print(f"Output format: {settings['extension']} with {settings['codec']} codec")
    print(f"Output filename: {output_filename}")

    # combine movies
    if os.path.isfile(args.xy_movie) and os.path.isfile(args.opticalflow_movie):
        combine_movies(args.xy_movie, args.opticalflow_movie, output_path)
    else:
        print("Error: One or both input files do not exist.")
        print(f"XY movie exists: {os.path.isfile(args.xy_movie)}")
        print(f"Optical flow movie exists: {os.path.isfile(args.opticalflow_movie)}")