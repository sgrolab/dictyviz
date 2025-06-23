import sys
import os
import cv2

def make_movie_from_frames(output_dir, output_filename="optical_flow_movie.avi", fps=10):
    # get list of all PNG frames in output_dir
    frames = sorted([f for f in os.listdir(output_dir) if f.startswith("flow_") and f.endswith(".png")])
    
    if not frames:
        print(f"No flow PNG frames found in: {output_dir}")
        return

    # read the first frame to get video dimensions
    first_frame_path = os.path.join(output_dir, frames[0])
    frame = cv2.imread(first_frame_path)
    height, width, _ = frame.shape

    # set up video writer
    output_path = os.path.join(output_dir, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for fname in frames:
        img = cv2.imread(os.path.join(output_dir, fname))
        writer.write(img)

    writer.release()
    print(f"Movie saved to: {output_path}")

def main(zarr_folder):
    parent_dir = os.path.dirname(zarr_folder)
    output_dir = os.path.join(parent_dir, "optical_flow_output")

    if not os.path.isdir(output_dir):
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)

    make_movie_from_frames(output_dir)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 makeOpticalFlowMovie.py <zarr_folder>")
        sys.exit(1)

    zarr_folder = sys.argv[1]
    main(zarr_folder)