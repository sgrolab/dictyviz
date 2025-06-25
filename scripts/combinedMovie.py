import os
import subprocess

def combine_movies(xy_movie, opticalflow_movie, output_path):
    ffmpeg_command = [
        "ffmpeg",
        "-i", xy_movie,
        "-i", opticalflow_movie,
        "-filter_complex", "[0:v][1:v]hstack=inputs=2",
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        output_path
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"Combined video saved to: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error combining videos: {e}")
        return False

if __name__ == "__main__":
    import argparse

    # parse command-line arguments
    parser = argparse.ArgumentParser(description="Combine XY movie and Optical Flow movie into a single video.")
    parser.add_argument("xy_movie", help="Path to the XY movie file.")
    parser.add_argument("opticalflow_movie", help="Path to the Optical Flow movie file.")
    args = parser.parse_args()

    # define output path
    output_dir = os.path.dirname(args.xy_movie)
    output_path = os.path.join(output_dir, "combined_movie.avi")

    # combine movies
    if os.path.isfile(args.xy_movie) and os.path.isfile(args.opticalflow_movie):
        combine_movies(args.xy_movie, args.opticalflow_movie, output_path)
    else:
        print("Error: One or both input files do not exist.")