import cv2
import subprocess
import os

def get_video_dimensions(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def combine_movies(xy_movie, opticalflow_movie, output_path):
    # gets dimensions of both videos
    xy_w, xy_h = get_video_dimensions(xy_movie)
    opt_w, opt_h = get_video_dimensions(opticalflow_movie)

    # build FFmpeg filter to equalize heights
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

    ffmpeg_command = [
        "ffmpeg",
        "-i", xy_movie,
        "-i", opticalflow_movie,
        "-filter_complex", filter_str,
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        output_path
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"Combined video saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error combining videos: {e}")

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