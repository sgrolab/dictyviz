import os
import subprocess
from tkinter import Tk, filedialog

def find_movie(zarr_folder, root):
    parent = os.path.dirname(zarr_folder)
    movies_dir = os.path.join(parent, "movies")

    if not os.path.isdir(movies_dir):
        print(f"Error: 'movies/' folder not found near {zarr_folder}")
        return None

    print(f"Please select a video file from: {movies_dir}")
    video_path = filedialog.askopenfilename(
        parent=root,
        initialdir=movies_dir,
        title="Select a video file",
        filetypes=[("AVI files", "*.avi")]
    )

    if not video_path or not os.path.isfile(video_path):
        print("No video file selected or invalid path.")
        return None

    return video_path

def find_opticalflow_movie(zarr_folder, root):
    parent = os.path.dirname(zarr_folder)
    opticalFlow_dir = os.path.join(parent, "optical_flow_output")

    if not os.path.isdir(opticalFlow_dir):
        print(f"Error: 'optical_flow_output/' folder not found near {zarr_folder}")
        return None

    print(f"Please select a video file from: {opticalFlow_dir}")
    video_path = filedialog.askopenfilename(
        parent=root,
        initialdir=opticalFlow_dir,
        title="Select a video file",
        filetypes=[("AVI files", "*.avi")]
    )

    if not video_path or not os.path.isfile(video_path):
        print("No video file selected or invalid path.")
        return None

    return video_path

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
        print(f"Combined video saved to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error combining videos: {e}")
        return False

if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    zarr_folder = filedialog.askdirectory(parent=root, title="Select the .zarr folder")

    if not zarr_folder:
        print("No folder selected.")
    else:
        xy_movie = find_movie(zarr_folder, root)
        opticalflow_movie = find_opticalflow_movie(zarr_folder, root)

        if xy_movie and opticalflow_movie:
            movies_dir = os.path.join(os.path.dirname(zarr_folder), "movies")
            output_path = os.path.join(movies_dir, "combined_movie.avi")
            combine_movies(xy_movie, opticalflow_movie, output_path)

    # close Tk instance after everything
    root.destroy()