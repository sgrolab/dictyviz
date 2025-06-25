import os
import subprocess
from tkinter import Tk, filedialog

def select_video(prompt):
    video_path = filedialog.askopenfilename(
        title=prompt,
        filetypes=[("AVI files", "*.avi"), ("All files", "*.*")]
    )
    if not video_path or not os.path.isfile(video_path):
        print(f"No valid video selected for: {prompt}")
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
        print(f"Combined video saved to: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error combining videos: {e}")
        return False

if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    print("Select the XY movie:")
    xy_movie = select_video("Select XY movie")
    print("Select the Optical Flow movie:")
    opticalflow_movie = select_video("Select Optical Flow movie")

    if xy_movie and opticalflow_movie:
        output_dir = os.path.dirname(xy_movie)
        output_path = os.path.join(output_dir, "combined_movie.avi")
        combine_movies(xy_movie, opticalflow_movie, output_path)

    root.destroy()