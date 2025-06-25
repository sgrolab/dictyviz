#!/bin/bash

# Prompt the user to select a folder containing a zarr file
xy_movie=$(zenity --file-selection --title="Select an xy movie file" --filename="$(pwd)/")
optical_movie=$(zenity --file-selection --title="Select an optical flow movie file" --filename="$(pwd)/")

# Check if the user canceled the dialog
if [ -z "$zarr_folder" ]; then
    echo "No folder selected. Exiting."
    exit 1
fi

# Define path variables
parent_directory=$(dirname "$xy_movie")
movie_folder="${parent_directory}/movies"
combined_movie="${movie_folder}/combined_movie.avi"

# Submit combinedMovie job if the combined movie doesn't exist
if [ -f "$combined_movie" ]; then
    echo "Combined movie already computed at: $combined_movie"
else
    echo "Submitting job to compute optical flow..."
    bsub -n 8 -W 01:00 python3 combinedMovie.py "${xy_movie}"
fi