#!/bin/bash

# Prompt the user to select movie files
xy_movie=$(zenity --file-selection --title="Select an XY movie file" --filename="$(pwd)/")
optical_movie=$(zenity --file-selection --title="Select an Optical Flow movie file" --filename="$(pwd)/")

# Check if user canceled either dialog
if [ -z "$xy_movie" ] || [ -z "$optical_movie" ]; then
    echo "No movie file selected. Exiting."
    exit 1
fi

# Define output path
parent_directory=$(dirname "$xy_movie")
movie_folder="${parent_directory}/movies"
combined_movie="${movie_folder}/combined_movie.avi"

# Submit job if combined movie doesn't already exist
if [ -f "$combined_movie" ]; then
    echo "Combined movie already exists at: $combined_movie"
else
    echo "Submitting job to create combined movie..."
    bsub -n 8 -W 01:00 python3 combined_movie.py "$xy_movie" "$optical_movie"
fi