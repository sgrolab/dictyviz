#!/bin/bash

# Prompt the user to select a max projection folder
zarr_folder=$(zenity --file-selection --directory --title="Select a zarr folder" --filename="$(pwd)/")

# Check if the user canceled the dialog
if [ -z "$zarr_folder" ]; then
    echo "No folder selected. Exiting."
    exit 1
fi

# Ask user for channel
channel=$(zenity --entry --title="Channel" --text="Enter channel (0 or 1):" --entry-text="")

# Ask user for optional cropID
cropID=$(zenity --entry --title="Crop ID" --text="Enter crop ID (leave empty if none):" --entry-text="")

# Define path variables 
parent_directory=$(dirname "$zarr_folder")
optical_output_folder="${parent_directory}/optical_flow_output"
flow_file="${optical_output_folder}/flow_raw.npy"
movie_file="${optical_output_folder}/optical_flow_movie.avi"

# Submit optical flow job if flow_raw.npy and optical_flow_movie.avi don't exist
if [ -f "$flow_file" ] && [ -f "$movie_file" ]; then
    echo "Optical flow already computed at: $flow_file"
else
    echo "Submitting job to compute optical flow..."
    if [ -z "$cropID" ]; then
        # No crop ID provided
        bsub -n 8 -W 24:00 python3 opticalFlow.py "${zarr_folder}" "${channel}"
    else
        # Crop ID provided
        bsub -n 8 -W 24:00 python3 opticalFlow.py "${zarr_folder}" "${channel}" "${cropID}"
    fi
fi 

