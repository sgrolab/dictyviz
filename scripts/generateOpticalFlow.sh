#/bin/bash

# Prompt the user to select an AVI file
raw_data=$(zenity --file-selection --title="Select a max projection folder" --filename="$(pwd)/")

# Check if the user canceled the dialog
if [ -z "$raw_data" ]; then
    echo "No file selected. Exiting."
    exit 1
fi

#define path variables 
parent_directory=$(dirname "$raw_data")
optical_output_folder="${parent_directory}/optical_flow_output"
flow_file="${optical_output_folder}/flow_raw.npy"
movie_file="${optical_output_folder}/optical_flow_movie.avi"

# submit optical flow job if flow_raw.npy and optical_flow_movie.avi doesn't exist
if [ -f "$flow_file" ] && [ -f "$movie_file" ]; then
    echo "Optical flow already computed at: $flow_raw_file"
else
    echo "Submitting job to compute optical flow..."
    bsub -n 8 -W 01:00 python3 OpticalFlow/opticalFlow.py "${raw_data}"
fi