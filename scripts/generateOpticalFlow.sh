#/bin/bash

# Prompt the user to select a folder containing a zarr file
zarr_folder=$(zenity --file-selection --directory --title="Select a zarr folder" --filename="$(pwd)/")

# Check if the user canceled the dialog
if [ -z "$zarr_folder" ]; then
    echo "No folder selected. Exiting."
    exit 1
fi

#define path variables 
parent_directory=$(dirname "$zarr_folder")
optical_output_folder="${parent_directory}/optical_flow_output"
flow_file="${optical_output_folder}/flow_raw.npy"
movie_file="${optical_output_folder}/optical_flow_movie.avi"

# submit optical flow job if flow_raw.npy doesn't exist
if [ -f "$flow_file" ]; then
    echo "Optical flow already computed at: $flow_raw_file"
else
    echo "Submitting job to compute optical flow..."
    bsub -n 8 -W 01:00 -K python3 OpticalFlow/AVIOpticalFlow.py "${zarr_folder}"
fi

#submit movie creation job if movie doesn't exist
if [ -f "$movie_file" ]; then
   echo "Optical flow movie already exists at: $movie_file"
else
   echo "Submitting job to create optical flow movie..."
   bsub -n 8 -W 01:00 python3 OpticalFlow/AVIMakeOpticalFlowMovie.py "${zarr_folder}"
fi
