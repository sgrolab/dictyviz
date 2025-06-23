#/bin/bash

# Prompt the user to select a folder containing a zarr file
zarr_folder=$(zenity --file-selection --directory --title="Select a zarr folder" --filename="$(pwd)/")

# Check if the user canceled the dialog
if [ -z "$zarr_folder" ]; then
    echo "No folder selected. Exiting."
    exit 1
fi

# submit optical flow job if flow_raw.npy doesn't exist
if [ -f "$flow_raw_file" ]; then
    echo "Optical flow already computed at: $flow_raw_file"
else
    echo "Submitting job to compute optical flow..."
    bsub -n 8 -W 01:00 -K python3 ./opticalFlow.py "${zarr_folder}"
fi

# submit movie creation job if movie doesn't exist
if [ -f "$movie_file" ]; then
    echo "Optical flow movie already exists at: $movie_file"
else
    echo "Submitting job to create optical flow movie..."
    bsub -n 8 -W 01:00 python3 ./makeOpticalFlowMovie.py "${zarr_folder}"
fi
