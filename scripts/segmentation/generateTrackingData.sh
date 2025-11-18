#!/bin/bash

# Prompt user for the zarr dataset to analyze
RESULTS_DIR=$(zenity --file-selection --directory --title="Select zarr directory" --filename="$(pwd)/")

# Let user select a Z range
Z_RANGE=$(zenity --entry --title="Select Z range" --text="Enter slice range (e.g. 0-100):")
# If the user cancels, exit the script
if [ $? -ne 0 ]; then
    echo "User cancelled the selection."
    exit 1
fi
# Split the z range into start and end
IFS='-' read -r START_Z END_Z <<< "$Z_RANGE"
# Validate the z range
if ! [[ "$START_Z" =~ ^[0-9]+$ ]] || ! [[ "$END_Z" =~ ^[0-9]+$ ]] || [ "$START_Z" -ge "$END_Z" ]; then
    zenity --error --text="Invalid z range. Please enter a valid range like 0-100."
    exit 1
fi

#Submit the cell shadow removal job to the cluster
bsub -K -J "remove_cell_shadows" -n 12 -gpu "num=1" -q gpu_h200 -W 24:00 python3 removeCellShadows.py "$RESULTS_DIR" "$START_Z" "$END_Z"

# Submit the tracking job to the cluster
bsub -J "segment_track_rocks" -n 16 -W 24:00 python3 segmentTrackRocks3D.py "$RESULTS_DIR" "$START_Z" "$END_Z"
