#!/bin/bash

# Prompt user for the zarr dataset to analyze
RESULTS_DIR=$(zenity --file-selection --directory --title="Select zarr directory" --filename="$(pwd)/")

# Let user select a frame range
# TODO: eventually replace with selecting z range and calculate for all t
FRAME_RANGE=$(zenity --entry --title="Select frame range" --text="Enter frame range (e.g. 0-100):")
# If the user cancels, exit the script
if [ $? -ne 0 ]; then
    echo "User cancelled the selection."
    exit 1
fi
# Split the frame range into start and end
IFS='-' read -r START_FRAME END_FRAME <<< "$FRAME_RANGE"
# Validate the frame range
if ! [[ "$START_FRAME" =~ ^[0-9]+$ ]] || ! [[ "$END_FRAME" =~ ^[0-9]+$ ]] || [ "$START_FRAME" -ge "$END_FRAME" ]; then
    zenity --error --text="Invalid frame range. Please enter a valid range like 0-100."
    exit 1
fi

#Submit the cell shadow removal job to the cluster
bsub -K -J "remove_cell_shadows" -n 8 -gpu "num=1" -q gpu_h200 -W 24:00 python3 removeCellShadows.py "$RESULTS_DIR" "$START_FRAME" "$END_FRAME"

# Submit the tracking job to the cluster
bsub -J "segment_track_rocks" -n 8 -W 24:00 python3 segmentTrackRocks3D.py "$RESULTS_DIR" "$START_FRAME" "$END_FRAME"
