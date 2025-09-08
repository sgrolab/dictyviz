#!/bin/bash

# Prompt user to select the optical_flow_3Dresults directory
RESULTS_DIR=$(zenity --file-selection --directory --title="Select optical_flow_3Dresults directory" --filename="$(pwd)/")

# Let user select the number of frames to average
NB_FRAMES=$(zenity --entry \
    --title="Number of Frames" \
    --text="Enter the number of frames to average:" \
    --entry-text="" \
    --width=300)
    
if [ $? -ne 0 ]; then
    echo "No frame number specified. Using default value of 5."
    NB_FRAMES=5
fi

echo "Selected directory: $RESULTS_DIR"
echo "Number of frames to average: $NB_FRAMES"

# Build the command with the selected directory and number of frames
bsub -n 8 -W 06:00 python3 averageFrames.py "$RESULTS_DIR" "$NB_FRAMES"