#!/bin/bash

# Prompt user to select the optical_flow_3Dresults directory
RESULTS_DIR=$(zenity --file-selection --directory --title="Select optical_flow_3Dresults directory" --filename="$(pwd)/")

echo "Results directory: $RESULTS_DIR"

# Let user select slice index
SLICE_INDEX=$(zenity --entry \
    --title="Select Z-Slice Index" \
    --text="Enter the Z-slice index to visualize:" \
    --entry-text="" \
    --width=300)

if [ $? -ne 0 ]; then
    echo "No slice selected. Using default (middle slice)."
    SLICE_INDEX=""
fi

bsub -n 8 -W 12:00 python3 movieSliceVisualization.py "$RESULTS_DIR" "$SLICE_INDEX"