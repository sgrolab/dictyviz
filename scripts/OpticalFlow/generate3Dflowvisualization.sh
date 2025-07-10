#!/bin/bash

# Prompt user to select the optical_flow_3Dresults directory
RESULTS_DIR=$(zenity --file-selection --directory --title="Select optical_flow_3Dresults directory" --filename="$(pwd)/")

# Get list of available numeric frame directories
FRAME_LIST=$(ls -1 "$RESULTS_DIR" | grep -E '^[0-9]+$' | sort -n)
if [ -z "$FRAME_LIST" ]; then
    zenity --error --text="No frame directories found in: $RESULTS_DIR"
    exit 1
fi

# Convert frame list to Zenity radio button format
FRAME_OPTIONS=""
FIRST_FRAME=true
for frame in $FRAME_LIST; do
    if [ "$FIRST_FRAME" = true ]; then
        FRAME_OPTIONS="TRUE $frame"
        FIRST_FRAME=false
    else
        FRAME_OPTIONS="$FRAME_OPTIONS FALSE $frame"
    fi
done

# Let user select a frame
SELECTED_FRAME=$(zenity --list \
    --radiolist \
    --title="Select Frame Number" \
    --text="Choose which frame to visualize:" \
    --column="Select" \
    --column="Frame" \
    --width=300 \
    --height=400 \
    $FRAME_OPTIONS)

if [ $? -ne 0 ] || [ -z "$SELECTED_FRAME" ]; then
    echo "No frame selected."
    exit 1
fi

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

echo "Selected frame: $SELECTED_FRAME"
echo "Selected slice: ${SLICE_INDEX:-'middle (default)'}"
echo "Results directory: $RESULTS_DIR"

# Build command with optional slice index
if [ ! -z "$SLICE_INDEX" ]; then
    bsub -n 8 -W 00:30 python3 visualize3dflow.py "$RESULTS_DIR" "$SELECTED_FRAME" "$SLICE_INDEX"
else
    bsub -n 8 -W 00:30 python3 visualize3dflow.py "$RESULTS_DIR" "$SELECTED_FRAME"
fi