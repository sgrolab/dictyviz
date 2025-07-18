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


echo "Selected frame: $SELECTED_FRAME"
echo "Results directory: $RESULTS_DIR"


bsub -n 6 -gpu "num=1" -q gpu_a100 -W 24:00 -J "generateFindRegions" python findRegions.py "$RESULTS_DIR" "$SELECTED_FRAME"