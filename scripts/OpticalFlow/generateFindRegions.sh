#!/bin/bash

# Prompt user to select the optical_flow_3Dresults directory
RESULTS_DIR=$(zenity --file-selection --directory --title="Select optical_flow_3Dresults directory" --filename="$(pwd)/")

# Let user select frame averaging option
FRAME_AVG=$(zenity --question \
    --title="Select Frame Averaging" \
    --text="Use frame averaged flow for smoother results?" \
    --width=300)
if [ $? -eq 0 ]; then
    FRAME_AVG=1  # User selected "Yes"
else
    FRAME_AVG=0  # User selected "No"
fi

# Let user select z slice
Z_SLICE=$(zenity --entry \
    --title="Select Z Slice" \
    --text="Enter the z slice to plot score maps (0-based index):" \
    --width=300)
if [ -z "$Z_SLICE" ]; then
    zenity --error --text="No z slice selected. Exiting." --width=300
    exit 1
fi

bsub -n 6 -gpu "num=1" -q gpu_a100 -W 24:00 -J "generateFindRegions" python findRegions.py "$RESULTS_DIR" "$FRAME_AVG" "$Z_SLICE"