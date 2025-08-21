#!/bin/bash

# Prompt the user to select a zarr folder
zarr_folder=$(zenity --file-selection --directory --title="Select a zarr folder" --filename="$(pwd)/")

# Check if the user canceled the dialog
if [ -z "$zarr_folder" ]; then
    echo "No folder selected. Exiting."
    exit 1
fi

# Prompt the user to select between the cells or the rocks channel to perform optical flow on
channel=$(zenity --list --title="Select Channel" --column="Channel" "cells" "rocks" --text="Select the channel to perform optical flow on:")
# Check if the user canceled the dialog
if [ -z "$channel" ]; then
    echo "No channel selected. Exiting."
    exit 1
fi

bsub -n 6 -gpu "num=1" -q gpu_h200 -W 24:00 -J "optical_flow_3D" python 3Dflow.py "${zarr_folder}" "${channel}"