#!/bin/bash

# Prompt the user to select a zarr folder
zarr_folder=$(zenity --file-selection --directory --title="Select a zarr folder" --filename="$(pwd)/")

# Check if the user canceled the dialog
if [ -z "$zarr_folder" ]; then
    echo "No folder selected. Exiting."
    exit 1
fi

bsub -n 6 -gpu "num=1" -q gpu_a100 -W 24:00 -J "optical_flow_3D" python 3Dopticalflow.py "${zarr_folder}"