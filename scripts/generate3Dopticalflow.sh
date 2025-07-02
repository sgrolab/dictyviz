#!/bin/bash

# Prompt the user to select a max projection folder
zarr_folder=$(zenity --file-selection --directory --title="Select a zarr folder" --filename="$(pwd)/")

# Check if the user canceled the dialog
if [ -z "$zarr_folder" ]; then
    echo "No folder selected. Exiting."
    exit 1
fi

bsub -n 6 -gpu "num=1" -q gpu_a100 -W 05:00 -J python3 OpticalFlow/3Dopticalflow.py "${zarr_folder}"