#!/bin/bash

# Prompt the user to select a max projection folder
zarr_folder=$(zenity --file-selection --directory --title="Select a zarr folder" --filename="$(pwd)/")

# Check if the user canceled the dialog
if [ -z "$zarr_folder" ]; then
    echo "No folder selected. Exiting."
    exit 1
fi

# Ask user for optional cropID
cropID=$(zenity --entry --title="Crop ID" --text="Enter crop ID (leave empty if none):" --entry-text="")

bsub -n 8 -W 24:00 python3 testParameters.py "${zarr_folder}" "${cropID}"

