#/bin/bash

# Prompt the user to select a folder containing a zarr file
zarr_folder=$(zenity --file-selection --directory --title="Select a zarr folder" --filename="$(pwd)/")

# Check if the user canceled the dialog
if [ -z "$zarr_folder" ]; then
    echo "No folder selected. Exiting."
    exit 1
fi

bsub -n 8 -W 01:00 python3 ./opticalFlow.py "${zarr_folder}"
