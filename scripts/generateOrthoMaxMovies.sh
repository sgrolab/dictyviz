#/bin/bash

# Prompt the user to select a zarr file
selected_folder=$(zenity --file-selection --directory --title="Select a zarr file" --filename="$(pwd)/")

# Check if the user canceled the dialog
if [ -z "$selected_folder" ]; then
    echo "No folder selected. Exiting."
    exit 1
fi

# submit max projection calculation jobs, waiting for sliced max projs to finish
bsub -n 8 -W 24:00 python calcOrthoMaxProjs.py "${selected_folder}"
bsub -n 8 -W 24:00 -K python calcSlicedOrthoMaxProjs.py "${selected_folder}"

# submit movie making jobs, waiting for max projs to finish
bsub -n 8 -W 12:00 -K python makeOrthoProjMovies.py "${selected_folder}"

./compressMovies.sh "${selected_folder}"