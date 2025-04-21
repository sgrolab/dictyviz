#/bin/bash

# Prompt the user to select a folder containing a zarr file
selected_folder=$(zenity --file-selection --directory --title="Select a folder containing a zarr file" --filename="$(pwd)/")

# Check if the user canceled the dialog
if [ -z "$selected_folder" ]; then
    echo "No folder selected. Exiting."
    exit 1
fi

# Identify a zarr file in the selected folder
zarr_folder=$(find "$selected_folder" -maxdepth 1 -type d -name "*.zarr" | head -n 1)

# Check if a zarr file was found
if [ -z "$zarr_folder" ]; then
    echo "No zarr file found in the selected folder. Exiting."
    exit 1
fi

echo "Found .zarr file: $zarr_folder"

# Check if max projections have already been calculated
if [ -d "${selected_folder}/analysis/max_projections/maxx" ]; then
    echo "Max projections already calculated, skipping."
else
    # Submit max projection calculation job
    bsub -n 8 -W 24:00 python calcOrthoMaxProjs.py "${zarr_folder}"
fi

# Check if sliced max projections have already been calculated
if [ -d "${selected_folder}/analysis/sliced_max_projections/sliced_maxx" ]; then
    echo "Sliced max projections already calculated, skipping."
else
    # Submit sliced max projection calculation job
    bsub -n 8 -W 24:00 -K python calcSlicedOrthoMaxProjs.py "${zarr_folder}"
fi

# Submit movie making jobs, waiting for max projs to finish
bsub -n 8 -W 12:00 -K python makeOrthoProjMovies.py "${selected_folder}"

# Submit compression job, waiting for movie making to finish
./compressMovies.sh "${selected_folder}"