#/bin/bash

# Prompt the user to select a folder containing a zarr file
zarr_folder=$(zenity --file-selection --directory --title="Select a zarr folder" --filename="$(pwd)/")

# Check if the user canceled the dialog
if [ -z "$zarr_folder" ]; then
    echo "No folder selected. Exiting."
    exit 1
fi

# Check if max projections have already been calculated
if [ -d "${zarr_folder}/../analysis/max_projections/maxx" ]; then
    echo "Max projections already calculated, skipping."
else
    # Submit max projection calculation job
    bsub -n 8 -W 24:00 python calcOrthoMaxProjs.py "${zarr_folder}"
fi

# Check if sliced max projections have already been calculated
if [ -d "${zarr_folder}/../analysis/sliced_max_projections/sliced_maxx" ]; then
    echo "Sliced max projections already calculated, skipping."
else
    # Submit sliced max projection calculation job
    bsub -n 8 -W 24:00 -K python calcSlicedOrthoMaxProjs.py "${zarr_folder}"
fi

# Submit movie making jobs, waiting for max projs to finish
bsub -n 8 -W 12:00 -K python makeOrthoProjMovies.py "${zarr_folder}"

# Submit compression job, waiting for movie making to finish
./compressMovies.sh "${zarr_folder}"