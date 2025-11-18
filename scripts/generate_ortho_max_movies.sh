#/bin/bash

# Prompt the user to select a folder containing a zarr file
zarr_folder=$(zenity --file-selection --directory --title="Select a zarr folder" --filename="$(pwd)/")

# Check if the user canceled the dialog
if [ -z "$zarr_folder" ]; then
    echo "No folder selected. Exiting."
    exit 1
fi

params_json="${zarr_folder}/../parameters.json"

# Use Python to extract cropping parameters if they exist
IFS=$'\t' read -r cropX cropY cropZ <<< "$(python3 -c "
import json, sys
try:
    with open('$params_json') as f:
        params = json.load(f)
    cropping = params.get('croppingParameters')
    if cropping:
        cropX = cropping.get('cropX', 'None')
        cropY = cropping.get('cropY', 'None')
        cropZ = cropping.get('cropZ', 'None')
        print(f'{cropX}\t{cropY}\t{cropZ}')
    else:
        print('None\tNone\tNone')
except Exception as e:
    print('False\tFalse\tFalse')
")"

echo "Cropping parameters: cropX=$cropX, cropY=$cropY, cropZ=$cropZ"

# Generate cropID based on cropping parameters
if [[ "$cropX" != "None" ]]; then
    cropX_min=$(echo "$cropX" | sed 's/\[//;s/\]//;s/ //g' | cut -d, -f1)
    cropX_max=$(echo "$cropX" | sed 's/\[//;s/\]//;s/ //g' | cut -d, -f2)
    cropX_ID="_X${cropX_min}-${cropX_max}"
else
    cropX_ID=""
fi

if [[ "$cropY" != "None" ]]; then
    cropY_min=$(echo "$cropY" | sed 's/\[//;s/\]//;s/ //g' | cut -d, -f1)
    cropY_max=$(echo "$cropY" | sed 's/\[//;s/\]//;s/ //g' | cut -d, -f2)
    cropY_ID="_Y${cropY_min}-${cropY_max}"
else
    cropY_ID=""
fi

if [[ "$cropZ" != "None" ]]; then
    cropZ_min=$(echo "$cropZ" | sed 's/\[//;s/\]//;s/ //g' | cut -d, -f1)
    cropZ_max=$(echo "$cropZ" | sed 's/\[//;s/\]//;s/ //g' | cut -d, -f2)
    cropZ_ID="_Z${cropZ_min}-${cropZ_max}"
else
    cropZ_ID=""
fi

if [[ "$cropX" == "False" ]]; then
    crop_ID=""
else
    crop_ID="${cropX_ID}${cropY_ID}${cropZ_ID}"
fi
echo "Crop ID: $crop_ID"

#TODO: use python to extract slice depth from parameters.json if it exists

# Check if max projections have already been calculated
if [ -d "${zarr_folder}/../analysis/max_projections${crop_ID}/maxx" ]; then
    echo "Max projections already calculated, skipping."
else
    # Submit max projection calculation job
    echo "Submitting max projection calculation job..."
    bsub -n 16 -W 24:00 python calc_ortho_max_projs.py "${zarr_folder}" "${crop_ID}"
fi

# Check if sliced max projections have already been calculated
if [ -d "${zarr_folder}/../analysis/sliced_max_projections${crop_ID}/sliced_maxx" ]; then
    echo "Sliced max projections already calculated, skipping."
else
    # Submit sliced max projection calculation job
    echo "Submitting sliced max projection calculation job..."
    bsub -n 16 -W 24:00 -K python calc_sliced_ortho_max_projs.py "${zarr_folder}" "${crop_ID}"
fi

# Submit movie making jobs, waiting for max projs to finish
bsub -n 8 -W 12:00 -K python make_ortho_proj_movies.py "${zarr_folder}" "${crop_ID}"

# Submit compression job, waiting for movie making to finish
./compress_movies.sh "${zarr_folder}" "${crop_ID}"