#!/bin/bash

# Prompt the user to select a folder containing a zarr file
optical_folder=$(zenity --file-selection --directory --title="Select an optical folder results folder" --filename="$(pwd)/")

# Check if the user canceled the dialog
if [ -z "$optical_folder" ]; then
    echo "No folder selected. Exiting."
    exit 1
fi

params_json="${optical_folder}/../parameters.json"

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

# Get parent directory for the analysis folder
parent_dir=$(dirname "$optical_folder")

# Submit max projection calculation job
echo "Submitting max projection calculation job..."
max_proj_job=$(bsub -n 8 -W 24:00 python calcOrthoFlowMaxProjections.py "${optical_folder}" "${crop_ID}")

# Extract job ID for dependency
max_proj_job_id=$(echo "$max_proj_job" | grep -o 'Job <[0-9]*>' | grep -o '[0-9]*')

# Submit movie making job, dependent on max projections job completion
echo "Submitting movie making job (depends on job $max_proj_job_id)..."
bsub -n 8 -W 12:00 -w "done($max_proj_job_id)" python makeOrthoFlowMaxMovies.py "${parent_dir}/analysis/optical_flow_max_projections${crop_ID}" "${crop_ID}"