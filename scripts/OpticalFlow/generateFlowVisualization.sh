#!/bin/bash

# set parameters

# Find the results folder to determine available timepoints
results_directory=$(zenity --file-selection --directory --title="Select the results folder" --filename="$(pwd)/")

frame_number=141
slice_index=10

bsub -n 8 -W 03:00 python3 flowVisualization.py "${results_directory}" "${frame_number}" "${slice_index}"