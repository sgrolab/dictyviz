#!/bin/bash

# Prompt user to select the optical_flow_3Dresults directory
RESULTS_DIR=$(zenity --file-selection --directory --title="Select optical_flow_3Dresults directory" --filename="$(pwd)/")

bsub -n 6 -gpu "num=1" -q gpu_a100 -W 24:00 -J "generateFindRegions" python findRegions.py "$RESULTS_DIR" 