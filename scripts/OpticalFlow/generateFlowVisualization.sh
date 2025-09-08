#!/bin/bash

# Prompt user to select the optical_flow_3Dresults directory
RESULTS_DIR=$(zenity --file-selection --directory --title="Select optical_flow_3Dresults directory" --filename="$(pwd)/")

echo "Results directory: $RESULTS_DIR"

bsub -n 8 -W 12:00 python3 flowVisualization.py "$RESULTS_DIR"