#!/bin/bash

# Get variables 
zarr_folder=$(zenity --file-selection --directory --title="Select a zarr folder" --filename="$(pwd)/")
timepoint=$(zenity --entry --title="Time Point" --text="Enter the time point to visualize:")

bsub -n 8 -W 05:00 python3 makeZStackMovie.py "$zarr_folder" "$timepoint" 

