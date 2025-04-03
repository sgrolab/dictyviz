#/bin/bash

# Prompt the user to select a folder using zenity
selected_folder=$(zenity --file-selection --directory --title="Select a folder containing .avi files" --filename="$(pwd)/")

# Check if the user canceled the dialog
if [ -z "$selected_folder" ]; then
    echo "No folder selected. Exiting."
    exit 1
fi

# Change to the selected folder
cd "$selected_folder" || { echo "Failed to change directory to $selected_folder. Exiting."; exit 1; }

#Get a list of all .avi files in the current directory
avi_files=$(ls *.avi 2>/dev/null)

# Check if there are no .avi files in the folder
if [ -z "$avi_files" ]; then
    echo "No .avi files found in the selected folder. Exiting."
    exit 1
fi

# Loop through each .avi file and compress it
for avi_file in $avi_files; do
    # Generate the output filename by replacing .avi with .mp4
    output_file="${avi_file%.avi}.mp4"

    # Check if the output file already exists
    if [ -f "$output_file" ]; then
        echo "Skipping $avi_file, output file $output_file already exists."
        continue
    fi

    # Get the file size in bytes
    file_size=$(stat -c%s "$avi_file")

    # Determine the CRF value based on the file size
    if [ "$file_size" -lt $((1 * 1024 * 1024 * 1024)) ]; then
        crf=28
    elif [ "$file_size" -lt $((2 * 1024 * 1024 * 1024)) ]; then
        crf=32
    else
        crf=36
    fi

    # Submit the compression job
    bsub \
        -n 4 -W 00:30 \
        ffmpeg -i "$avi_file" \
        -c:v libx265 -pix_fmt yuv420p -crf "$crf" \
        "$output_file"
done