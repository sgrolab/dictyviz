#/bin/bash

# Check if a folder was provided as a command-line argument
if [ -n "$1" ]; then
    selected_folder="$1"
    # Validate that the provided path is a directory
    if [ ! -d "$selected_folder" ]; then
        echo "Error: The provided path '$selected_folder' is not a valid directory."
        exit 1
    fi
else
    # Prompt the user to select a folder using zenity
    selected_folder=$(zenity --file-selection --directory --title="Select a zarr file" --filename="$(pwd)/")

    # Check if the user canceled the dialog
    if [ -z "$selected_folder" ]; then
        echo "No folder selected. Exiting."
        exit 1
    fi
fi

# Change to the movies directory inside the selected folder
movies_dir="$selected_folder/movies"
if [ ! -d "$movies_dir" ]; then
    echo "Error: The 'movies' directory does not exist in the selected folder."
    exit 1
fi

cd "$movies_dir" || { echo "Failed to change directory to $movies_dir. Exiting."; exit 1; }

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
        -c:v libx264 -pix_fmt yuv420p -crf "$crf" \
        "$output_file"
done