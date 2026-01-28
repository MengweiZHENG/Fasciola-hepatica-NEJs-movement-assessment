#!/bin/bash
set -e  # Exit immediately if a command fails
source ~/miniforge3/etc/profile.d/conda.sh

# Check if a name was provided as an argument
if [ -z "$1" ]; then
  echo "Usage: ./run.sh <folder>"
  exit 1
fi

DIR="$1"  # Take folder name from first argument

##############################################
## Create conda envs if needed (uncomment if first time running)
# conda create -n pixel  -c conda-forge python=3.11 opencv matplotlib -y
# conda create -n merge -c conda-forge pandas -y
# conda create --name r_env -c conda-forge r-base=4.2 r-dplyr r-ggplot2 r-svglite -y
##############################################

##  Run pixel processing
conda activate pixel

rm -rf "$DIR.output_pixel"
mkdir -p "$DIR.output_pixel"

# Loop over subfolders in DIR
for line in "$DIR"/*; do
    # Skip if not a directory
    [ -d "$line" ] || continue

    foldername=$(basename "$line")

    rm -rf "$line/raw_tif"
    mkdir -p "$line/raw_tif"

    cp "$line"/*Plate_R*TIF "$line/raw_tif/"

    python /home/neil/Mengwei/TIFF_stack/pixels.py -i "$line/raw_tif/" -o "$line/$foldername.out.txt" \
        --video-output "$line/$foldername.movie.avi" \
        --background-image "$line/$foldername.background.png" \
        --heatmap-image "$line/$foldername.heatmap.png" \
        --threshold 90 --fps 1

    cp "$line/$foldername.out.txt" "$DIR.output_pixel/"
done

conda deactivate
