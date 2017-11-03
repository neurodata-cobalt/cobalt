#!/bin/bash

set -e  # Comment this out if you want to ignore exceptions

IMG_DIR=./annotated_img/                        # Path to annotated images
OUTPUT_DIR=./farsight_output/                   # Path to where outputs will be saved
FARSIGHT_BIN=segment_nuclei                     # Command to run FARSIGHT
CENTERS_DIR=./../jliu118/annotation-csv/        # Path to where csv centers are

if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR;
fi

results_csv="farsight_results.csv"
echo "subvolume, annotated count, farsight count, accuracy, precision, recall, mse" > $OUTPUT_DIR$results_csv

printf "\nComputed centers will be saved to $OUTPUT_DIR\n"
printf "Results will be saved in $OUTPUT_DIR$result_csv\n\n"

for entry in "$IMG_DIR"/*
do
    if [[ "$entry" == *.tif ]] || [[ "$entry" == *.tiff ]]; then
        fname=${entry%.tif}   # Removes .tif
        fname=${fname%.tiff}  # Removes .tiff
        fname=${fname##*/}    # Removes path prefix
        printf "Processing $fname.tif\n"

        # Run FARSIGHT on the annotated image
        $FARSIGHT_BIN $entry "$OUTPUT_DIR${fname%.*}_output.tif" >/dev/null
        output_dat="${fname}_seg_final.dat"
        output_txt="${fname}._seedPoints.txt"

        # FARSIGHT saves the output to the source directory
        # So we have to manually move the output to the output directory
        mv "$IMG_DIR$output_dat" "$OUTPUT_DIR$output_dat"
        mv "$IMG_DIR$output_txt" "$OUTPUT_DIR$output_txt"

        # FARSIGHT saves the centers to txt in x y z format.
        # We fix this and save as csv in z y x format.
        output_csv="${fname}_centers.csv"
        echo "z, y, x" > "$OUTPUT_DIR$output_csv"
        while IFS=' ' read -r x y z
        do
        	echo "$z , $y , $x" >> "$OUTPUT_DIR$output_csv"
        done <"$OUTPUT_DIR$output_txt"

        annotated_centers="${fname}.csv"
        drawn_output="${fname}_drawn.tif"
        python3 evaluate_centers.py $OUTPUT_DIR$output_csv $CENTERS_DIR$annotated_centers $OUTPUT_DIR$results_csv $OUTPUT_DIR$drawn_output $entry
        printf "Elapsed time: $SECONDS\n\n"
    fi
done
