#!/bin/bash
PATH_TO_FARSIGHT=./img/cell_detection_0_f_centers.tif
EXPERIMENT=cell_detection_0
python3 ingest_tif_stack.py -collection cell_detection -experiment $EXPERIMENT -channel farsight_predictions -tif_stack $PATH_TO_FARSIGHT --type annotation --new_channel True --source_channel raw_data --config ./intern.cfg

# PATH_TO_FARSIGHT=./img/cell_detection_1_f_centers.tif
# EXPERIMENT=cell_detection_1
# python3 ingest_tif_stack.py -collection cell_detection -experiment $EXPERIMENT -channel farsight_predictions -tif_stack $PATH_TO_FARSIGHT --type annotation --new_channel True --source_channel raw_data --config ./intern.cfg
