#!/bin/bash

# DOWNLOAD TRAIN- AND TESTSET W/ ANNOTATIONS FROM SOURCE
cd ../../mpi_inf_3dhp/
bash get_dataset.sh
bash get_testset.sh

# PREPARE DATASETS
cd ../../3DHPE/utils/
bash get_datasets.sh

# EXTRACT ANNOTATIONS AND CLEAN UP INVALID FRAMES
python get_annot.py
python clean_frames.py