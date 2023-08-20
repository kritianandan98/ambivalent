#!/bin/bash

DATASET_CSV="/home/kriti/train.csv"
WORKSPACE="/home/kriti/ambivalent"

python3 -m src.features.create_features pack_audio_files_to_hdf5 --dataset_csv=$DATASET_CSV --workspace=$WORKSPACE --minidata
python3 -m src.features.create_features pack_audio_files_to_hdf5 --dataset_csv=$DATASET_CSV --workspace=$WORKSPACE

CUDA_VISIBLE_DEVICES=0 python3 -m src.sklearn_main train --workspace=$WORKSPACE --cuda