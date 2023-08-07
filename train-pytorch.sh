#!/bin/bash

DATASET_CSV="/home/kriti/ambivalent/src/hard-26.48-train.csv"
WORKSPACE="/home/kriti/ambivalent"

python3 -m src.features.create_features pack_audio_files_to_hdf5 --dataset_csv=$DATASET_CSV --workspace=$WORKSPACE --minidata --noisy
python3 -m src.features.create_features pack_audio_files_to_hdf5 --dataset_csv=$DATASET_CSV --workspace=$WORKSPACE --noisy
#python3 -m src.features.create_features pack_audio_files_to_hdf5 --dataset_csv=$DATASET_CSV --workspace=$WORKSPACE --minidata 
#python3 -m src.features.create_features pack_audio_files_to_hdf5 --dataset_csv=$DATASET_CSV --workspace=$WORKSPACE
# features - best
CUDA_VISIBLE_DEVICES=0 python3 -m src.pytorch_main train --workspace=$WORKSPACE --cuda
