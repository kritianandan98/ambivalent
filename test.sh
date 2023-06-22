#!/bin/bash

DATASET_CSV="/home/kriti/test.csv"
WORKSPACE="/home/kriti/ambivalent"

#python3 -m src.features.create_features pack_audio_files_to_hdf5 --dataset_csv=$DATASET_CSV --workspace=$WORKSPACE --test

CUDA_VISIBLE_DEVICES=0 python3 -m src.main test --workspace=$WORKSPACE --cuda
