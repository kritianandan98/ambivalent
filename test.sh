#!/bin/bash

DATASET_CSV="/home/kriti/ambivalent/src/hard-40.19-test.csv"
WORKSPACE="/home/kriti/ambivalent"

#python3 -m src.features.create_features pack_audio_files_to_hdf5 --dataset_csv=$DATASET_CSV --workspace=$WORKSPACE --test

CUDA_VISIBLE_DEVICES=0 python3 -m src.pytorch_main test --workspace=$WORKSPACE --cuda