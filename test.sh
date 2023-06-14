#!/bin/bash

DATASET_CSV="/home/kriti/test.csv"
WORKSPACE="/home/kriti/panns_transfer_to_gtzan"
MODEL_CHECKPOINT="/home/kriti/panns_transfer_to_gtzan/checkpoints/main/holdout_fold=1/Transfer_Cnn14/pretrain=False/loss_type=clip_nll/augmentation=none/batch_size=8/freeze_base=False/3_epoch.pth"

python3 utils/features.py pack_audio_files_to_hdf5 --dataset_csv=$DATASET_CSV --workspace=$WORKSPACE --test

#PRETRAINED_CHECKPOINT_PATH="/home/tiger/released_models/sed/Cnn14_mAP=0.431.pth"

#CUDA_VISIBLE_DEVICES=3 python3 pytorc/home/tiger/panns_transfer_to_gtzanh/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type="Transfer_Cnn14" --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=10000 --cuda
#CUDA_VISIBLE_DEVICES=0 python3 /home/kriti/panns_transfer_to_gtzan/pytorch/main.py test --workspace=$WORKSPACE --model_type="Cnn14" --holdout_fold=1 --loss_type=clip_nll --batch_size=8 --model_path=$MODEL_CHECKPOINT --cuda
#####
#MODEL_TYPE="Transfer_Cnn14"
#PRETRAINED_CHECKPOINT_PATH="/vol/vssp/msos/qk/bytedance/workspaces_important/pub_audioset_tagging_cnn_transfer/checkpoints/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn13/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=32/660000_iterations.pth"
#python3 pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --freeze_base --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --few_shots=10 --random_seed=1000 --resume_iteration=0 --stop_iteration=10000 --cuda