"""
Create Log Mel features and store it in an hdf5 file
"""
import os
import sys
import numpy as np
import argparse
import h5py
import librosa
import matplotlib.pyplot as plt
import time
import csv
import math
import torch
import re
import random
import pandas as pd
from tqdm import trange
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from config import (sample_rate, classes_num, mel_bins, fmin, fmax, window_size, 
    hop_size, window, pad_mode, center, ref, amin, top_db, idx_to_lb)

import config
from utilities import create_folder, float32_to_int16


def to_one_hot(k, classes_num):
    target = np.zeros(classes_num)
    target[k] = 1
    return target


def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]


def pack_audio_files_to_hdf5(args):
    # Arguments & parameters
    dataset_csv = args.dataset_csv
    workspace = args.workspace
    minidata = args.minidata

    # train or test data
    test = args.test

    sample_rate = config.sample_rate
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    lb_to_idx = config.lb_to_idx

    spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)
    
    logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
    
    # Spec augmenter
    #    self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
    #        freq_drop_width=8, freq_stripes_num=2)

    if minidata:
        packed_hdf5_path = os.path.join(workspace, 'features', 'minidata_waveform.h5')
    elif test:
        packed_hdf5_path = os.path.join(workspace, 'features', 'test-waveform.h5')
    else:
        packed_hdf5_path = os.path.join(workspace, 'features', 'waveform.h5')
    create_folder(os.path.dirname(packed_hdf5_path))

    df = pd.read_csv(dataset_csv)
    audio_names = df["wavfile"]
    audio_paths = df['audio-path']
    targets = df["noisy-label"]
    ground_truth = df["ground-truth"]
    
    #audio_names = sorted(audio_names)
    #audio_paths = sorted(audio_paths)

    meta_dict = {
        'audio_name': np.array(audio_names), 
        'audio_path': np.array(audio_paths), 
        'target': np.array([lb_to_idx[target] for target in targets]), 
        'gt': np.array([lb_to_idx[gt] for gt in ground_truth]),
        'fold': np.arange(len(audio_names)) % 10 + 1}
    #print(len(meta_dict['audio_name']))
    #print(meta_dict['fold'])
    
    if minidata:
        mini_num = 10
        total_num = len(meta_dict['audio_name'])
        random_state = np.random.RandomState(1234)
        indexes = random_state.choice(total_num, size=mini_num, replace=False)
        for key in meta_dict.keys():
            meta_dict[key] = meta_dict[key][indexes]

    audios_num = len(meta_dict['audio_name'])

    feature_time = time.time()
    with h5py.File(packed_hdf5_path, 'w') as hf:
        hf.create_dataset(
            name='audio_name', 
            shape=(audios_num,), 
            dtype='S80')
        
        hf.create_dataset(
            name='audio_path', 
            shape=(audios_num,), 
            dtype='S80')

        #hf.create_dataset(
        #    name='waveform', 
        #    shape=(audios_num, clip_samples), 
        #    dtype=np.int16)

        hf.create_dataset(
            name='logmel_feat', 
            shape=(audios_num, 1501, mel_bins), 
            dtype=np.float32)

        hf.create_dataset(
            name='target', 
            shape=(audios_num, classes_num), 
            dtype=np.float32)
        
        hf.create_dataset(
            name='gt', 
            shape=(audios_num, classes_num), 
            dtype=np.float32)

        hf.create_dataset(
            name='fold', 
            shape=(audios_num,), 
            dtype=np.int32)  
 
        for n in trange(audios_num):
            audio_name = meta_dict['audio_name'][n]
            fold = meta_dict['fold'][n]
            audio_path = meta_dict['audio_path'][n]
            (audio, fs) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

            audio = pad_truncate_sequence(audio, clip_samples)

            x = spectrogram_extractor(torch.FloatTensor(float32_to_int16(audio).reshape(1, -1)))   # (batch_size, 1, time_steps, freq_bins)
            features = torch.squeeze(logmel_extractor(x)).numpy()

            hf['audio_name'][n] = audio_name.encode()
            hf['audio_path'][n] = audio_path.encode()
            hf['logmel_feat'][n] = features
            hf['target'][n] = to_one_hot(meta_dict['target'][n], classes_num)
            hf['fold'][n] = meta_dict['fold'][n]
            hf['gt'][n] = to_one_hot(meta_dict['gt'][n], classes_num)

    print('Write hdf5 to {}'.format(packed_hdf5_path))
    print('Time: {:.3f} s'.format(time.time() - feature_time))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    # Calculate log mel features for all audio files
    parser_pack_audio = subparsers.add_parser('pack_audio_files_to_hdf5')
    parser_pack_audio.add_argument('--dataset_csv', type=str, required=True, help='CSV file of dataset.')
    parser_pack_audio.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_pack_audio.add_argument('--minidata', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    parser_pack_audio.add_argument('--test', action='store_true', default=False, help='Train or test data')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.mode == 'pack_audio_files_to_hdf5':
        pack_audio_files_to_hdf5(args)
        
    else:
        raise Exception('Incorrect arguments!')