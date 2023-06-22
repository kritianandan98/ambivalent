"""
Create audio features from files.
"""
import os
import numpy as np
import argparse
import h5py
import librosa
import time
import pandas as pd
from tqdm import trange
from sklearn.model_selection import train_test_split
from .. import config
from ..utils import create_folder


def to_one_hot(k, classes_num):
    target = np.zeros(classes_num)
    target[k] = 1
    return target


def get_features(audio_vec):
    fmin = config.fmin
    fmax = config.fmax
    mel_bins = config.mel_bins
    n_fft = config.n_fft
    hop_size = config.hop_size
    n_mfcc = config.n_mfcc
    sample_rate = config.sample_rate
    feature_name = config.feature_name

    mfccs = librosa.feature.mfcc(y=audio_vec, sr=sample_rate, n_mfcc=n_mfcc, fmin=fmin, fmax=fmax, n_fft=n_fft, hop_length=hop_size)
    mfccs_mean = np.mean(mfccs.T, axis=0).reshape(-1)
    mfccs_std = np.std(mfccs.T, axis=0).reshape(-1)
    mfccs_max = np.max(mfccs.T, axis=0).reshape(-1) 
    #print("mfcc", mfccs.shape)

    mels = librosa.feature.melspectrogram(y=audio_vec, sr=sample_rate, n_mels=mel_bins, fmin=fmin, fmax=fmax, n_fft=n_fft, hop_length=hop_size)
    mels_mean = np.mean(mels.T, axis=0).reshape(-1)
    mels_std = np.std(mels.T, axis=0).reshape(-1)
    mels_max = np.max(mels.T, axis=0).reshape(-1) 
    #print("mels", mels.shape)   

    
    if feature_name == "mfcc":
        return mfccs
    elif feature_name == "log-mels":
        logmel = librosa.power_to_db(mels)
        return logmel
    else:
        stft = np.abs(librosa.stft(audio_vec))
    
        pitches, magnitudes = librosa.piptrack(y=audio_vec, sr=sample_rate, S=stft, fmin=fmin, fmax=fmax, n_fft=n_fft, hop_length=hop_size)
        pitch = []
        for i in range(magnitudes.shape[1]):
            index = magnitudes[:, 1].argmax()
            pitch.append(pitches[index, i])
            
        #print("pitches", pitches.shape)
        pitch_tuning_offset = librosa.pitch_tuning(pitches).reshape(-1)
        #print("pitch tuning offset", pitch_tuning_offset.shape)
        pitch_mean = np.mean(pitch).reshape(-1)
        #print("pitch mean", pitch_mean.shape)
        pitch_std = np.std(pitch).reshape(-1)
        #print("pitch std", pitch_std.shape)
        pitch_max = np.max(pitch).reshape(-1)
        #print("pitch max", pitch_max.shape)
        pitch_min = np.min(pitch).reshape(-1)
        #print("pitch min", pitch_min.shape)
        
        cent = librosa.feature.spectral_centroid(y=audio_vec, sr=sample_rate, n_fft=n_fft, hop_length=hop_size)
        cent = cent / np.sum(cent)
        #print("centroid", cent.shape)
        cent_mean = np.mean(cent).reshape(-1)
        cent_std = np.std(cent).reshape(-1)
        cent_max = np.max(cent).reshape(-1)
        
        flatness = librosa.feature.spectral_flatness(y=audio_vec, n_fft=n_fft, hop_length=hop_size)
        flatness_mean = np.mean(flatness).reshape(-1)
        #print("flatnesss", flatness.shape)
        #print("flatness mean", flatness_mean.shape)
        
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate, n_fft=n_fft, hop_length=hop_size) 
        contrast_mean = np.mean(contrast.T, axis=0).reshape(-1)  
        #print("contrast", contrast.shape)
        #print("contrast mean", contrast_mean.shape)
        
        zerocr = librosa.feature.zero_crossing_rate(y=audio_vec)
        zerocr_mean = np.mean(zerocr).reshape(-1)
        #print("zero cr", zerocr.shape)

        rms = librosa.feature.rms(y=audio_vec+0.0001) # incase of empty signal
        rms_mean = np.mean(rms, axis=-1).reshape(-1)
        rms_std = np.std(rms, axis=-1).reshape(-1)
        rms_max = np.max(rms, axis=-1).reshape(-1)
        #print("rms shape", rms.shape)
        
        y_harmonic = np.mean(librosa.effects.hpss(y=audio_vec)[0]).reshape(-1)
        sig_mean = np.mean(abs(audio_vec)).reshape(-1)
        sig_std = np.std(audio_vec).reshape(-1)
        #print("harmonic", y_harmonic.shape)
 
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        chroma_mean = np.mean(chroma.T, axis=0).reshape(-1) 
        #print("chroma", chroma.shape)  

        ext_features_mean = np.concatenate([
            flatness_mean, zerocr_mean, cent_mean, cent_std,
            cent_max, pitch_mean, pitch_max, pitch_min, pitch_std,
            pitch_tuning_offset, rms_mean, rms_max, rms_std, y_harmonic, sig_mean, sig_std])
        
        if feature_name == "all-fixed":
            return np.concatenate((ext_features_mean, mfccs_mean, mfccs_std, mfccs_max, chroma_mean, mels_mean, contrast_mean))
        elif feature_name == "all-timesteps":
            return np.vstack([flatness, zerocr, cent, rms, mfccs, chroma, mels, contrast])


def create_hdf5_files(df, name, args):

    workspace = args.workspace
    minidata = args.minidata

    sample_rate = config.sample_rate
    mel_bins = config.mel_bins
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    lb_to_idx = config.lb_to_idx
    feature_name = config.feature_name

    if minidata:
        packed_hdf5_path = os.path.join(workspace, 'features', feature_name + '_' + name + '_minidata_waveform.h5')
    else:
        packed_hdf5_path = os.path.join(workspace, 'features', feature_name + '_' + name + '_waveform.h5')
    create_folder(os.path.dirname(packed_hdf5_path))

    audio_names = df["wavfile"]
    audio_paths = df['audio-path']
    targets = df["noisy-label"]
    ground_truth = df["ground-truth"]

    meta_dict = {
        'audio_name': np.array(audio_names), 
        'audio_path': np.array(audio_paths), 
        'target': np.array([lb_to_idx[target] for target in targets]), 
        'gt': np.array([lb_to_idx[gt] for gt in ground_truth])}
  
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
        features_grp = hf.create_group('features')
        info_grp = hf.create_group('info')

        info_grp.create_dataset(
            name='audio_name', 
            shape=(audios_num,), 
            dtype='S80')
        
        info_grp.create_dataset(
            name='audio_path', 
            shape=(audios_num,), 
            dtype='S80')

        info_grp.create_dataset(
            name='target', 
            shape=(audios_num, classes_num), 
            dtype=np.float32)
        
        info_grp.create_dataset(
            name='gt', 
            shape=(audios_num, classes_num), 
            dtype=np.float32)
 
        avg_time = 0
        max_time = 0

        for n in trange(audios_num):
            # Get features from audio
            audio_name = meta_dict['audio_name'][n]
            audio_path = meta_dict['audio_path'][n]
            (audio, fs) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

            duration = librosa.get_duration(y=audio, sr=sample_rate)
            avg_time += duration
            max_time = max(max_time, duration)

            #audio = pad_truncate_sequence(audio, clip_samples)

            features = np.array(get_features(audio), dtype=np.float32)
            #hf['feature_shape'][n] = features.shape
            #print(type(features))
            #print("feature shape", features.shape)

            features_grp.create_dataset(str(n), data=features)

            info_grp['audio_name'][n] = audio_name.encode()
            info_grp['audio_path'][n] = audio_path.encode()
            info_grp['target'][n] = to_one_hot(meta_dict['target'][n], classes_num)
            info_grp['gt'][n] = to_one_hot(meta_dict['gt'][n], classes_num)

        #print(hf['features']['1'][0])

    avg_time /= audios_num
    print('Avg time length of audio', avg_time)
    print('Write hdf5 to {}'.format(packed_hdf5_path))
    print('Time: {:.3f} s'.format(time.time() - feature_time))


def create_dataset(args):
    # Arguments & parameters
    dataset_csv = args.dataset_csv

    # train or test data
    test = args.test

    df = pd.read_csv(dataset_csv)

    if not test:
        # create train/val splits
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=0, stratify=df[['noisy-label']])
        create_hdf5_files(train_df, "train", args)
        create_hdf5_files(val_df, "val", args)
    else:
        create_hdf5_files(df, "test", args)


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
        create_dataset(args)
        
    else:
        raise Exception('Incorrect arguments!')