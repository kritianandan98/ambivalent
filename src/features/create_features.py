"""
Create and store audio features from audio files.
"""
import os
import torch
import numpy as np
import argparse
import h5py
import librosa
import time
import pandas as pd
from tqdm import trange
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
from .. import config
from ..utils import create_folder
import matplotlib.pyplot as plt
from skimage.transform import resize

torch.manual_seed(17)


def to_one_hot(k: int, classes_num: int) -> np.ndarray:
    """
    Convert labels to one-hot encoded labels.
    Args
        k           : emotion label ID
        classes_num : number of unique emotion labels
    Returns
        target      : one-hot encoded numpy array
    """

    target = np.zeros(classes_num)
    target[k] = 1
    return target


def zero_padding(data: np.ndarray, segment_length: int) -> np.ndarrray:
    """
    Zero-padding the data back and forth; to reach the desired size.
    Args
        data           : input data (shape(len_data,)) and must len_data <= segment_length
        segment_length : desired size (sample)
    Returns
        padded_data : data with desired size
    """

    padding = segment_length - np.shape(data)[0]
    pre = int(np.floor(padding / 2))
    pos = int(-np.floor(-padding / 2))
    padded_data = np.concatenate(
        [np.zeros(pre, dtype=np.float32), data, np.zeros(pos, dtype=np.float32)], axis=0
    )
    return padded_data


def normalize(data: np.ndarray) -> np.ndarray:
    """
    This function performs the normalization operation ([- 1,1]) of the data.
    Args
        data                : input data
    Returns
        normalized_samples  : normalized data
    """
    EPS = np.finfo(float).eps
    samples_99_percentile = np.percentile(np.abs(data), 99.9)
    normalized_samples = data / (samples_99_percentile + EPS)
    normalized_samples = np.clip(normalized_samples, -1, 1)
    return normalized_samples


def pad_sequences_last_value(data: np.ndarray, segment_length: int) -> np.ndarray:
    """
    This function pads the end of the sequence with last value.
    Args
        data                : input data
        segment_length      : desired size (sample)
    Returns
        padded_data         : data with desired size
    """
    seq_len = data.shape[1]
    if seq_len >= segment_length:
        padded_data = data[:, :segment_length]
    else:
        pad_width = ((0, 0), (0, segment_length - seq_len))
        padded_data = np.pad(data, pad_width, mode="edge")
    return padded_data


def get_features(audio_vec: np.ndarray, audio_name: str, audio_path: str):
    """
    Function to get features from a given audio file.
    Args
        audio_vec   : time series audio in floating numpy array format
        audio_name  : name of the audio file
        audio_path  : path to audio file
    Returns
        None
    """
    workspace = config.workspace

    fmin = config.fmin
    fmax = config.fmax
    mel_bins = config.mel_bins
    n_fft = config.n_fft
    hop_size = config.hop_size
    timesteps = config.timesteps
    n_mfcc = config.n_mfcc
    sample_rate = config.sample_rate
    feature_name = config.feature_name
    clip_samples = config.clip_samples

    if feature_name == "mean-mfcc":
        mfccs = librosa.feature.mfcc(
            y=audio_vec,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            fmin=fmin,
            fmax=fmax,
            n_fft=n_fft,
            hop_length=hop_size,
        )
        mfccs_mean = np.mean(mfccs.T, axis=0).reshape(-1)
        return mfccs_mean

    elif feature_name == "audios-segmented":
        min_samples = 1 * 16000
        features = []
        (y, fs) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

        if len(y) <= config.clip_samples:
            y = librosa.util.pad_center(y, size=clip_samples, mode="constant")
            features.append(y)  # (timesteps)

        else:
            for start_frame in range(0, len(y), clip_samples):
                end_frame = min(len(y), start_frame + clip_samples)
                seg_y = y[start_frame:end_frame]
                if len(seg_y) < min_samples:  # discard segments less than 1 second
                    continue
                if len(seg_y) < config.clip_samples:
                    seg_y = librosa.util.pad_center(
                        seg_y, size=clip_samples, mode="constant"
                    )
                features.append(seg_y)

        features = np.array(features, dtype=np.float32)  # n_segments, timesteps, n_mels
        return features

    elif feature_name == "audios":
        (y, fs) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
        return np.array(y, dtype=np.float32)

    elif feature_name == "logmels-a":
        min_samples = 1 * 16000
        features = []
        (y, fs) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
        sig_mean = np.mean(y)
        sig_std = np.std(y)
        y = (y - sig_mean) / sig_std
        if len(y) <= config.clip_samples:
            y = librosa.util.pad_center(y, size=config.clip_samples, mode="constant")
            mels = librosa.feature.melspectrogram(
                y=y,
                sr=sample_rate,
                n_mels=mel_bins,
                fmin=fmin,
                fmax=fmax,
                n_fft=n_fft,
                hop_length=hop_size,
            )
            logmels = librosa.power_to_db(mels)
            melnormalized = (logmels - np.mean(logmels)) / np.std(logmels)
            features.append(melnormalized.T)  # (timesteps, n_mels)
        else:
            for start_frame in range(0, len(y), config.clip_samples):
                end_frame = min(len(y), start_frame + config.clip_samples)
                seg_y = y[start_frame:end_frame]
                if len(seg_y) < min_samples:  # discard segments less than 1 second
                    continue
                if len(seg_y) < config.clip_samples:
                    seg_y = librosa.util.pad_center(
                        seg_y, size=config.clip_samples, mode="constant"
                    )
                mels = librosa.feature.melspectrogram(
                    y=seg_y,
                    sr=sample_rate,
                    n_mels=mel_bins,
                    fmin=fmin,
                    fmax=fmax,
                    n_fft=n_fft,
                    hop_length=hop_size,
                )
                logmels = librosa.power_to_db(mels)
                melnormalized = (logmels - np.mean(logmels)) / np.std(logmels)
                features.append(melnormalized.T)

        features = np.array(features, dtype=np.float32)  # n_segments, timesteps, n_mels
        return features  # timesteps, n_mels

    elif feature_name == "logmels-b":
        min_samples = 1 * 16000
        features = []
        (y, fs) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
        y = normalize(y)
        if len(y) <= config.clip_samples:
            mels = librosa.feature.melspectrogram(
                y=y,
                sr=sample_rate,
                n_mels=mel_bins,
                fmin=fmin,
                fmax=fmax,
                n_fft=n_fft,
                hop_length=hop_size,
            )
            logmels = librosa.power_to_db(mels)
            logmels = pad_sequences_last_value(logmels, timesteps)
            features.append(logmels.T)  # (timesteps, n_mels)
        else:
            for start_frame in range(0, len(y), config.clip_samples):
                end_frame = min(len(y), start_frame + config.clip_samples)
                seg_y = y[start_frame:end_frame]
                if len(seg_y) < min_samples:  # discard segments less than 1 second
                    continue
                mels = librosa.feature.melspectrogram(
                    y=seg_y,
                    sr=sample_rate,
                    n_mels=mel_bins,
                    fmin=fmin,
                    fmax=fmax,
                    n_fft=n_fft,
                    hop_length=hop_size,
                )
                logmels = librosa.power_to_db(mels)
                logmels = pad_sequences_last_value(logmels, timesteps)
                features.append(logmels.T)

        features = np.array(features, dtype=np.float32)  # n_segments, timesteps, n_mels
        return features  # timesteps, n_mels

    elif feature_name == "logmels-c":
        min_samples = 1 * 16000
        features = []
        (y, fs) = librosa.core.load(audio_path, sr=config.sample_rate, mono=True)
        y = normalize(y)
        if len(y) <= config.clip_samples:
            pad_length = config.clip_samples - len(y)
            y = np.pad(y, (0, pad_length), "constant")
            mels = librosa.feature.melspectrogram(
                y=y,
                sr=sample_rate,
                n_mels=mel_bins,
                fmin=fmin,
                fmax=fmax,
                n_fft=n_fft,
                hop_length=hop_size,
            )
            logmels = librosa.power_to_db(mels)
            resized_logmels = resize(logmels, (128, timesteps), mode="reflect")
            features.append(resized_logmels.T)  # (timesteps, n_mels)
        else:
            for start_frame in range(0, len(y), config.clip_samples):
                end_frame = min(len(y), start_frame + config.clip_samples)
                seg_y = y[start_frame:end_frame]
                if len(seg_y) < min_samples:  # discard segments less than 1 second
                    continue
                mels = librosa.feature.melspectrogram(
                    y=seg_y,
                    sr=sample_rate,
                    n_mels=mel_bins,
                    fmin=fmin,
                    fmax=fmax,
                    n_fft=n_fft,
                    hop_length=hop_size,
                )
                logmels = librosa.power_to_db(mels)
                resized_logmels = resize(logmels, (128, timesteps), mode="reflect")
                features.append(resized_logmels.T)

        features = np.array(features, dtype=np.float32)  # n_segments, timesteps, n_mels
        return features  # timesteps, n_mels

    else:
        stft = np.abs(librosa.stft(audio_vec))

        # Speech Energy
        rms = librosa.feature.rms(y=audio_vec + 0.0001)[0]  # incase of empty signal
        rms_mean = np.mean(rms, axis=-1).reshape(-1)
        rms_std = np.std(rms, axis=-1).reshape(-1)
        rms_max = np.max(rms, axis=-1).reshape(-1)

        # harmonics
        y_harmonic = (
            np.mean(librosa.effects.hpss(y=audio_vec)[0]) * 1000
        )  # harmonic (scaled by 1000)
        y_harmonic = y_harmonic.reshape(-1)

        # signal statistics
        sig_mean = np.mean(abs(audio_vec)).reshape(-1)
        sig_std = np.std(audio_vec).reshape(-1)

        # 12 chromas
        # chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        # chroma_mean = np.mean(chroma.T, axis=0).reshape(-1)

        # pause
        silence = 0
        for e in rms:
            if e <= 0.4 * rms_mean[0]:
                silence += 1
        silence /= float(len(rms))
        silence = np.array(silence, ndmin=1)

        # based on the pitch detection algorithm mentioned here:
        # http://access.feld.cvut.cz/view.php?cisloclanku=2009060001
        # Pitch
        cl = 0.45 * sig_mean
        center_clipped = []
        for s in audio_vec:
            if s >= cl:
                center_clipped.append(s - cl)
            elif s <= -cl:
                center_clipped.append(s + cl)
            elif np.abs(s) < cl:
                center_clipped.append(0)
        auto_corrs = librosa.core.autocorrelate(np.array(center_clipped, dtype=object))
        auto_corrs_max = 1000 * np.max(auto_corrs) / len(auto_corrs)
        auto_corrs_max = np.array(auto_corrs_max, ndmin=1)
        auto_corrs_std = np.array(np.std(auto_corrs), ndmin=1)

        if feature_name == "handcrafted":
            return np.concatenate(
                (
                    sig_mean,
                    sig_std,
                    rms_mean,
                    rms_std,
                    silence,
                    y_harmonic,
                    auto_corrs_std,
                    auto_corrs_max,
                )
            )

        elif feature_name == "melspec-image":
            fig, ax = plt.subplots()
            librosa.display.specshow(
                logmels,
                x_axis="time",
                y_axis="mel",
                sr=sample_rate,
                fmin=fmin,
                fmax=fmax,
                ax=ax,
                cmap="magma",
            )
            img_dir = os.path.join(workspace, "images")
            img_path = os.path.join(img_dir, audio_name + ".png")
            fig.savefig(img_path)
            # img_arr = read_image(img_path)
            img_arr = Image.open(img_path).convert("RGB")
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            # save the transofrmed image to file
            img_trans = transform(img_arr)
            img_trans_arr = np.array(img_trans)
            im = Image.fromarray(img_trans_arr, "RGB")
            im.save(os.path.join(img_dir, audio_name + "-feat.png"))
            return img_arr


def create_hdf5_files(df: pd.DataFrame, name: str, args: argparse.Namespace):
    workspace = args.workspace
    minidata = args.minidata
    test = args.test
    noisy = args.noisy

    sample_rate = config.sample_rate
    mel_bins = config.mel_bins
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    lb_to_idx = config.lb_to_idx
    feature_name = config.feature_name

    if minidata:
        packed_hdf5_path = os.path.join(
            workspace, "features", feature_name + "_" + name + "_minidata_waveform.h5"
        )
    else:
        packed_hdf5_path = os.path.join(
            workspace, "features", feature_name + "_" + name + "_waveform.h5"
        )
    create_folder(os.path.dirname(packed_hdf5_path))

    audio_names = df["wavfile"]
    audio_paths = df["audio-path"]
    targets = df["noisy-label"]
    ground_truth = df["ground-truth"]
    soft_gt = df["soft-gt"]

    meta_dict = {
        "audio_name": np.array(audio_names),
        "audio_path": np.array(audio_paths),
        "target": np.array([lb_to_idx[gt] for gt in ground_truth]),
        "gt": np.array([lb_to_idx[gt] for gt in ground_truth]),
    }

    total = []
    for x in soft_gt:
        row = []
        for num in x.strip()[1:-1].split():
            row.append(float(num))
        total.append(row)
    meta_dict["soft-gt"] = np.array(total)

    if noisy and name == "train":  # train noisy or clean
        meta_dict["target"] = np.array([lb_to_idx[target] for target in targets])

    if minidata:
        mini_num = 10
        total_num = len(meta_dict["audio_name"])
        random_state = np.random.RandomState(1234)
        indexes = random_state.choice(total_num, size=mini_num, replace=False)
        for key in meta_dict.keys():
            meta_dict[key] = meta_dict[key][indexes]

    audios_num = len(meta_dict["audio_name"])

    feature_time = time.time()
    with h5py.File(packed_hdf5_path, "w") as hf:
        features_grp = hf.create_group("features")
        info_grp = hf.create_group("info")

        info_grp.create_dataset(
            name="audio_name",
            shape=(audios_num,),
            maxshape=(None,),
            dtype="S80",
            chunks=True,
        )

        info_grp.create_dataset(
            name="audio_path",
            shape=(audios_num,),
            maxshape=(None,),
            dtype="S200",
            chunks=True,
        )

        info_grp.create_dataset(
            name="target",
            shape=(audios_num, classes_num),
            maxshape=(None, classes_num),
            dtype=np.float32,
            chunks=True,
        )

        info_grp.create_dataset(
            name="gt",
            shape=(audios_num, classes_num),
            maxshape=(None, classes_num),
            dtype=np.float32,
            chunks=True,
        )

        info_grp.create_dataset(
            name="soft-gt",
            shape=(audios_num, classes_num),
            maxshape=(None, classes_num),
            dtype=np.float32,
            chunks=True,
        )

        avg_time = 0
        max_time = 0

        idx = 0
        for n in trange(audios_num):
            # Get features from audio
            audio_name = meta_dict["audio_name"][n]
            audio_path = meta_dict["audio_path"][n]
            (audio, fs) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

            duration = librosa.get_duration(y=audio, sr=sample_rate)
            avg_time += duration
            max_time = max(max_time, duration)

            # audio = pad_truncate_sequence(audio, clip_samples)

            features = get_features(audio, audio_name, audio_path)

            if not test:
                total_segments = len(features)
                new_shape = info_grp["audio_name"].shape[0] + (total_segments - 1)
                info_grp["audio_name"].resize(size=(new_shape,))
                info_grp["audio_path"].resize(size=(new_shape,))
                info_grp["target"].resize(size=(new_shape, classes_num))
                info_grp["gt"].resize(size=(new_shape, classes_num))
                info_grp["soft-gt"].resize(size=(new_shape, classes_num))
                for seg_features in features:
                    features_grp.create_dataset(str(idx), data=seg_features)
                    info_grp["audio_name"][idx] = audio_name.encode()
                    info_grp["audio_path"][idx] = audio_path.encode()
                    info_grp["target"][idx] = to_one_hot(
                        meta_dict["target"][n], classes_num
                    )
                    info_grp["gt"][idx] = to_one_hot(meta_dict["gt"][n], classes_num)
                    info_grp["soft-gt"][idx] = meta_dict["soft-gt"][n]
                    idx += 1
            else:
                features_grp.create_dataset(str(idx), data=features)

                info_grp["audio_name"][idx] = audio_name.encode()
                info_grp["audio_path"][idx] = audio_path.encode()
                info_grp["target"][idx] = to_one_hot(
                    meta_dict["target"][n], classes_num
                )
                info_grp["gt"][idx] = to_one_hot(meta_dict["gt"][n], classes_num)
                info_grp["soft-gt"][idx] = meta_dict["soft-gt"][n]
                idx += 1

    avg_time /= audios_num
    print("Avg time length of audio", avg_time)
    print("Write hdf5 to {}".format(packed_hdf5_path))
    print("Time: {:.3f} s".format(time.time() - feature_time))


def upsample(classes: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Upsample data by duplicating samples.
    """
    for target, num in classes.items():
        sampled_df = df.loc[df["ground-truth"] == config.idx_to_lb[target]].sample(
            num, random_state=5, replace=True
        )
        df = pd.concat([df, sampled_df])

    return df


def create_dataset(args):
    dataset_csv = args.dataset_csv
    df = pd.read_csv(dataset_csv)

    # train or test data
    test = args.test

    if not test:
        # duplicate samples (upsampling)
        df = upsample(config.upsample, df)
        # create train/val splits
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=0,
            stratify=df[["ground-truth"]],
            shuffle=True,
        )
        create_hdf5_files(train_df, "train", args)
        create_hdf5_files(val_df, "val", args)
    else:
        create_hdf5_files(df, "test", args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest="mode")

    # Calculate log mel features for all audio files
    parser_pack_audio = subparsers.add_parser("pack_audio_files_to_hdf5")
    parser_pack_audio.add_argument(
        "--dataset_csv", type=str, required=True, help="CSV file of dataset."
    )
    parser_pack_audio.add_argument(
        "--workspace", type=str, required=True, help="Directory of your workspace."
    )
    parser_pack_audio.add_argument(
        "--minidata",
        action="store_true",
        default=False,
        help="Set True for debugging on a small part of data.",
    )
    parser_pack_audio.add_argument(
        "--noisy", action="store_true", default=False, help="Noisy train or clean train"
    )
    parser_pack_audio.add_argument(
        "--test", action="store_true", default=False, help="Train or test data"
    )

    # Parse arguments
    args = parser.parse_args()

    if args.mode == "pack_audio_files_to_hdf5":
        create_dataset(args)

    else:
        raise Exception("Incorrect arguments!")
