import os
import logging
import numpy as np
import torch
from torch import Tensor
from typing import List


def create_folder(fd: str):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path: str):
    path = os.path.realpath(path)
    na_ext = path.split("/")[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def get_sub_filepaths(folder: str) -> List:
    paths = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            paths.append(path)
    return paths


def move_data_to_device(x, device):
    if "float" in str(x.dtype):
        x = torch.Tensor(x)
    elif "int" in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def create_logging(log_dir: str, filemode: str):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, "{:04d}.log".format(i1))):
        i1 += 1

    log_path = os.path.join(log_dir, "{:04d}.log".format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        filename=log_path,
        filemode=filemode,
    )

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    return logging


def do_mixup(x: Tensor, mixup_lambda: float) -> Tensor:
    """
    Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes
    (1, 3, 5, ...).
    Args:
      x             : (batch_size * 2, ...)
      mixup_lambda  : (batch_size * 2,)
    Returns:
      out           : (batch_size, ...)
    """
    out = (
        x[0::2].transpose(0, -1) * mixup_lambda[0::2]
        + x[1::2].transpose(0, -1) * mixup_lambda[1::2]
    ).transpose(0, -1)
    return out


def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]


def forward(model, generator, return_input=False, return_target=False) -> dict:
    """
    Forward data to a model.

    Args:
        model         : object
        generator     : object
        return_input  : bool
        return_target : bool
    Returns:
        audio_name                    : (audios_num,)
        clipwise_output               : (audios_num, classes_num)
        (ifexist) segmentwise_output  : (audios_num, segments_num, classes_num)
        (ifexist) framewise_output    : (audios_num, frames_num, classes_num)
        (optional) return_input   : (audios_num, segment_samples)
        (optional) return_target  : (audios_num, classes_num)
    """
    output_dict = {}
    device = next(model.parameters()).device

    # Forward data to a model in mini-batches
    for n, batch_data_dict in enumerate(generator):
        # print(n)
        # batch_waveform = move_data_to_device(batch_data_dict['waveform'], device)
        batch_waveform = move_data_to_device(batch_data_dict["features"], device)
        with torch.no_grad():
            model.eval()
            batch_output = model(batch_waveform)

        append_to_dict(output_dict, "audio_name", batch_data_dict["audio_name"])

        append_to_dict(output_dict, "clipwise_output", batch_output.data.cpu().numpy())

        if return_input:
            append_to_dict(output_dict, "waveform", batch_data_dict["waveform"])

        if return_target:
            if "target" in batch_data_dict.keys():
                append_to_dict(output_dict, "target", batch_data_dict["target"])
            if "soft-gt" in batch_data_dict.keys():
                append_to_dict(output_dict, "soft-gt", batch_data_dict["soft-gt"])

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict


def interpolate(x, ratio):
    """
    Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.

    Args
        x         : (batch_size, time_steps, classes_num)
        ratio     : int, ratio to interpolate
    Returns
        upsampled : (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: Tensor, frames_num):
    """
    Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args
      framewise_output  : (batch_size, frames_num, classes_num)
      frames_num        : int, number of frames to pad
    Returns
      output            : (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1
    )
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


def expected_calibration_error_multiclass(
    y_true: np.ndarray, y_prob: np.ndarray, num_bins=10
) -> float:
    """
    Calculate the Expected Calibration Error (ECE) for a probabilistic multiclass classification model.

    Args
        y_true (numpy array): True labels (ground truth) as integers (0 to num_classes-1).
        y_prob (numpy array): Predicted probabilities for each class (shape: [n_samples, num_classes]).
        num_bins (int): Number of bins to divide the probability range into (default is 10).

    Returns
        float: Expected Calibration Error (ECE) value.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Calculate confidence bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)

    # Initialize variables to accumulate values for ECE calculation
    ece = 0.0
    bin_correct = np.zeros(num_bins)
    bin_total = np.zeros(num_bins)

    # Calculate predicted classes for each sample
    y_pred = np.argmax(y_prob, axis=1)

    # Calculate the confidence of the predictions (maximum probability)
    confidences = np.max(y_prob, axis=1)

    # Iterate over each prediction and calculate ECE
    for i in range(num_bins):
        # Find indices of predictions falling into the current bin
        bin_indices = np.logical_and(
            confidences >= bin_boundaries[i], confidences < bin_boundaries[i + 1]
        )

        # Count total number of predictions in the bin for each class
        bin_total[i] = np.sum(bin_indices)

        # Calculate the accuracy for this bin for each class
        if bin_total[i] > 0:
            bin_accuracy = np.mean(y_pred[bin_indices] == y_true[bin_indices])
            bin_correct[i] = bin_accuracy * bin_total[i]
            ece += (
                np.abs(bin_accuracy - np.mean(confidences[bin_indices])) * bin_total[i]
            )

    # Normalize ECE by the total number of predictions
    ece /= np.sum(bin_total)

    return ece


def brier_score_multiclass(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate the Brier score for multiclass classification.

    Args
        y_true : True probabilities for each class (shape: [n_samples, num_classes]).
        y_prob : Predicted probabilities for each class (shape: [n_samples, num_classes]).

    Returns
        brier_score : Brier score value.
    """

    # Calculate the mean squared difference between predicted probabilities and true labels
    brier_score = np.mean(np.sum((y_prob - y_true) ** 2, axis=1))

    return brier_score
