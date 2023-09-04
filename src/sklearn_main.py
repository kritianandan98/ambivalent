"""
File to train and test the model using ML algorithms
"""
import os
import h5py
import numpy as np
import pandas as pd
import argparse
import logging
import xgboost as xgb

import torch
import torch.nn as nn
from . import config

from .utils import get_filename, create_logging
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# from torchsummary import summary

os.environ["WANDB_MODE"] = "online"


def upsample(classes: dict, features, targets):
    temp_df = pd.DataFrame(
        data=np.concatenate([features, targets.reshape(-1, 1)], axis=-1),
        columns=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "target"],
    )

    for target, num in classes.items():
        sampled_df = temp_df.loc[temp_df["target"] == target].sample(
            num, random_state=5, replace=True
        )
        temp_df = pd.concat([temp_df, sampled_df])

    X = temp_df.to_numpy()[:, :-1]
    y = temp_df.to_numpy()[:, -1]
    y = y.astype("int")

    return X, y


def get_data(hdf5_path):
    features = []
    targets = []
    gt = []
    with h5py.File(hdf5_path, "r") as hf:
        for index in range(len(hf["info"]["audio_name"])):
            audio_name = hf["info"]["audio_name"][index].decode()
            features.append(hf["features"][str(index)][:])
            targets.append(hf["info"]["target"][index].astype(np.float32))
            gt.append(hf["info"]["gt"][index].astype(np.float32))
    return (
        np.array(features),
        np.argmax(np.array(targets, dtype="int"), axis=-1),
        np.argmax(np.array(gt, dtype="int"), axis=-1),
    )


def main(args):
    # Arugments & parameters
    workspace = args.workspace
    filename = args.filename
    resume_epoch = args.resume_epoch
    minidata = args.minidata

    run_name = config.run_name
    model_type = config.model_type
    pretrained_checkpoint_path = config.pretrained_checkpoint_path
    freeze_base = config.freeze_base
    loss_type = config.loss_type
    augmentation = config.augmentation
    batch_size = config.batch_size
    run_name = config.run_name
    classes_num = config.classes_num
    feature_name = config.feature_name
    labels = config.labels
    pretrain = True if pretrained_checkpoint_path else False

    if minidata:
        train_hdf5_path = os.path.join(
            workspace, "features", feature_name + "_" + "train_minidata_waveform.h5"
        )
        val_hdf5_path = os.path.join(
            workspace, "features", feature_name + "_" + "val_minidata_waveform.h5"
        )
    else:
        train_hdf5_path = os.path.join(
            workspace, "features", feature_name + "_" + "train_waveform.h5"
        )
        val_hdf5_path = os.path.join(
            workspace, "features", feature_name + "_" + "val_waveform.h5"
        )

    test_hdf5_path = os.path.join(
        workspace, "features", feature_name + "_" + "test_waveform.h5"
    )

    logs_dir = os.path.join(
        workspace,
        "logs",
        filename,
        model_type,
        "{}".format(run_name),
        "pretrain={}".format(pretrain),
        "loss_type={}".format(loss_type),
        "augmentation={}".format(augmentation),
        "batch_size={}".format(batch_size),
        "freeze_base={}".format(freeze_base),
    )
    create_logging(logs_dir, "w")
    logging.info(args)

    logging.info(f"Loading model: {model_type}, classes: {classes_num}")

    # Parallel
    print("GPU number: {}".format(torch.cuda.device_count()))

    train_X, _, train_y = get_data(train_hdf5_path)

    train_X, train_y = upsample({0: 100, 1: 300, 3: 600, 5: 200}, train_X, train_y)

    val_X, _, val_y = get_data(val_hdf5_path)
    test_X, _, test_y = get_data(test_hdf5_path)

    full_X = np.concatenate([train_X, val_X], axis=0)
    # print("full X", full_X.shape)
    full_y = np.concatenate([train_y, val_y], axis=0)
    # print("full y", full_y.shape)

    emotion_dict = {"ang": 0, "exc": 1, "fru": 2, "hap": 3, "neu": 4, "sad": 5}

    if model_type == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(650,),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size="auto",
            learning_rate="adaptive",
            learning_rate_init=0.01,
            power_t=0.5,
            max_iter=1000,
            shuffle=True,
            random_state=None,
            tol=0.0001,
            verbose=False,
            warm_start=True,
            momentum=0.8,
            nesterovs_momentum=True,
            early_stopping=False,
            validation_fraction=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
        )
    elif model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=1200,
            min_samples_split=25,
            class_weight="balanced",
            verbose=True,
        )
    elif model_type == "xgboost":
        model = xgb.XGBClassifier(
            max_depth=7,
            learning_rate=0.008,
            objective="multi:softprob",
            n_estimators=1200,
            sub_sample=0.8,
            num_class=len(emotion_dict),
            booster="gbtree",
            n_jobs=4,
        )

    model = model.fit(train_X, train_y)
    predictions = model.predict(train_X)

    train_acc = accuracy_score(train_y, predictions)
    train_f1 = f1_score(train_y, predictions, average="weighted")

    logging.info("Train accuracy: {:.3f}".format(train_acc))
    logging.info("Train F1: {:.3f}".format(train_f1))

    predictions = model.predict(val_X)

    val_acc = accuracy_score(val_y, predictions)
    val_f1 = f1_score(val_y, predictions, average="weighted")

    logging.info("Val accuracy: {:.3f}".format(val_acc))
    logging.info("Val F1: {:.3f}".format(val_f1))

    logging.info("Val report")
    print(classification_report(y_true=val_y, y_pred=predictions, target_names=labels))

    predictions = model.predict(test_X)

    logging.info("Test report")
    print(classification_report(y_true=test_y, y_pred=predictions, target_names=labels))

    test_acc = accuracy_score(test_y, predictions)
    test_f1 = f1_score(test_y, predictions, average="weighted")

    logging.info("Test accuracy: {:.3f}".format(test_acc))
    logging.info("Test F1: {:.3f}".format(test_f1))

    logging.info("Storing predictions...")
    # store results in a CSV
    df = pd.DataFrame(columns=["wavfile", "ground-truth", "prediction", "audio-path"])

    with h5py.File(test_hdf5_path, "r") as hf:
        for i, audiopath in enumerate(hf["info"]["audio_name"]):
            df.loc[i, "wavfile"] = audiopath
            df.loc[i, "prediction"] = config.idx_to_lb[
                np.argmax(predictions[i], axis=-1)
            ]
            df.loc[i, "ground-truth"] = config.idx_to_lb[
                np.argmax(hf["info"]["gt"][i], axis=-1)
            ]  # store ground truth in
            df.loc[i, "audio-path"] = hf["info"]["audio_path"][i]

    df.to_csv("result.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example of parser. ")
    subparsers = parser.add_subparsers(dest="mode")

    # Train
    parser_train = subparsers.add_parser("train")
    parser_train.add_argument(
        "--workspace", type=str, required=True, help="Directory of your workspace."
    )
    parser_train.add_argument("--resume_epoch", type=int)
    parser_train.add_argument("--minidata", action="store_true", default=False)
    parser_train.add_argument("--cuda", action="store_true", default=False)

    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == "train":
        main(args)
    else:
        raise Exception("Error argument!")
