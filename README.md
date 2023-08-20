# Ambivalent: self-label correction (ProSelfLC) for SER
This repository contains the experiments carried out for the project topic *Self-label correction for noisy labels in Speech
Emotion Recognition*.

## Installation
``` 
pip install requirements.txt
```

## Dataset Creation

Dataset creation is described in detail in section 3.1. IEMOCAP dataset was obtained by following the instructions on [this](https://sail.usc.edu/iemocap/iemocap_release.htm) website. 

`data-explore.ipynb` contains information about the creation of noisy datasets.

## Experiments

### Experiment: I
To run experiment I, edit `config.py` according to the settings mentioned in section. Edit the train paths in `train-pytorch.sh` to train the appropriate models. For training, `hard-26.48-train.csv` is used for noisy-A and `hard-40.19-train.csv` is used for noisy-B models. Use the `--noisy` option when training with noisy labels instead of clean labels.

For testing, use `test.sh`. You can use either `hard-26.48-test.csv` or `hard-40.19-test.csv` (both are the same) as test paths.

To validate the ML-algorithms experiment, use `train-sklearn.sh` with `feature_name = handcrafted`.

`results_analysis.ipynb` contains the code to reproduce the classification report and confusion matrices for the given config

Set an appropritate `run_name` in `config.py` for all experiment runs.

> All experiments use [wandb](https://wandb.ai/site) for tracking, turning it to offline mode will default to not saving any model.

### Experiment: II

To run experiments with ProSelfLC, change `loss_type` under `config.py` to `proselflc`. 

Use `results_analysis.ipynb` to reproduce the results as mentioned previously.

Use `pretty_plots.ipynb` to generate the line graphs mentioned in section 4.2.1 and 4.2.3. These require the CSV logs from wandb cloud.

### Experiment: III

Use `results_analysis.ipynb` to generate the gender-specific confusion matrices and classification results.

Each train run saves a file called `train-predictions-<model-name>-<hyperparameter-setting>.csv` file that contains information about labels that were flipped during training. Use `results_analysis.ipynb` to get the label pairs that were flipped sorted in descending order during the specific run.


## Reference

Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. "Panns: Large-scale pretrained audio neural networks for audio pattern recognition." IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894. 

Wang, X., Hua, Y., Kodirov, E., Clifton, D.A., and Robertson, N.M. "Proselflc: Progressive self label correction for training robust deep neural networks." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 752-761, 2021.
