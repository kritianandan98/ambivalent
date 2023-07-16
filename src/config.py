import math 

workspace = '/home/kriti/ambivalent'

# Feature config
sample_rate = 16000
clip_samples = sample_rate * 5
mel_bins = 128
n_mfcc = 12 # set to 12 for all-timesteps, all-fixed and 128 for mfccs
fmin = 0
fmax = 8000
n_fft = 512
hop_size = 256
timesteps = math.ceil(clip_samples / hop_size)
device = 'cuda'
feature_name = "logmels-a" # hand-crafted, mean-mfcc, log-mels, all-timesteps, all-fixed, audio, logmels-a

# hyperparameters
run_name = "BiLSTM-1e-4-b64-logmels-a" # mlp, 
epochs = 100
model_type = "BiLSTM"
loss_type = "cce"
learning_rate = 1e-4
batch_size = 16
grad_accum = 8 # accumulate every xth batch
num_workers = 8
pretrained_checkpoint_path = None
freeze_base = None
augmentation = 'none'
segment = True

# classes
labels = ["ang", "exc", "fru", "hap", "neu", "sad"]
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}
classes_num = len(labels)