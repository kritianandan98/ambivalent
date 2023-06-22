workspace = '/home/kriti/ambivalent'

# Feature config
sample_rate = 16000
clip_samples = sample_rate * 5
mel_bins = 128
n_mfcc = 12
fmin = 50
fmax = 8000
n_fft = 2048
hop_size = 512
device = 'cuda'
feature_name = "all-fixed"

# hyperparameters
run_name = "DNN-all-1e-3-weighted-cce"
epochs = 200
model_type = "DNN"
loss_type = "cce"
learning_rate = 1e-3
batch_size = 256
num_workers = 8
pretrained_checkpoint_path = None
freeze_base = None
augmentation = 'none'

# classes
labels = ["ang", "exc", "fru", "hap", "neu", "sad"]
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}
classes_num = len(labels)