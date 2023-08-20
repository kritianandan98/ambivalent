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
feature_name = "audios-segmented" # 'audios-segmented' for Wav2Vec2, 'logmels-a' for BiLSTM, 'handcrafted' for ML-algo exp

# hyperparameters
run_name = "Wav2Vec2-final-baseline-noisy-hard-40.19" # mlp, 
epochs = 20
model_type = "Wav2Vec2" # BiLSTM, 
loss_type = "cce" # proselflc or cce
learning_rate = 1e-5
batch_size = 16
grad_accum = 8 # accumulate every xth batch
num_workers = 8
pretrained_checkpoint_path = None
freeze_base = None
augmentation = 'none'
segment = True

# classes (hard)
labels = ["ang", "exc", "fru", "hap", "neu", "sad"]
upsample = {0: 100, 1: 300, 3: 600, 5: 200}
#labels = ['ang', 'exc', 'neu', 'sad']
#upsample = {0: 100, 1: 300, 3: 200}
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}
classes_num = len(labels)