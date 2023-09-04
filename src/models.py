"""
Model architectures (CNN6, CNN10, CNN14)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import do_mixup
from . import config
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class MLP(nn.Module):
    def __init__(self, classes_num):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(8, 650)
        # self.l2 = nn.Linear(2000, 1000)
        # self.l3 = nn.Linear(1000, 500)
        # self.l4 = nn.Linear(500, 100)
        # self.l5 = nn.Linear(100, 50)

        self.l6 = nn.Linear(650, classes_num)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):  # flattening (n, 1, 28, 28)--> (n, 784)
        # print(x.shape)
        x = F.relu(self.l1(x))
        # x = self.dropout(x)
        # x = F.relu(self.l2(x))
        # x = self.dropout(x)
        # x = F.relu(self.l3(x))
        # x = F.relu(self.l4(x))
        # x = F.relu(self.l5(x))
        x = self.l6(x)
        return x


class LSTM(nn.Module):
    def __init__(self, classes_num):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=config.mel_bins, hidden_size=150, num_layers=3, batch_first=True
        )

        # The linear layer that maps from hidden state space to tag space
        self.hidden2out = nn.Linear(in_features=150, out_features=classes_num)

    def forward(self, input_seq):
        lstm_out, (hidden, ct) = self.lstm(input_seq)
        x = self.hidden2out(hidden[-1])
        return x


class BiLSTM(nn.Module):
    def __init__(self, classes_num):
        super(BiLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=config.mel_bins,
            hidden_size=150,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
        )

        # The linear layer that maps from hidden state space to tag space
        self.pool = nn.AdaptiveAvgPool2d((1, 300))
        self.hidden2out = nn.Linear(in_features=300, out_features=classes_num)

    def forward(self, input_seq):
        out, (hidden, ct) = self.lstm(input_seq)
        out = self.pool(out)

        # squeeze the tensor to shape (batch_size, hidden_size) before feeding to Fully Connected Layer
        out = out.squeeze(1)
        out = self.hidden2out(out)
        out = F.softmax(out, -1)
        return out


class Wav2Vec2(nn.Module):
    def __init__(
        self,
        num_classes,
        upstream_model="wav2vec2",
        feature_dim=768,
        unfreeze_last_conv_layers=False,
    ):
        super().__init__()
        self.upstream = torch.hub.load("s3prl/s3prl", upstream_model)  # wav2vec2

        for param in self.upstream.parameters():
            param.requires_grad = False

        for param in self.upstream.model.encoder.layers.parameters():
            param.requires_grad = True

        if unfreeze_last_conv_layers:
            for param in self.upstream.model.feature_extractor.conv_layers[
                5:
            ].parameters():
                param.requires_grad = True

        self.fc = nn.Sequential(nn.Linear(feature_dim, num_classes))

    def forward(self, x):
        x = self.upstream(x)["last_hidden_state"]
        x = torch.mean(x, dim=1)
        out = self.fc(x)
        out = F.softmax(out, -1)
        return out


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(79872, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # batch_size, 1, time_steps, mel_bins
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class Cnn6(nn.Module):
    def __init__(self, classes_num):
        super(Cnn6, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        # x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        input = input.unsqueeze(1)  # (batch_size, 1, time_steps, mel_bins)

        x = input.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # if self.training:
        #    x = self.spec_augmenter(x)

        # Mixup on spectrogram
        # if self.training and mixup_lambda is not None:
        #    x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return clipwise_output


class Cnn10(nn.Module):
    def __init__(self, classes_num):
        super(Cnn10, self).__init__()

        self.bn0 = nn.BatchNorm2d(128)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        # self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
        #    freq_drop_width=8, freq_stripes_num=2)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        x = input.unsqueeze(1)  # (batch_size, 1, time_steps, mel_bins)
        # print(input.shape)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # if self.training:
        #    x = self.spec_augmenter(x)

        # Mixup on spectrogram
        # if self.training and mixup_lambda is not None:
        #    x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {"clipwise_output": clipwise_output, "embedding": embedding}

        return clipwise_output


class Cnn14(nn.Module):
    def __init__(
        self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num
    ):
        super(Cnn14, self).__init__()

        sample_rate = 16000
        clip_samples = sample_rate * 30

        mel_bins = 64
        fmin = 50
        fmax = 14000
        window_size = 512
        hop_size = 160
        window = "hann"
        pad_mode = "reflect"
        center = True
        device = "cuda"
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.softmax(self.fc_audioset(x))

        output_dict = {"clipwise_output": clipwise_output, "embedding": embedding}

        return output_dict


class Transfer_Cnn14(nn.Module):
    def __init__(
        self,
        sample_rate,
        window_size,
        hop_size,
        mel_bins,
        fmin,
        fmax,
        classes_num,
        freeze_base,
    ):
        """Classifier for a new task using pretrained Cnn14 as a sub module."""
        super(Transfer_Cnn14, self).__init__()
        audioset_classes_num = 527

        self.base = Cnn14(
            sample_rate,
            window_size,
            hop_size,
            mel_bins,
            fmin,
            fmax,
            audioset_classes_num,
        )

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint["model"])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)"""
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict["embedding"]

        clipwise_output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        output_dict["clipwise_output"] = clipwise_output

        return clipwise_output
