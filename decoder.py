import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor


class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, device, out_channels, kernel_size, in_channels=1, sample_rate=24000):
        super(SincConv, self).__init__()

        if in_channels != 1:
            raise ValueError("SincConv only supports one input channel")

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.device = device

        if kernel_size % 2 == 0:
            self.kernel_size += 1

        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        filbandwidthsmel = np.linspace(np.min(fmel), np.max(fmel), self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)
        self.mel = filbandwidthsf
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)

    def forward(self, x):
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(2 * fmax * self.hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * np.sinc(2 * fmin * self.hsupp / self.sample_rate)
            hideal = hHigh - hLow
            self.band_pass[i, :] = torch.Tensor(np.hamming(self.kernel_size)) * torch.Tensor(hideal)

        band_pass_filter = self.band_pass.to(self.device)
        filters = band_pass_filter.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(Residual_block, self).__init__()
        self.first = first
        self.bn1 = nn.BatchNorm1d(nb_filts[0]) if not first else None
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        self.conv1 = nn.Conv1d(nb_filts[0], nb_filts[1], kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm1d(nb_filts[1])
        self.conv2 = nn.Conv1d(nb_filts[1], nb_filts[1], kernel_size=3, padding=1, stride=1)
        self.downsample = (nb_filts[0] != nb_filts[1])
        self.conv_downsample = nn.Conv1d(nb_filts[0], nb_filts[1], kernel_size=1, stride=1) if self.downsample else None
        self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


class WatermarkExtractor(nn.Module):
    def __init__(self, num_bits=64, device='cuda'):
        super(WatermarkExtractor, self).__init__()
        self.device = device

        self.Sinc_conv = SincConv(device=self.device, out_channels=80, kernel_size=251, in_channels=1)

        self.block1 = Residual_block(nb_filts=[80, 160], first=True)
        self.block2 = Residual_block(nb_filts=[160, 320])
        self.block3 = Residual_block(nb_filts=[320, 640])

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.attn1 = self.fc(160)
        self.attn2 = self.fc(320)
        self.attn3 = self.fc(640)

        self.gru = nn.GRU(input_size=640, hidden_size=256, num_layers=2, batch_first=True)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_bits)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = self.Sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)

        x1 = self.block1(x)
        attn1 = self.sigmoid(self.attn1(self.avgpool(x1).squeeze(-1)))
        x1 = x1 * attn1.view(attn1.size(0), attn1.size(1), 1)

        x2 = self.block2(x1)
        attn2 = self.sigmoid(self.attn2(self.avgpool(x2).squeeze(-1)))
        x2 = x2 * attn2.view(attn2.size(0), attn2.size(1), 1)

        x3 = self.block3(x2)
        attn3 = self.sigmoid(self.attn3(self.avgpool(x3).squeeze(-1)))
        x3 = x3 * attn3.view(attn3.size(0), attn3.size(1), 1)

        x3 = x3.permute(0, 2, 1)
        self.gru.flatten_parameters()
        _, h_n = self.gru(x3)

        h_n = h_n[-1]
        x = F.relu(self.fc1(h_n))
        watermark = self.sigmoid(self.fc2(x))

        return watermark

    def fc(self, in_features):
        return nn.Linear(in_features, in_features)




