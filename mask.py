import os
import sys
import argparse
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset import LJSpeech
# Assume DiffWave model is already available
from diffwave.model import DiffWave
from base_network import get_mask_conv  # Assume get_mask_conv and VGG19Loss are defined
from diffwave.params import AttrDict, params as base_params
from utils import *
import wandb
def get_fingerprint_local(network):
    for p in network.parameters():
        p.requires_grad = False

    print("Finding layers for watermark embedding:")
    output_layers = []

    for name, layer in reversed(list(network.named_modules())):
        if isinstance(layer, nn.Conv1d) and 'output_projection' in name:
            output_layers.append((name, layer))
            if len(output_layers) == 5:
                break

    if len(output_layers) < 5:
        raise ValueError("Not enough output_projection layers to embed watermark.")

    for name, layer in output_layers:
        print(f"Embedding watermark in layer: {name}")
        name_list = name.split('.')
        temp_layer = network
        for i, name_part in enumerate(name_list):
            if i == len(name_list) - 1:
                setattr(temp_layer, name_part, get_mask_conv(getattr(temp_layer, name_part), 'cuda:0'))
            else:
                temp_layer = getattr(temp_layer, name_part)

    print(f"Watermark embedding complete for {len(output_layers)} Conv1d layers.")


def stft_loss_fn(x, y, n_fft=512, hop_length=256, win_length=512):
    """
    Compute the L1 loss between STFT magnitudes of two signals.
    """
    stft_x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)
    stft_y = torch.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)
    return torch.mean(torch.abs(torch.abs(stft_x) - torch.abs(stft_y)))

def compute_loss(new_audio_fake, ori_audio_fake, model, alpha=100, beta=10):
    mse_loss_fn = nn.MSELoss()
    mse_loss = mse_loss_fn(new_audio_fake, ori_audio_fake)
    stft_loss = stft_loss_fn(new_audio_fake, ori_audio_fake)
    return alpha * mse_loss + beta * stft_loss


dw_args = cal_param()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_model(model, ori_model, train_loader, args):
    f_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-4
    )

    for epoch in range(args.epochs):
        with tqdm(train_loader, total=len(train_loader), ncols=0) as pbar:
            for index, mel_spectrogram in enumerate(pbar):
                mel_spectrogram = mel_spectrogram.to(device)

                with torch.no_grad():
                    ori_audio = audio_sample(ori_model, mel_spectrogram, device, dw_args)

                new_audio = audio_sample(model, mel_spectrogram, device, dw_args)

                g_loss = compute_loss(new_audio, ori_audio, model)
                weight_loss, bias_loss = 0, 0

                for name, layer in model.named_modules():
                    if isinstance(layer, get_mask_conv):
                        weight_loss += layer.get_weight_mask_loss(layer.weight_mask)
                        bias_loss += layer.get_weight_mask_loss(layer.bias_mask)

                loss = g_loss + weight_loss + bias_loss

                f_optimizer.zero_grad()
                loss.backward()
                f_optimizer.step()

                wandb.log({
                    'g_loss': g_loss.item(),
                    'weight_loss': weight_loss.item(),
                    'bias_loss': bias_loss.item(),
                    'total_loss': loss.item(),
                    'epoch': epoch
                })

                # if args.verbose and index % 50 == 0:
                #     save_audio_path = os.path.join(args.save_path, f'{index}_generated.wav')
                #     torchaudio.save(save_audio_path, new_audio.cpu(), sample_rate=22050)

                pbar.set_postfix({
                    'g_loss': g_loss.item(),
                    'weight_loss': weight_loss.item(),
                    'bias_loss': bias_loss.item()
                })
        torch.save(model.state_dict(), os.path.join(args.save_mask_path, f'{args.mask_name}_epoch_{epoch}.pth'))


def main(args):
    wandb.init(
        project="diffwave-watermark",
        name=args.mask_name,
        config=args
    )
    train_dataset = LJSpeech(root=args.root_path)
    train_loader = DataLoader(train_dataset, batch_size=1, drop_last=True)
    ori_model = DiffWave(AttrDict(base_params))
    ori_checkpoint = torch.load(args.model_checkpoint_path)
    ori_model.load_state_dict(ori_checkpoint['model'])
    ori_model.eval()
    ori_model.to(device)

    model = DiffWave(AttrDict(base_params))
    model.load_state_dict(ori_checkpoint['model'])
    model.to(device)
    get_fingerprint_local(model)

    train_model(model, ori_model, train_loader, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help='gpu id to train')
    parser.add_argument('--root-path', type=str, help='path of dataset', default='/home/data/LJSpeech-1.1/wavs')
    parser.add_argument('--verbose', type=bool, default=True,help='enable or disable verbose mode to save generated audio during training (default: True)')
    parser.add_argument('--save-path', default='./results', help='path to save generated audio if verbose')
    parser.add_argument('--epochs', default=5, type=int, metavar='N', help='number of total epochs to train')
    parser.add_argument('--batch-size', default=1, type=int, metavar='N', help='number of batch size to train')
    parser.add_argument('--mask-name', type=str, help='name of saved tuned checkpoints',default='diffwave_mask')
    parser.add_argument('--save-mask-path', default='./checkpoints', help='path to save mask checkpoints')
    parser.add_argument('--model-checkpoint-path', type=str, help='path to original model checkpoint',default='/home/diffwave-master/src/diffwave/model/diffwave.pt')

    args = parser.parse_args()

    main(args)