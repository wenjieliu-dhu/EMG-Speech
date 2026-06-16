import os
import sys
import torch
import torch.nn as nn
from diffwave.model import DiffWave
from base_network import get_mask_conv, get_fingerprint
from utils import load_model_state
from diffwave.params import AttrDict, params as base_params
from utils import *
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset import LJSpeech
import argparse
from pystoi import stoi
from torch_pesq import PesqLoss
from decoder import WatermarkExtractor
import random
from skimage.metrics import structural_similarity as ssim
import librosa
import Attacks as Attacker

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def hook_fn(module, args):
    return [args[0],fingerprint]


def hook_fn(module, args):
    return [args[0], fingerprint]


def get_fin_model(network, args):

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
                setattr(temp_layer, name_part, get_mask_conv(getattr(temp_layer, name_part), device))
            else:
                temp_layer = getattr(temp_layer, name_part)

    print(f"Watermark embedding complete for {len(output_layers)} Conv1d layers.")

    network = load_model_state(network, torch.load(os.path.join(args.save_mask_path, args.mask_name + '_epoch_2.pth')))

    for p in network.parameters():
        p.requires_grad = False

    for name, layer in reversed(list(network.named_modules())):
        if isinstance(layer, get_mask_conv):
            name_list = name.split('.')
            temp_layer = network
            for i, name_part in enumerate(name_list):
                if i == len(name_list) - 1:
                    setattr(temp_layer, name_part, get_fingerprint(getattr(temp_layer, name_part), args.finger_dim))
                else:
                    temp_layer = getattr(temp_layer, name_part)

    for name, layer in reversed(list(network.named_modules())):
        if isinstance(layer, get_fingerprint):
            layer.w_mask[layer.w_mask < 3] = 0
            layer.w_mask[layer.w_mask >= 3] = 6
            layer.set_mask()
            print("Watermarking kernel size and embedding rate:")
            print(layer.ori_weight.size())
            print(layer.get_rate())

    network = network.cuda()
    network.eval()

    for name, layer in network.named_modules():
        if isinstance(layer, get_fingerprint):
            layer.register_forward_pre_hook(hook_fn)

    return network


def audio_to_mel_spectrogram(audio, sample_rate=22050, n_mels=128):
    """
    Convert the audio waveform to Mel spectrogram.
    """
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
    return mel_spectrogram


def stft_loss_fn(x, y, n_fft=512, hop_length=256, win_length=512):
    """
    Compute the L1 loss between STFT magnitudes of two signals.
    """
    stft_x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)
    stft_y = torch.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)

    return torch.mean(torch.abs(torch.abs(stft_x) - torch.abs(stft_y)))


def compute_loss(new_audio_fake, ori_audio_fake, alpha=1, beta=1):
    mse_loss_fn = nn.MSELoss()
    mse_loss = mse_loss_fn(new_audio_fake, ori_audio_fake)

    stft_loss = stft_loss_fn(new_audio_fake, ori_audio_fake)

    return alpha * mse_loss + beta * stft_loss


mos_func = PesqLoss(0.5, sample_rate=22050)
dw_args = cal_param()


def train_model(model, extraction_model, ori_model, train_loader, args):
    f_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=1e-5,
                                   betas=(0.9, 0.99),
                                   weight_decay=1e-5)
    e_optimizer = torch.optim.Adam(extraction_model.parameters(), lr=1e-5)

    d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(f_optimizer, T_max=10, eta_min=1e-6)
    e_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(e_optimizer, T_max=10, eta_min=1e-6)

    bce_loss_fn = nn.BCELoss()

    attacker = Attacker()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        total_loss = 0.0
        watermark_loss = 0.0
        total_accuracy = 0.0
        total_stoi = 0.0
        total_mos = 0.0
        total_samples = 0
        total_ssim = 0.0

        with tqdm(train_loader, total=len(train_loader), ncols=0) as pbar:
            for index, mel_spectrogram in enumerate(pbar):
                mel_spectrogram = mel_spectrogram.to(device)

                fingers = torch.randint(0, 2, (args.batch_size, finger_dim), dtype=torch.float).to(device)
                global fingerprint
                fingerprint = fingers

                with torch.no_grad():
                    ori_audio = audio_sample(ori_model, mel_spectrogram, device, dw_args)

                new_audio = audio_sample(model, mel_spectrogram, device, dw_args)

                attacker = Attacker()

                attack_choice = random.randint(0, 6)
                attacked_audio = attacker(new_audio, attack_choice, flag=True)

                g_loss = compute_loss(new_audio, ori_audio)

                # extracted_watermark = extraction_model(new_audio.unsqueeze(1))

                attack_audio = attacked_audio.unsqueeze(1).to(device)
                extracted_watermark = extraction_model(attack_audio)

                w_loss = bce_loss_fn(extracted_watermark, fingers)

                total_loss = g_loss + 1 * w_loss

                f_optimizer.zero_grad()
                e_optimizer.zero_grad()

                total_loss.backward()

                f_optimizer.step()
                e_optimizer.step()

                extracted_watermark_binary = (extracted_watermark >= 0.5).float()
                correct_bits = (extracted_watermark_binary == fingers).sum().item()

                total_bits = torch.numel(fingers)
                batch_accuracy = correct_bits / total_bits
                total_accuracy += batch_accuracy * args.batch_size
                total_samples += args.batch_size

                aud_numpy = ori_audio[-1, :].cpu().detach().numpy()
                wmd_numpy = new_audio[-1, :].cpu().detach().numpy()
                stoi_score = stoi(aud_numpy, wmd_numpy, 22050)
                mos_score = mos_func.mos(ori_audio.cpu(), new_audio.cpu())

                ori_mel = audio_to_mel_spectrogram(aud_numpy)
                new_mel = audio_to_mel_spectrogram(wmd_numpy)
                ssim_score = ssim(ori_mel, new_mel, data_range=1.0)

                total_stoi += stoi_score * args.batch_size
                mos_score = mos_score.mean()
                total_mos += mos_score.item() * args.batch_size
                total_ssim += ssim_score * args.batch_size

                wandb.log({
                    'g_loss': g_loss.item(),
                    'watermark_loss': w_loss.item(),
                    'total_loss': total_loss.item(),
                    'acc': batch_accuracy,
                    'stoi_score': stoi_score,
                    'mos_score': mos_score.item(),
                    'ssim_score': ssim_score,
                    'epoch': epoch
                })

                if args.verbose and index % 500 == 0:
                    save_audio_path = os.path.join(args.save_path, f'{index}_generated.wav')
                    torchaudio.save(save_audio_path, new_audio.cpu(), sample_rate=16000)

                #                     wandb.log({
                #                         f"audio_{index}": wandb.Audio(save_audio_path, sample_rate=22050, caption=f"Generated Audio {index}")
                #                     })

                pbar.set_postfix({
                    'g_loss': g_loss.item(),
                    'watermark_loss': w_loss.item(),
                    'total_loss': total_loss.item(),
                    'stoi_score': stoi_score,
                    'mos_score': mos_score.item(),
                    'ssim_score': ssim_score,
                    'accuracy': batch_accuracy
                })

        epoch_accuracy = total_accuracy / total_samples
        epoch_stoi = total_stoi / total_samples
        epoch_mos = total_mos / total_samples
        epoch_ssim = total_ssim / total_samples

        print(f"Epoch {epoch + 1} - Watermark Extraction Accuracy: {epoch_accuracy * 100:.2f}%")
        print(f"Epoch {epoch + 1} - Average STOI: {epoch_stoi:.4f}, Average MOS: {epoch_mos:.4f}")
        print(f"Epoch {epoch + 1} - Average SSIM: {epoch_ssim:.4f}")

        torch.save(model.state_dict(), os.path.join(args.save_watermark_path,
                                                    f'{args.save_name}_epoch_{epoch + 1}.pth'))
        torch.save(extraction_model.state_dict(),
                   os.path.join(args.save_watermark_path, f'{args.save_name}_extractor_epoch_{epoch + 1}.pth'))
        wandb.save(os.path.join(args.save_watermark_path, f'{args.save_name}_epoch_{epoch + 1}.pth'))
        wandb.save(os.path.join(args.save_watermark_path, f'{args.save_name}_extractor_epoch_{epoch + 1}.pth'))

        d_scheduler.step()
        e_scheduler.step()


def main(args):

    global finger_dim
    finger_dim = args.finger_dim

    wandb.init(
        project="diffwave-wm",
        name=args.save_name,
        config=args
    )

    train_dataset = LJSpeech(root=args.root_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)

    ori_model = DiffWave(AttrDict(base_params))
    ori_checkpoint = torch.load(args.model_checkpoint_path)
    ori_model.load_state_dict(ori_checkpoint['model'])
    ori_model.eval()
    ori_model.to(device)

    model = DiffWave(AttrDict(base_params))
    model.to(device)
    model.load_state_dict(ori_checkpoint['model'])
    model = get_fin_model(model, args)
    watermarked_checkpoint = torch.load(os.path.join(args.save_watermark_path, 'diffwave-watermark_epoch_61.pth'))
    model.load_state_dict(watermarked_checkpoint)
    model.to(device)

    extraction_model = WatermarkExtractor(num_bits=args.finger_dim)
    extraction_model_checkpoint = torch.load(
        os.path.join(args.save_watermark_path, 'diffwave-watermark_extractor_epoch_61.pth'))
    extraction_model.load_state_dict(extraction_model_checkpoint)
    extraction_model.to(device)

    train_model(model, extraction_model, ori_model, train_loader, args)


def parse_args():
    parser = argparse.ArgumentParser(description='DiffWave Watermark Embedding and Extraction')

    parser.add_argument('--gpu', type=str, help='gpu id to train')
    parser.add_argument('--root-path', type=str, help='path to the dataset', default='/root/tmp/LJSpeech-1.1/wavs')
    parser.add_argument('--save-path', type=str, default='/root/LJSpeech-1.1/result', help='path to save generated audio files')
    parser.add_argument('--save-name', help='dir of checkpoints to save', default='diffwave-watermark')
    parser.add_argument('--save-watermark-path', type=str, default='/root/diffwave-master/src/checkpoints', help='path to save models with watermarks')
    parser.add_argument('--mask-name', type=str, default='diffwave_mask', help='prefix name for saving watermarked model checkpoints')
    parser.add_argument('--save-mask-path', type=str, help='path of mask checkpoints',default='/root/diffwave-master/src/checkpoints')
    parser.add_argument('--model-checkpoint-path', type=str, help='path to the pretrained model checkpoint', default='/root/diffwave-master/src/diffwave/model/diffwave.pt')
    parser.add_argument('--finger-dim', type=int, default=16, help='lenth of watermarking')
    parser.add_argument('--epochs', type=int, default=500, help='number of total epochs to train')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for training')
    parser.add_argument('--verbose', action='store_true', default=True, help='whether to save generated audio during training (default: False)')


    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
