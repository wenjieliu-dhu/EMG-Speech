# EMG-Speech

Code release for the EMG-to-speech audio watermarking experiments. The repository contains the training, generation, and evaluation scripts used for the watermarking framework and baseline comparisons.

This public repository intentionally excludes model checkpoints, generated audio, intermediate mel/NumPy files, server packages, and experiment result folders.

## Repository Contents

- `mask.py`: trains the DiffWave mask modules used for watermark embedding.
- `watermark.py`: trains the watermarked DiffWave model and watermark extractor.
- `generate_system_wavs.py`: generates Baseline, Post-hoc, and Ours audio from mel spectrogram manifests.
- `evaluate_main_experiment.py`: evaluates audio quality and watermark metrics.
- `evaluate_wm_attack_robustness.py`: evaluates watermark robustness under attacks.
- `strong_watermark_baselines.py`: traditional watermarking baselines.
- `diffwave/`: DiffWave source code used by the experiments. Pretrained weights are not included.
- `audioseal-main/` and `wavmark-main/`: baseline implementations used for comparison.

## Environment

Python 3.8+ and CUDA-capable PyTorch are recommended.

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from the official PyTorch instructions if the default package does not match your GPU environment.

## Data And Checkpoints

The following artifacts are required to reproduce the full experiments but are not stored in this repository:

- training/evaluation audio data
- mel spectrogram `.npy` files
- DiffWave pretrained checkpoint
- trained mask, watermarked model, and extractor checkpoints
- generated audio and result CSV files

Expected local layout:

```text
data/wavs/
data/mels/
checkpoints/diffwave.pt
checkpoints/diffwave_mask_epoch_2.pth
checkpoints/diffwave-watermark_epoch_61.pth
checkpoints/diffwave-watermark_extractor_epoch_61.pth
```

If you use a different layout, pass paths explicitly with the command-line arguments shown below.

## Training

Train the mask modules:

```bash
python mask.py \
  --root-path data/wavs \
  --model-checkpoint-path checkpoints/diffwave.pt \
  --save-mask-path checkpoints \
  --mask-name diffwave_mask \
  --epochs 5
```

Train the watermarked model and extractor:

```bash
python watermark.py \
  --root-path data/wavs \
  --model-checkpoint-path checkpoints/diffwave.pt \
  --save-mask-path checkpoints \
  --save-watermark-path checkpoints \
  --mask-name diffwave_mask \
  --save-name diffwave-watermark \
  --finger-dim 16 \
  --epochs 500
```

## Generation

Generate audio for Baseline, Post-hoc, and Ours:

```bash
python generate_system_wavs.py \
  --input_manifest_csv data/manifest.csv \
  --output_manifest_csv results/generated_manifest.csv \
  --output_dir results/generated_wavs \
  --baseline_ckpt checkpoints/diffwave.pt \
  --mask_ckpt checkpoints/diffwave_mask_epoch_2.pth \
  --wm_ckpt checkpoints/diffwave-watermark_epoch_61.pth \
  --target_bits 1011001010110010 \
  --seed 1234
```

## Evaluation

Evaluate the main experiment:

```bash
python evaluate_main_experiment.py \
  --manifest_csv results/generated_manifest.csv \
  --ours_extractor_ckpt checkpoints/diffwave-watermark_extractor_epoch_61.pth \
  --target_bits 1011001010110010
```

Evaluate watermark robustness:

```bash
python evaluate_wm_attack_robustness.py \
  --manifest_csv results/generated_manifest.csv \
  --target_bits 1011001010110010
```

## Notes

Random seeds used in the release scripts include `1234`, `2345`, and `3456`. Large artifacts are excluded by `.gitignore`; keep checkpoints and generated results outside Git or distribute them separately.

## Citation

If this code is useful for your research, please cite the corresponding paper.
