#!/usr/bin/env python3
"""Main experiment evaluation for EMG->SoftUnits->Mel->Vocoder systems.

This script evaluates three vocoder systems conditioned on predicted Mel spectrograms:
- Baseline: standard DiffWave without watermark
- Post-hoc: standard DiffWave + traditional post-processing watermark
- Ours: watermarked DiffWave (embedded during reverse diffusion)

Metrics:
- Audio quality: WER (Whisper + jiwer), STOI, PESQ, MCD, SSIM (Log-Mel domain)
- Watermark: Bit Accuracy (ACC), Bit Error Rate (BER), Message Exact Match Rate (EMR)

Input pairing is controlled by a manifest CSV.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore

try:
    from jiwer import wer as jiwer_wer
except Exception:
    jiwer_wer = None  # type: ignore

try:
    from pesq import pesq as pesq_fn
except Exception:
    pesq_fn = None  # type: ignore

try:
    from pystoi import stoi as stoi_fn
except Exception:
    stoi_fn = None  # type: ignore

try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None  # type: ignore


REQUIRED_COLUMNS = [
    "id",
    "text",
    "gt_wav",
    "gt_mel",
    "baseline_wav",
    "posthoc_wav",
    "ours_wav",
    "posthoc_result",
]

SYSTEMS = ["Baseline", "Post-hoc", "Ours"]
TABLE_COLUMNS = ["WER", "STOI", "PESQ", "MCD", "SSIM", "ACC", "BER", "EMR"]


@dataclass
class SampleRecord:
    sample_id: str
    text: str
    gt_wav: str
    gt_mel: str
    baseline_wav: str
    posthoc_wav: str
    ours_wav: str
    posthoc_result: str


def ensure_runtime_dependencies() -> None:
    missing = []
    if pd is None:
        missing.append("pandas")
    if jiwer_wer is None:
        missing.append("jiwer")
    if stoi_fn is None:
        missing.append("pystoi")
    if ssim_fn is None:
        missing.append("scikit-image")

    if missing:
        raise RuntimeError(
            "Missing required packages: "
            + ", ".join(missing)
            + ". Please install them before running evaluation."
        )
    if pesq_fn is None:
        warnings.warn(
            "Optional package 'pesq' is not installed. PESQ values will be reported as NaN.",
            RuntimeWarning,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main experiment evaluator (WER/STOI/PESQ/MCD/SSIM + watermark metrics).")
    parser.add_argument("--manifest_csv", type=str, required=True, help="CSV manifest with paired paths and text labels.")
    parser.add_argument("--output_dir", type=str, default="results/main_eval", help="Directory to save reports.")

    parser.add_argument("--whisper_model", type=str, default="medium", help="Whisper model size/name.")
    parser.add_argument(
        "--whisper_language",
        type=str,
        default="auto",
        help="Whisper language code. Use 'auto' for language detection.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device for Whisper and extractor (e.g., cuda, cpu).",
    )
    parser.add_argument("--eval_sr", type=int, default=16000, help="Resample rate for WER/STOI/PESQ evaluation.")

    parser.add_argument("--n_mels", type=int, default=80, help="Mel bins for MCD computation.")
    parser.add_argument("--n_fft", type=int, default=1024, help="FFT size for MCD mel extraction.")
    parser.add_argument("--hop_length", type=int, default=256, help="Hop size for MCD mel extraction.")
    parser.add_argument("--win_length", type=int, default=1024, help="Window size for MCD mel extraction.")
    parser.add_argument("--f_min", type=float, default=0.0, help="f_min for mel extraction.")
    parser.add_argument(
        "--f_max",
        type=float,
        default=None,
        help="f_max for mel extraction. None means sr/2.",
    )
    parser.add_argument(
        "--gt_mel_is_log",
        action="store_true",
        default=False,
        help="Set this when GT mel .npy is already log-mel.",
    )

    parser.add_argument("--ours_extractor_ckpt", type=str, required=True, help="Checkpoint for Ours WatermarkExtractor.")
    parser.add_argument(
        "--target_bits",
        type=str,
        default="1011001010110010",
        help="Global target bit string (all samples share this payload).",
    )
    parser.add_argument(
        "--extractor_sr",
        type=int,
        default=22050,
        help="Resample rate expected by WatermarkExtractor.",
    )
    parser.add_argument(
        "--posthoc_bits_key",
        type=str,
        default="pred_bits",
        help="Preferred key name for bit string in Post-hoc result files.",
    )
    parser.add_argument(
        "--posthoc_spread_seed",
        type=int,
        default=1234,
        help="Seed for spread-spectrum post-hoc decode from wav.",
    )
    parser.add_argument(
        "--wm_decode_mode",
        type=str,
        default="windowed",
        choices=["single", "windowed"],
        help="Decode strategy for Ours/Baseline extractor path.",
    )
    parser.add_argument(
        "--wm_window_sec",
        type=float,
        default=1.0,
        help="Window size (sec) for windowed watermark decode.",
    )
    parser.add_argument(
        "--wm_window_hop_sec",
        type=float,
        default=0.5,
        help="Hop size (sec) for windowed watermark decode.",
    )
    parser.add_argument(
        "--wm_calibration",
        type=str,
        default="crossfit",
        choices=["none", "crossfit"],
        help="Per-bit threshold calibration mode for Ours.",
    )
    parser.add_argument(
        "--wm_crossfit_folds",
        type=int,
        default=5,
        help="Fold count for cross-fit watermark calibration.",
    )
    parser.add_argument(
        "--wm_enable_bit_flip",
        dest="wm_enable_bit_flip",
        action="store_true",
        help="Allow per-bit inversion in watermark calibration.",
    )
    parser.add_argument(
        "--wm_disable_bit_flip",
        dest="wm_enable_bit_flip",
        action="store_false",
        help="Disable per-bit inversion in watermark calibration.",
    )
    parser.set_defaults(wm_enable_bit_flip=True)

    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for quick smoke run.")
    parser.add_argument(
        "--systems",
        type=str,
        default="Baseline,Post-hoc,Ours",
        help="Comma-separated systems to evaluate: Baseline, Post-hoc, Ours.",
    )

    for c in REQUIRED_COLUMNS:
        parser.add_argument(f"--col_{c}", type=str, default=c, help=f"Manifest column name for {c}.")

    return parser.parse_args()


def parse_systems(systems_text: str) -> List[str]:
    aliases = {
        "baseline": "Baseline",
        "posthoc": "Post-hoc",
        "post-hoc": "Post-hoc",
        "ours": "Ours",
    }
    systems: List[str] = []
    for item in str(systems_text).split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"Unknown system in --systems: {item}")
        system = aliases[key]
        if system not in systems:
            systems.append(system)
    if not systems:
        raise ValueError("--systems must include at least one system.")
    return systems


def normalize_bits(bit_string: str, target_len: int) -> str:
    bits = "".join(ch for ch in str(bit_string) if ch in "01")
    if len(bits) < target_len:
        bits = bits.ljust(target_len, "0")
    return bits[:target_len]


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"_", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_float(x: float) -> float:
    try:
        x = float(x)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return float("nan")


def normalize_state_dict(sd: dict) -> dict:
    if isinstance(sd, dict) and ("state_dict" in sd) and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if isinstance(sd, dict) and ("model" in sd) and isinstance(sd["model"], dict):
        sd = sd["model"]
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def infer_num_bits_from_ckpt(sd: dict) -> Optional[int]:
    for k in ["fc2.weight", "module.fc2.weight"]:
        if k in sd and hasattr(sd[k], "shape") and len(sd[k].shape) == 2:
            return int(sd[k].shape[0])
    for k in ["fc2.bias", "module.fc2.bias"]:
        if k in sd and hasattr(sd[k], "shape") and len(sd[k].shape) == 1:
            return int(sd[k].shape[0])
    return None


def maybe_fix_mel_shape(mel: np.ndarray, n_mels: int) -> np.ndarray:
    mel = np.asarray(mel)
    if mel.ndim == 3 and mel.shape[0] == 1:
        mel = mel[0]

    if mel.ndim != 2:
        raise ValueError(f"GT mel must be 2D or [1,*,*], but got shape={mel.shape}")

    if mel.shape[0] == n_mels:
        return mel
    if mel.shape[1] == n_mels:
        return mel.T

    if mel.shape[0] < mel.shape[1]:
        return mel
    return mel.T


def load_audio_mono(path: str) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    wav = wav.float().clamp(-1.0, 1.0)
    return wav, int(sr)


def resample_audio(wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return wav
    return torchaudio.functional.resample(wav, src_sr, dst_sr)


def align_pair(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    min_len = min(a.shape[-1], b.shape[-1])
    if min_len <= 0:
        return a[..., :0], b[..., :0]
    return a[..., :min_len], b[..., :min_len]


def compute_stoi(gt_eval: torch.Tensor, pred_eval: torch.Tensor, sr: int) -> float:
    if stoi_fn is None:
        return float("nan")
    gt_eval, pred_eval = align_pair(gt_eval, pred_eval)
    if gt_eval.shape[-1] < int(0.3 * sr):
        return float("nan")
    gt_np = gt_eval.squeeze(0).cpu().numpy()
    pred_np = pred_eval.squeeze(0).cpu().numpy()
    try:
        return safe_float(stoi_fn(gt_np, pred_np, sr, extended=False))
    except Exception:
        return float("nan")


def compute_pesq(gt_eval: torch.Tensor, pred_eval: torch.Tensor, sr: int) -> float:
    if pesq_fn is None:
        return float("nan")
    gt_eval, pred_eval = align_pair(gt_eval, pred_eval)
    if gt_eval.shape[-1] < int(0.25 * sr):
        return float("nan")
    gt_np = gt_eval.squeeze(0).cpu().numpy()
    pred_np = pred_eval.squeeze(0).cpu().numpy()
    try:
        mode = "wb" if sr == 16000 else "nb"
        return safe_float(pesq_fn(sr, gt_np, pred_np, mode))
    except Exception:
        return float("nan")


def compute_generated_logmel(
    wav: torch.Tensor,
    sr: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    f_min: float,
    f_max: Optional[float],
) -> np.ndarray:
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        power=2.0,
        center=True,
    )
    mel = mel_transform(wav).squeeze(0)
    mel = torch.clamp(mel, min=1e-10)
    logmel = torch.log(mel)
    return logmel.cpu().numpy()


def prepare_gt_logmel(path: str, n_mels: int, gt_mel_is_log: bool) -> np.ndarray:
    gt_mel = np.load(path)
    gt_mel = maybe_fix_mel_shape(gt_mel, n_mels=n_mels)
    gt_mel = np.asarray(gt_mel, dtype=np.float32)
    if not gt_mel_is_log:
        gt_mel = np.log(np.clip(gt_mel, a_min=1e-10, a_max=None))
    return gt_mel


def compute_mcd(gen_logmel: np.ndarray, gt_logmel: np.ndarray) -> float:
    if gen_logmel.ndim != 2 or gt_logmel.ndim != 2:
        return float("nan")
    if gen_logmel.shape[0] != gt_logmel.shape[0]:
        return float("nan")

    t = min(gen_logmel.shape[1], gt_logmel.shape[1])
    if t <= 0:
        return float("nan")

    diff = gen_logmel[:, :t] - gt_logmel[:, :t]
    per_frame_dist = np.sqrt(np.sum(diff * diff, axis=0))
    return safe_float(np.mean(per_frame_dist))


def compute_logmel_ssim(gen_logmel: np.ndarray, gt_logmel: np.ndarray) -> float:
    if ssim_fn is None:
        return float("nan")
    if gen_logmel.ndim != 2 or gt_logmel.ndim != 2:
        return float("nan")
    if gen_logmel.shape[0] != gt_logmel.shape[0]:
        return float("nan")

    t = min(gen_logmel.shape[1], gt_logmel.shape[1])
    if t <= 1:
        return float("nan")

    a = np.asarray(gen_logmel[:, :t], dtype=np.float32)
    b = np.asarray(gt_logmel[:, :t], dtype=np.float32)
    try:
        low = float(min(float(np.min(a)), float(np.min(b))))
        high = float(max(float(np.max(a)), float(np.max(b))))
        data_range = max(high - low, 1e-6)
        return safe_float(ssim_fn(a, b, data_range=data_range))
    except Exception:
        return float("nan")


def compute_bit_metrics(pred_bits: Optional[str], target_bits: str) -> Tuple[float, float, float]:
    if pred_bits is None:
        return float("nan"), float("nan"), float("nan")

    n = len(target_bits)
    pred = normalize_bits(pred_bits, n)
    matched = sum(1 for a, b in zip(pred, target_bits) if a == b)
    acc = matched / n
    ber = 1.0 - acc
    emr = 1.0 if pred == target_bits else 0.0
    return safe_float(acc), safe_float(ber), safe_float(emr)


def load_whisper_model(model_name: str, device: str):
    try:
        import whisper  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import whisper. Install with: pip install -U openai-whisper"
        ) from e

    return whisper.load_model(model_name, device=device)


def transcribe_with_whisper(model, audio_16k: torch.Tensor, language: str, fp16: bool) -> str:
    audio_np = audio_16k.squeeze(0).cpu().numpy().astype(np.float32)
    kwargs = {"fp16": fp16, "task": "transcribe"}
    if language != "auto":
        kwargs["language"] = language
    result = model.transcribe(audio_np, **kwargs)
    text = result.get("text", "") if isinstance(result, dict) else ""
    return normalize_text(text)


def _to_int16_numpy(wav: torch.Tensor) -> np.ndarray:
    x = wav.squeeze(0).detach().cpu().numpy()
    x = np.clip(x, -1.0, 1.0)
    return np.round(x * 32767.0).astype(np.int16)


def _decode_posthoc_lsb_from_wav(wav: torch.Tensor, bit_len: int) -> str:
    x_i16 = _to_int16_numpy(wav)
    lsb = (x_i16.astype(np.int32) & 1).astype(np.int32)
    bits = []
    for i in range(bit_len):
        vals = lsb[i::bit_len]
        if len(vals) == 0:
            bits.append("0")
        else:
            bits.append("1" if int(np.round(np.mean(vals))) >= 1 else "0")
    return "".join(bits)


def _pn_sequence(length: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=(length,)).astype(np.float32)


def _decode_posthoc_spread_from_wav(wav: torch.Tensor, bit_len: int, seed: int) -> str:
    x = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
    t = len(x)
    out = []
    for i in range(bit_len):
        pn = _pn_sequence(t, seed + 9973 * i)
        corr = float(np.dot(x, pn) / max(1, t))
        out.append("1" if corr >= 0.0 else "0")
    return "".join(out)


def parse_posthoc_method(path: str) -> Optional[str]:
    if not path or (not os.path.exists(path)):
        return None
    try:
        suffix = Path(path).suffix.lower()
        if suffix in {".json", ".jsonl"}:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if suffix == ".jsonl":
                for line in raw.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        m = str(obj.get("method", "")).strip().lower()
                        if m:
                            return m
            else:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    m = str(obj.get("method", "")).strip().lower()
                    if m:
                        return m
    except Exception:
        return None
    return None


def decode_posthoc_bits_from_wav(
    posthoc_wav: str,
    method: str,
    target_len: int,
    spread_seed: int,
) -> Optional[str]:
    if not posthoc_wav or (not os.path.exists(posthoc_wav)):
        return None
    try:
        wav, _ = load_audio_mono(posthoc_wav)
        method = str(method).strip().lower()
        if method == "lsb":
            return normalize_bits(_decode_posthoc_lsb_from_wav(wav, target_len), target_len)
        if method == "spread":
            return normalize_bits(_decode_posthoc_spread_from_wav(wav, target_len, seed=int(spread_seed)), target_len)
    except Exception:
        return None
    return None


def parse_bits_from_posthoc_file(path: str, preferred_key: str, target_len: int) -> Optional[str]:
    if not path or (not os.path.exists(path)):
        return None

    def extract_from_mapping(obj: object, key_first: str) -> Optional[str]:
        keys = [
            key_first,
            "pred_bits",
            "extracted_bits",
            "bits",
            "message_bits",
            "wm_bits",
            "watermark_bits",
            "pred_code16",
            "code16",
        ]

        if isinstance(obj, dict):
            for k in keys:
                if k in obj:
                    val = obj[k]
                    if isinstance(val, (str, int, float)):
                        bits = "".join(ch for ch in str(val) if ch in "01")
                        if bits:
                            return normalize_bits(bits, target_len)
            for v in obj.values():
                out = extract_from_mapping(v, key_first)
                if out is not None:
                    return out
        elif isinstance(obj, list):
            for item in obj:
                out = extract_from_mapping(item, key_first)
                if out is not None:
                    return out
        return None

    suffix = Path(path).suffix.lower()

    if suffix in {".json", ".jsonl"}:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if suffix == ".jsonl":
                for line in raw.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    out = extract_from_mapping(obj, preferred_key)
                    if out is not None:
                        return out
            else:
                obj = json.loads(raw)
                out = extract_from_mapping(obj, preferred_key)
                if out is not None:
                    return out
        except Exception:
            pass

    if suffix in {".csv", ".tsv"}:
        try:
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
            if not df.empty:
                row = df.iloc[0].to_dict()
                out = extract_from_mapping(row, preferred_key)
                if out is not None:
                    return out
        except Exception:
            pass

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        line_key_pattern = re.compile(rf"{re.escape(preferred_key)}\s*[:=]\s*([01]{{4,}})", re.IGNORECASE)
        m = line_key_pattern.search(text)
        if m:
            return normalize_bits(m.group(1), target_len)

        generic_keys = [
            "pred_bits",
            "extracted_bits",
            "bits",
            "message_bits",
            "wm_bits",
            "watermark_bits",
            "pred_code16",
            "code16",
        ]
        for key in generic_keys:
            pattern = re.compile(rf"{re.escape(key)}\s*[:=]\s*([01]{{4,}})", re.IGNORECASE)
            m = pattern.search(text)
            if m:
                return normalize_bits(m.group(1), target_len)

        candidates = re.findall(r"[01]{4,}", text)
        if candidates:
            candidates = sorted(candidates, key=len, reverse=True)
            return normalize_bits(candidates[0], target_len)
    except Exception:
        return None

    return None


def load_manifest_and_validate(args: argparse.Namespace) -> List[SampleRecord]:
    manifest_path = args.manifest_csv
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest_csv not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    colmap = {k: getattr(args, f"col_{k}") for k in REQUIRED_COLUMNS}
    missing = [colmap[k] for k in REQUIRED_COLUMNS if colmap[k] not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    records: List[SampleRecord] = []
    dropped = 0

    for _, row in df.iterrows():
        sid = str(row[colmap["id"]]).strip()
        text = str(row[colmap["text"]]).strip()

        gt_wav = str(row[colmap["gt_wav"]]).strip()
        gt_mel = str(row[colmap["gt_mel"]]).strip()
        baseline_wav = str(row[colmap["baseline_wav"]]).strip()
        posthoc_wav = str(row[colmap["posthoc_wav"]]).strip()
        ours_wav = str(row[colmap["ours_wav"]]).strip()
        posthoc_result = str(row[colmap["posthoc_result"]]).strip()

        systems_to_eval = getattr(args, "_systems_to_eval", SYSTEMS)
        required_paths = [gt_wav, gt_mel]
        if "Baseline" in systems_to_eval:
            required_paths.append(baseline_wav)
        if "Post-hoc" in systems_to_eval:
            required_paths.append(posthoc_wav)
        if "Ours" in systems_to_eval:
            required_paths.append(ours_wav)
        if any((not p) or (not os.path.exists(p)) for p in required_paths):
            dropped += 1
            continue

        records.append(
            SampleRecord(
                sample_id=sid,
                text=text,
                gt_wav=gt_wav,
                gt_mel=gt_mel,
                baseline_wav=baseline_wav,
                posthoc_wav=posthoc_wav,
                ours_wav=ours_wav,
                posthoc_result=posthoc_result,
            )
        )

    if dropped > 0:
        warnings.warn(f"Dropped {dropped} manifest rows due to missing required files.")

    if args.max_samples is not None:
        records = records[: max(0, int(args.max_samples))]

    if not records:
        raise RuntimeError("No valid records after manifest validation.")

    return records


def load_ours_extractor(ckpt_path: str, device: str, target_bits_len: int):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ours_extractor_ckpt not found: {ckpt_path}")

    from decoder import WatermarkExtractor

    ckpt_raw = torch.load(ckpt_path, map_location=device)
    state = normalize_state_dict(ckpt_raw)
    num_bits = infer_num_bits_from_ckpt(state) or target_bits_len

    if num_bits != target_bits_len:
        warnings.warn(
            f"Extractor num_bits={num_bits} differs from target_bits_len={target_bits_len}. "
            f"Metrics will compare on length={target_bits_len} with trunc/pad normalization."
        )

    model = WatermarkExtractor(num_bits=num_bits, device=device).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _normalize_prob_vector(probs: np.ndarray, target_len: int) -> np.ndarray:
    out = np.full((target_len,), 0.5, dtype=np.float32)
    if probs is None:
        return out
    arr = np.asarray(probs, dtype=np.float32).reshape(-1)
    if arr.size <= 0:
        return out
    n = min(target_len, int(arr.size))
    out[:n] = arr[:n]
    return out


def _window_start_indices(total_len: int, win_len: int, hop_len: int) -> List[int]:
    if total_len <= win_len:
        return [0]
    starts = list(range(0, total_len - win_len + 1, hop_len))
    tail = total_len - win_len
    if starts[-1] != tail:
        starts.append(tail)
    return starts


def extract_probs_with_ours_extractor(
    extractor,
    wav: torch.Tensor,
    sr: int,
    extractor_sr: int,
    target_len: int,
    device: str,
    decode_mode: str,
    window_sec: float,
    window_hop_sec: float,
) -> Optional[np.ndarray]:
    try:
        wav_in = resample_audio(wav, sr, extractor_sr)
        wav_in = wav_in.to(device)
        total_len = int(wav_in.shape[-1])
        if total_len <= 0:
            return None
        with torch.no_grad():
            mode = str(decode_mode).strip().lower()
            if mode == "single":
                probs = extractor(wav_in)[0].detach().float().cpu().numpy()
                return _normalize_prob_vector(probs, target_len)

            win_len = max(1, int(round(float(window_sec) * float(extractor_sr))))
            hop_len = max(1, int(round(float(window_hop_sec) * float(extractor_sr))))
            win_len = min(win_len, total_len)

            starts = _window_start_indices(total_len=total_len, win_len=win_len, hop_len=hop_len)
            prob_list: List[np.ndarray] = []
            for st in starts:
                chunk = wav_in[..., st : st + win_len]
                if int(chunk.shape[-1]) < win_len:
                    pad_right = win_len - int(chunk.shape[-1])
                    chunk = torch.nn.functional.pad(chunk, (0, pad_right))
                probs = extractor(chunk)[0].detach().float().cpu().numpy()
                prob_list.append(_normalize_prob_vector(probs, target_len))
            if not prob_list:
                return None
            stacked = np.stack(prob_list, axis=0)
            return np.mean(stacked, axis=0).astype(np.float32)
    except Exception:
        return None


def bits_from_prob_vector(
    prob_vec: Optional[np.ndarray],
    thresholds: Optional[np.ndarray] = None,
    bit_flip: Optional[np.ndarray] = None,
) -> Optional[str]:
    if prob_vec is None:
        return None
    p = np.asarray(prob_vec, dtype=np.float32).reshape(-1)
    if p.size <= 0:
        return None

    if thresholds is None:
        thr = np.full_like(p, 0.5, dtype=np.float32)
    elif np.isscalar(thresholds):
        thr = np.full_like(p, float(thresholds), dtype=np.float32)
    else:
        thr = _normalize_prob_vector(np.asarray(thresholds, dtype=np.float32), int(p.size))

    bits = (p >= thr).astype(np.int32)

    if bit_flip is not None:
        flip_raw = np.asarray(bit_flip, dtype=np.int32).reshape(-1)
        flip = np.zeros((bits.size,), dtype=np.int32)
        n = min(bits.size, int(flip_raw.size))
        flip[:n] = flip_raw[:n]
        bits = np.where(flip > 0, 1 - bits, bits)

    return "".join("1" if int(v) == 1 else "0" for v in bits.tolist())


def _target_bits_array(target_bits: str) -> np.ndarray:
    return np.asarray([1 if ch == "1" else 0 for ch in target_bits], dtype=np.int32)


def _bit_accuracy_vector(pred_matrix: np.ndarray, target_vec: np.ndarray) -> np.ndarray:
    if pred_matrix.size <= 0:
        return np.zeros((target_vec.size,), dtype=np.float32)
    return np.mean(pred_matrix == target_vec[None, :], axis=0).astype(np.float32)


def _bits_to_matrix(bits_list: List[Optional[str]], n_bits: int) -> np.ndarray:
    out = np.zeros((len(bits_list), n_bits), dtype=np.int32)
    for i, bits in enumerate(bits_list):
        if bits is None:
            continue
        normalized = normalize_bits(bits, n_bits)
        out[i, :] = np.asarray([1 if ch == "1" else 0 for ch in normalized], dtype=np.int32)
    return out


def fit_per_bit_calibration(
    train_probs: np.ndarray,
    target_bits: str,
    enable_bit_flip: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_bits = train_probs.shape[1]
    target_vec = _target_bits_array(target_bits)

    thresholds = np.full((n_bits,), 0.5, dtype=np.float32)
    flips = np.zeros((n_bits,), dtype=np.int32)
    bit_acc_pre = np.zeros((n_bits,), dtype=np.float32)
    bit_acc_post = np.zeros((n_bits,), dtype=np.float32)

    candidate_thresholds = np.linspace(0.05, 0.95, 91, dtype=np.float32)

    for bit_idx in range(n_bits):
        p = train_probs[:, bit_idx]
        y = np.full((train_probs.shape[0],), int(target_vec[bit_idx]), dtype=np.int32)

        pred_pre = (p >= 0.5).astype(np.int32)
        bit_acc_pre[bit_idx] = float(np.mean(pred_pre == y))

        best_acc = -1.0
        best_thr = 0.5
        best_flip = 0

        for thr in candidate_thresholds:
            pred = (p >= float(thr)).astype(np.int32)
            candidates = [(float(np.mean(pred == y)), 0)]
            if enable_bit_flip:
                candidates.append((float(np.mean((1 - pred) == y)), 1))

            for acc_val, flip_val in candidates:
                better = acc_val > (best_acc + 1e-12)
                if (not better) and abs(acc_val - best_acc) <= 1e-12:
                    cur_dist = abs(float(thr) - 0.5)
                    best_dist = abs(float(best_thr) - 0.5)
                    better = (cur_dist < (best_dist - 1e-12)) or (
                        abs(cur_dist - best_dist) <= 1e-12 and int(flip_val) < int(best_flip)
                    )
                if better:
                    best_acc = float(acc_val)
                    best_thr = float(thr)
                    best_flip = int(flip_val)

        thresholds[bit_idx] = float(best_thr)
        flips[bit_idx] = int(best_flip)
        bit_acc_post[bit_idx] = float(best_acc)

    return thresholds, flips, bit_acc_pre, bit_acc_post


def run_crossfit_calibration(
    prob_matrix: np.ndarray,
    target_bits: str,
    folds: int,
    enable_bit_flip: bool,
    seed: int = 1234,
) -> Tuple[List[str], List[str], Dict[str, object]]:
    n_samples, n_bits = prob_matrix.shape
    if n_samples <= 0:
        raise ValueError("prob_matrix is empty.")

    fold_count = max(2, min(int(folds), n_samples))
    shuffled = np.arange(n_samples, dtype=np.int32)
    rng = np.random.RandomState(seed)
    rng.shuffle(shuffled)
    fold_splits = np.array_split(shuffled, fold_count)

    calibrated_bits: List[Optional[str]] = [None] * n_samples
    fold_reports: List[Dict[str, object]] = []

    for fold_idx, val_idx in enumerate(fold_splits, start=1):
        if val_idx.size <= 0:
            continue
        train_idx = np.setdiff1d(shuffled, val_idx, assume_unique=False)
        if train_idx.size <= 0:
            train_idx = val_idx

        train_probs = prob_matrix[train_idx]
        val_probs = prob_matrix[val_idx]

        thresholds, flips, train_acc_pre, train_acc_post = fit_per_bit_calibration(
            train_probs=train_probs,
            target_bits=target_bits,
            enable_bit_flip=enable_bit_flip,
        )

        val_bits_raw = [bits_from_prob_vector(v, thresholds=0.5, bit_flip=None) for v in val_probs]
        val_bits_cal = [bits_from_prob_vector(v, thresholds=thresholds, bit_flip=flips) for v in val_probs]

        for local_i, global_i in enumerate(val_idx.tolist()):
            calibrated_bits[int(global_i)] = val_bits_cal[local_i]

        y_vec = _target_bits_array(target_bits)
        val_pre_acc = _bit_accuracy_vector(_bits_to_matrix(val_bits_raw, n_bits), y_vec)
        val_post_acc = _bit_accuracy_vector(_bits_to_matrix(val_bits_cal, n_bits), y_vec)

        fold_reports.append(
            {
                "fold": int(fold_idx),
                "num_train": int(train_idx.size),
                "num_val": int(val_idx.size),
                "thresholds": [float(x) for x in thresholds.tolist()],
                "bit_flip": [int(x) for x in flips.tolist()],
                "train_bit_acc_pre": [float(x) for x in train_acc_pre.tolist()],
                "train_bit_acc_post": [float(x) for x in train_acc_post.tolist()],
                "val_bit_acc_pre": [float(x) for x in val_pre_acc.tolist()],
                "val_bit_acc_post": [float(x) for x in val_post_acc.tolist()],
            }
        )

    for i in range(n_samples):
        if calibrated_bits[i] is None:
            calibrated_bits[i] = bits_from_prob_vector(prob_matrix[i], thresholds=0.5, bit_flip=None)

    calibrated_bits_fixed = [normalize_bits(x or "", n_bits) for x in calibrated_bits]
    raw_bits = [normalize_bits(bits_from_prob_vector(v, thresholds=0.5, bit_flip=None) or "", n_bits) for v in prob_matrix]

    y_vec = _target_bits_array(target_bits)
    bit_acc_raw = _bit_accuracy_vector(_bits_to_matrix(raw_bits, n_bits), y_vec)
    bit_acc_cal = _bit_accuracy_vector(_bits_to_matrix(calibrated_bits_fixed, n_bits), y_vec)

    diagnostics: Dict[str, object] = {
        "mode": "crossfit",
        "seed": int(seed),
        "folds": int(fold_count),
        "n_samples": int(n_samples),
        "n_bits": int(n_bits),
        "bit_acc_pre_global": [float(x) for x in bit_acc_raw.tolist()],
        "bit_acc_post_global": [float(x) for x in bit_acc_cal.tolist()],
        "fold_reports": fold_reports,
    }

    return calibrated_bits_fixed, raw_bits, diagnostics


def aggregate_and_render(df: pd.DataFrame, output_dir: str) -> Tuple[pd.DataFrame, str]:
    numeric_cols = [
        "wer",
        "stoi",
        "pesq",
        "mcd",
        "ssim",
        "wm_acc",
        "wm_ber",
        "wm_emr",
    ]

    summary = (
        df.groupby("system", as_index=True)[numeric_cols]
        .mean(numeric_only=True)
        .reindex([s for s in SYSTEMS if s in set(df["system"].astype(str).tolist())])
        .reset_index()
    )

    summary_path = os.path.join(output_dir, "system_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    def fmt(x: float, percent: bool = False, ndigits: int = 3) -> str:
        if x is None or (isinstance(x, float) and (not math.isfinite(x))):
            return "N/A"
        if percent:
            return f"{x * 100:.{ndigits}f}"
        return f"{x:.{ndigits}f}"

    markdown_lines = [
        "| System | WER(down) | STOI(up) | PESQ(up) | MCD(down) | SSIM(up) | ACC(up) | BER(down) | EMR(up) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary.iterrows():
        markdown_lines.append(
            "| {sys} | {wer} | {stoi} | {pesq} | {mcd} | {ssim} | {acc} | {ber} | {emr} |".format(
                sys=row["system"],
                wer=fmt(row["wer"], percent=True),
                stoi=fmt(row["stoi"]),
                pesq=fmt(row["pesq"]),
                mcd=fmt(row["mcd"]),
                ssim=fmt(row["ssim"]),
                acc=fmt(row["wm_acc"], percent=True),
                ber=fmt(row["wm_ber"], percent=True),
                emr=fmt(row["wm_emr"], percent=True),
            )
        )

    markdown_table = "\n".join(markdown_lines)
    markdown_path = os.path.join(output_dir, "final_markdown_table.md")
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_table + "\n")

    return summary, markdown_table

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    ensure_runtime_dependencies()
    systems_to_eval = parse_systems(args.systems)
    args._systems_to_eval = systems_to_eval

    target_bits = normalize_bits(args.target_bits, len("".join(ch for ch in args.target_bits if ch in "01")))
    if len(target_bits) == 0:
        raise ValueError("target_bits must include at least one binary character (0/1).")

    records = load_manifest_and_validate(args)

    whisper_model = load_whisper_model(args.whisper_model, args.device)
    use_fp16 = args.device.startswith("cuda") and torch.cuda.is_available()

    ours_extractor = load_ours_extractor(args.ours_extractor_ckpt, args.device, target_bits_len=len(target_bits))

    transcript_cache: Dict[str, str] = {}
    gt_mel_cache: Dict[str, np.ndarray] = {}

    rows: List[Dict[str, object]] = []
    ours_row_indices: List[int] = []
    ours_prob_vectors: List[Optional[np.ndarray]] = []

    for idx, rec in enumerate(records, start=1):
        gt_wav, gt_sr = load_audio_mono(rec.gt_wav)
        gt_eval = resample_audio(gt_wav, gt_sr, args.eval_sr)

        if rec.gt_mel not in gt_mel_cache:
            gt_mel_cache[rec.gt_mel] = prepare_gt_logmel(
                rec.gt_mel,
                n_mels=args.n_mels,
                gt_mel_is_log=bool(args.gt_mel_is_log),
            )
        gt_logmel = gt_mel_cache[rec.gt_mel]

        ref_text = normalize_text(rec.text)

        system_to_wav = {
            "Baseline": rec.baseline_wav,
            "Post-hoc": rec.posthoc_wav,
            "Ours": rec.ours_wav,
        }

        for system_name, gen_path in system_to_wav.items():
            if system_name not in systems_to_eval:
                continue
            gen_wav, gen_sr = load_audio_mono(gen_path)
            gen_eval = resample_audio(gen_wav, gen_sr, args.eval_sr)

            if gen_path not in transcript_cache:
                transcript_cache[gen_path] = transcribe_with_whisper(
                    whisper_model,
                    audio_16k=gen_eval,
                    language=args.whisper_language,
                    fp16=use_fp16,
                )
            pred_text = transcript_cache[gen_path]

            wer = float("nan")
            if ref_text:
                try:
                    wer = safe_float(jiwer_wer(ref_text, pred_text))
                except Exception:
                    wer = float("nan")

            stoi = compute_stoi(gt_eval, gen_eval, sr=args.eval_sr)
            pesq = compute_pesq(gt_eval, gen_eval, sr=args.eval_sr)

            try:
                gen_logmel = compute_generated_logmel(
                    gen_wav,
                    sr=gen_sr,
                    n_mels=args.n_mels,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    win_length=args.win_length,
                    f_min=args.f_min,
                    f_max=args.f_max,
                )
                mcd = compute_mcd(gen_logmel, gt_logmel)
                ssim = compute_logmel_ssim(gen_logmel, gt_logmel)
            except Exception:
                mcd = float("nan")
                ssim = float("nan")

            if system_name in {"Baseline", "Ours"}:
                prob_vec = extract_probs_with_ours_extractor(
                    ours_extractor,
                    wav=gen_wav,
                    sr=gen_sr,
                    extractor_sr=args.extractor_sr,
                    target_len=len(target_bits),
                    device=args.device,
                    decode_mode=args.wm_decode_mode,
                    window_sec=args.wm_window_sec,
                    window_hop_sec=args.wm_window_hop_sec,
                )
                pred_bits = bits_from_prob_vector(prob_vec, thresholds=0.5, bit_flip=None)
            else:
                prob_vec = None
                method = parse_posthoc_method(rec.posthoc_result)
                pred_bits = None
                if method:
                    pred_bits = decode_posthoc_bits_from_wav(
                        posthoc_wav=rec.posthoc_wav,
                        method=method,
                        target_len=len(target_bits),
                        spread_seed=args.posthoc_spread_seed,
                    )
                if pred_bits is None:
                    warnings.warn(
                        f"Post-hoc wav decode failed or method missing for id={rec.sample_id}; "
                        "fallback to posthoc_result bits.",
                        RuntimeWarning,
                    )
                    pred_bits = parse_bits_from_posthoc_file(
                        rec.posthoc_result,
                        preferred_key=args.posthoc_bits_key,
                        target_len=len(target_bits),
                    )

            wm_acc, wm_ber, wm_emr = compute_bit_metrics(pred_bits, target_bits)

            row_index = len(rows)
            if system_name == "Ours":
                ours_row_indices.append(row_index)
                ours_prob_vectors.append(prob_vec)

            rows.append(
                {
                    "id": rec.sample_id,
                    "system": system_name,
                    "wav_path": gen_path,
                    "ref_text": ref_text,
                    "pred_text": pred_text,
                    "wer": wer,
                    "stoi": stoi,
                    "pesq": pesq,
                    "mcd": mcd,
                    "ssim": ssim,
                    "wm_pred_bits_raw": pred_bits if system_name == "Ours" else None,
                    "wm_pred_bits": pred_bits,
                    "wm_acc": wm_acc,
                    "wm_ber": wm_ber,
                    "wm_emr": wm_emr,
                }
            )

        if idx % 10 == 0 or idx == len(records):
            print(f"[Progress] {idx}/{len(records)} samples processed")

    if str(args.wm_calibration).strip().lower() == "crossfit":
        valid_pairs: List[Tuple[int, np.ndarray]] = []
        for row_idx, prob_vec in zip(ours_row_indices, ours_prob_vectors):
            if prob_vec is None:
                continue
            valid_pairs.append((row_idx, _normalize_prob_vector(prob_vec, len(target_bits))))

        if len(valid_pairs) >= 2:
            prob_matrix = np.stack([v for _, v in valid_pairs], axis=0).astype(np.float32)
            calibrated_bits, raw_bits, diagnostics = run_crossfit_calibration(
                prob_matrix=prob_matrix,
                target_bits=target_bits,
                folds=int(args.wm_crossfit_folds),
                enable_bit_flip=bool(args.wm_enable_bit_flip),
                seed=1234,
            )
            for i, (row_idx, _) in enumerate(valid_pairs):
                rows[row_idx]["wm_pred_bits_raw"] = raw_bits[i]
                rows[row_idx]["wm_pred_bits"] = calibrated_bits[i]
                wm_acc, wm_ber, wm_emr = compute_bit_metrics(calibrated_bits[i], target_bits)
                rows[row_idx]["wm_acc"] = wm_acc
                rows[row_idx]["wm_ber"] = wm_ber
                rows[row_idx]["wm_emr"] = wm_emr

            raw_acc_list = [compute_bit_metrics(raw_bits[i], target_bits)[0] for i in range(len(raw_bits))]
            cal_acc_list = [compute_bit_metrics(calibrated_bits[i], target_bits)[0] for i in range(len(calibrated_bits))]
            diagnostics["ours_acc_mean_pre"] = safe_float(np.nanmean(np.asarray(raw_acc_list, dtype=np.float32)))
            diagnostics["ours_acc_mean_post"] = safe_float(np.nanmean(np.asarray(cal_acc_list, dtype=np.float32)))
            diagnostics["decode_mode"] = str(args.wm_decode_mode)
            diagnostics["window_sec"] = float(args.wm_window_sec)
            diagnostics["window_hop_sec"] = float(args.wm_window_hop_sec)
            diagnostics["bit_flip_enabled"] = bool(args.wm_enable_bit_flip)

            diag_path = os.path.join(args.output_dir, "wm_calibration_diagnostics.json")
            with open(diag_path, "w", encoding="utf-8") as f:
                json.dump(diagnostics, f, ensure_ascii=False, indent=2)
            print(f"[Saved] Watermark calibration diagnostics: {diag_path}")
        else:
            warnings.warn(
                "wm_calibration=crossfit requested, but fewer than 2 valid Ours probability vectors were found. "
                "Skip calibration.",
                RuntimeWarning,
            )

    utter_df = pd.DataFrame(rows)
    utter_path = os.path.join(args.output_dir, "utterance_metrics.csv")
    utter_df.to_csv(utter_path, index=False, encoding="utf-8")

    _, markdown_table = aggregate_and_render(utter_df, args.output_dir)

    print("\n=== Main Experiment Summary (Markdown) ===")
    print(markdown_table)
    print(f"\n[Saved] Utterance metrics: {utter_path}")
    print(f"[Saved] System summary: {os.path.join(args.output_dir, 'system_summary.csv')}")
    print(f"[Saved] Markdown table: {os.path.join(args.output_dir, 'final_markdown_table.md')}")


if __name__ == "__main__":
    main()
