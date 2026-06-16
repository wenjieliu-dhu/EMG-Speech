#!/usr/bin/env python3
"""Evaluate WavMark clean-condition audio quality for Table I."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

from evaluate_main_experiment import (
    compute_bit_metrics,
    compute_generated_logmel,
    compute_logmel_ssim,
    compute_mcd,
    compute_pesq,
    compute_stoi,
    load_audio_mono,
    load_whisper_model,
    normalize_bits,
    normalize_text,
    prepare_gt_logmel,
    resample_audio,
    safe_float,
    transcribe_with_whisper,
)
from strong_watermark_baselines import (
    add_extra_site_packages,
    add_source_path,
    find_default_wavmark_ckpt,
    install_resampy_import_shim,
)

try:
    from jiwer import wer as jiwer_wer
except Exception:
    jiwer_wer = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WavMark clean quality and payload evaluation.")
    parser.add_argument("--manifest_csv", type=str, default="results/eval_manifest.csv")
    parser.add_argument("--output_dir", type=str, default="results/wavmark_clean_quality_seed1234")
    parser.add_argument("--wavmark_src", type=str, default=r"E:\postgraduate\project\EMG-Speech\wavmark-main")
    parser.add_argument("--wavmark_ckpt", type=str, default="")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--target_bits", type=str, default="1011001010110010")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--eval_sr", type=int, default=16000)
    parser.add_argument("--whisper_model", type=str, default="base")
    parser.add_argument("--whisper_language", type=str, default="en")
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--win_length", type=int, default=1024)
    parser.add_argument("--f_min", type=float, default=0.0)
    parser.add_argument("--f_max", type=float, default=8000.0)
    parser.add_argument("--gt_mel_is_log", action="store_true", default=False)
    parser.add_argument("--wavmark_min_snr", type=float, default=20.0)
    parser.add_argument("--wavmark_max_snr", type=float, default=38.0)
    parser.add_argument("--reuse_saved", action="store_true", default=False)
    parser.add_argument(
        "--extra_site_packages",
        type=str,
        default=";".join(
            [
                r"E:\postgraduate\anaconda3\envs\segmark\Lib\site-packages",
                r"E:\postgraduate\anaconda3\envs\ste-gan\Lib\site-packages",
                r"E:\postgraduate\anaconda3\envs\LaWa\Lib\site-packages",
            ]
        ),
        help="Optional read-only site-packages paths used for missing WavMark dependencies.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, percent: bool = False) -> str:
    if value != value:
        return "--"
    if percent:
        value *= 100.0
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    wav_dir = out_dir / "generated_wavs" / "wavmark"
    wav_dir.mkdir(parents=True, exist_ok=True)

    add_extra_site_packages(args.extra_site_packages)
    wavmark_src = Path(args.wavmark_src)
    add_source_path(str(wavmark_src / "src" if (wavmark_src / "src").exists() else wavmark_src))
    install_resampy_import_shim()
    import wavmark  # type: ignore

    ckpt = args.wavmark_ckpt or find_default_wavmark_ckpt()
    model = wavmark.load_model(ckpt) if ckpt else wavmark.load_model()
    model = model.to(args.device)
    model.eval()

    df = pd.read_csv(args.manifest_csv)
    if args.max_samples is not None:
        df = df.iloc[: max(0, int(args.max_samples))].copy()

    target_bits = normalize_bits(args.target_bits, len("".join(ch for ch in args.target_bits if ch in "01")))
    payload = [int(ch) for ch in target_bits]

    whisper_model = load_whisper_model(args.whisper_model, args.device)
    use_fp16 = args.device.startswith("cuda") and torch.cuda.is_available()

    rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []
    gt_mel_cache: Dict[str, np.ndarray] = {}

    for idx, rec in df.iterrows():
        sample_id = str(rec["id"])
        wavmark_path = wav_dir / f"{sample_id}_wavmark.wav"

        if args.reuse_saved and wavmark_path.exists():
            watermarked, wm_sr = sf.read(str(wavmark_path), dtype="float32")
            if watermarked.ndim > 1:
                watermarked = np.mean(watermarked, axis=1).astype(np.float32)
            if wm_sr != 16000:
                wav_tensor = torch.from_numpy(watermarked).view(1, -1)
                watermarked = torchaudio.functional.resample(wav_tensor, wm_sr, 16000).squeeze(0).numpy()
        else:
            base_wav, base_sr = load_audio_mono(str(rec["baseline_wav"]))
            if base_sr != 16000:
                base_wav = torchaudio.functional.resample(base_wav, base_sr, 16000)
            signal = base_wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
            signal = np.clip(signal, -1.0, 1.0)
            watermarked, _info = wavmark.encode_watermark(
                model,
                signal,
                payload,
                min_snr=float(args.wavmark_min_snr),
                max_snr=float(args.wavmark_max_snr),
                show_progress=False,
            )
            watermarked = np.asarray(watermarked, dtype=np.float32).reshape(-1)
            sf.write(str(wavmark_path), watermarked, 16000)

        start = time.perf_counter()
        decoded, dec_info = wavmark.decode_watermark(model, watermarked, show_progress=False)
        decode_time = time.perf_counter() - start
        pred_bits = "" if decoded is None else "".join("1" if int(v) else "0" for v in np.asarray(decoded).reshape(-1).tolist())
        wm_acc, wm_ber, wm_emr = compute_bit_metrics(pred_bits, target_bits)

        gt_wav, gt_sr = load_audio_mono(str(rec["gt_wav"]))
        gt_eval = resample_audio(gt_wav, gt_sr, args.eval_sr)
        gen_wav = torch.from_numpy(watermarked).view(1, -1)
        gen_eval = resample_audio(gen_wav, 16000, args.eval_sr)

        ref_text = normalize_text(str(rec["text"]))
        pred_text = transcribe_with_whisper(
            whisper_model,
            audio_16k=gen_eval,
            language=args.whisper_language,
            fp16=use_fp16,
        )
        wer = float("nan")
        if ref_text and jiwer_wer is not None:
            try:
                wer = safe_float(jiwer_wer(ref_text, pred_text))
            except Exception:
                wer = float("nan")

        stoi = compute_stoi(gt_eval, gen_eval, sr=args.eval_sr)
        pesq = compute_pesq(gt_eval, gen_eval, sr=args.eval_sr)
        gt_mel_path = str(rec["gt_mel"])
        if gt_mel_path not in gt_mel_cache:
            gt_mel_cache[gt_mel_path] = prepare_gt_logmel(
                gt_mel_path,
                n_mels=args.n_mels,
                gt_mel_is_log=bool(args.gt_mel_is_log),
            )
        try:
            gen_logmel = compute_generated_logmel(
                gen_wav,
                sr=16000,
                n_mels=args.n_mels,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                win_length=args.win_length,
                f_min=args.f_min,
                f_max=args.f_max,
            )
            mcd = compute_mcd(gen_logmel, gt_mel_cache[gt_mel_path])
            ssim = compute_logmel_ssim(gen_logmel, gt_mel_cache[gt_mel_path])
        except Exception:
            mcd = float("nan")
            ssim = float("nan")

        rows.append(
            {
                "id": sample_id,
                "system": "WavMark",
                "wav_path": str(wavmark_path),
                "ref_text": ref_text,
                "pred_text": pred_text,
                "wer": wer,
                "stoi": stoi,
                "pesq": pesq,
                "mcd": mcd,
                "ssim": ssim,
                "wm_pred_bits": pred_bits,
                "wm_acc": wm_acc,
                "wm_ber": wm_ber,
                "wm_emr": wm_emr,
                "num_decode_hits": len(dec_info.get("results", [])) if isinstance(dec_info, dict) else None,
                "decode_time_sec": decode_time,
            }
        )
        manifest_rows.append(
            {
                "id": sample_id,
                "wavmark_wav": str(wavmark_path),
                "gt_wav": str(rec["gt_wav"]),
                "gt_mel": gt_mel_path,
                "text": str(rec["text"]),
            }
        )
        if len(rows) % 10 == 0 or len(rows) == len(df):
            print(f"[Progress] WavMark quality {len(rows)}/{len(df)}")

    write_csv(out_dir / "wavmark_quality_utterance_metrics.csv", rows)
    write_csv(out_dir / "wavmark_generated_manifest.csv", manifest_rows)

    summary = pd.DataFrame(rows)[["wer", "stoi", "pesq", "mcd", "ssim", "wm_acc", "wm_ber", "wm_emr"]].mean(numeric_only=True)
    summary_row = {
        "system": "WavMark",
        "wer": float(summary["wer"]),
        "stoi": float(summary["stoi"]),
        "pesq": float(summary["pesq"]),
        "mcd": float(summary["mcd"]),
        "ssim": float(summary["ssim"]),
        "wm_acc": float(summary["wm_acc"]),
        "wm_ber": float(summary["wm_ber"]),
        "wm_emr": float(summary["wm_emr"]),
    }
    write_csv(out_dir / "wavmark_quality_system_summary.csv", [summary_row])
    md = (
        "| System | STOI | WER(%) | SSIM | PESQ | MCD | ACC | BER(%) | EMR(%) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        f"| WavMark | {fmt(summary_row['stoi'])} | {fmt(summary_row['wer'], percent=True)} | "
        f"{fmt(summary_row['ssim'])} | {fmt(summary_row['pesq'])} | {fmt(summary_row['mcd'])} | "
        f"{fmt(summary_row['wm_acc'])} | {fmt(summary_row['wm_ber'], percent=True)} | "
        f"{fmt(summary_row['wm_emr'], percent=True)} |\n"
    )
    (out_dir / "wavmark_quality_summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
