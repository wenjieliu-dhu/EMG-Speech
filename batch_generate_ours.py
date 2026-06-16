#!/usr/bin/env python3
"""Batched Ours-only DiffWave generation for server runs.

This script groups mel spectrograms by length, pads within a batch, generates
watermarked audio, and slices each output back to its original mel length.
It reuses the model-building utilities from generate_system_wavs.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchaudio

import generate_system_wavs as gen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched Ours-only watermarked DiffWave generation.")
    parser.add_argument("--input_manifest_csv", required=True)
    parser.add_argument("--output_manifest_csv", default="results/ours_seed3456/generated_manifest.csv")
    parser.add_argument("--output_dir", default="results/ours_seed3456/generated_wavs")
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--mask_ckpt", required=True)
    parser.add_argument("--wm_ckpt", required=True)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=3456)
    parser.add_argument("--target_bits", default="1011001010110010")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_total_mel_frames", type=int, default=2400)
    parser.add_argument("--col_id", default="id")
    parser.add_argument("--col_mel", default="mel_npy")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_manifest(path: str, id_col: str, mel_col: str) -> List[Dict[str, str]]:
    base = Path(path).resolve().parent
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if id_col not in (reader.fieldnames or []):
            raise ValueError(f"missing ID column: {id_col}")
        if mel_col not in (reader.fieldnames or []):
            raise ValueError(f"missing mel column: {mel_col}")
        for row in reader:
            sid = str(row[id_col]).strip()
            mel_raw = str(row[mel_col]).strip()
            if not sid or not mel_raw:
                continue
            mel_path = Path(mel_raw)
            if not mel_path.is_absolute():
                mel_path = base / mel_path
            if not mel_path.exists():
                continue
            arr = np.load(str(mel_path), mmap_mode="r")
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr_shape = arr.shape[1:]
            else:
                arr_shape = arr.shape
            if len(arr_shape) != 2:
                continue
            n_frames = int(arr_shape[1] if arr_shape[0] == 80 else arr_shape[0])
            rows.append({"id": sid, "mel_npy": str(mel_path), "n_frames": str(n_frames)})
    if not rows:
        raise RuntimeError("No valid rows found in manifest.")
    return rows


def make_batches(rows: Sequence[Dict[str, str]], batch_size: int, max_total_frames: int) -> List[List[Dict[str, str]]]:
    sorted_rows = sorted(rows, key=lambda r: int(r["n_frames"]))
    batches: List[List[Dict[str, str]]] = []
    cur: List[Dict[str, str]] = []
    cur_max = 0
    for row in sorted_rows:
        frames = int(row["n_frames"])
        next_max = max(cur_max, frames)
        next_count = len(cur) + 1
        too_many = next_count > max(1, batch_size)
        too_large = next_count * next_max > max_total_frames and len(cur) > 0
        if too_many or too_large:
            batches.append(cur)
            cur = []
            cur_max = 0
        cur.append(row)
        cur_max = max(cur_max, frames)
    if cur:
        batches.append(cur)
    return batches


def load_mel(path: str, n_mels: int) -> torch.Tensor:
    mel = torch.from_numpy(np.load(path)).float()
    mel = gen.maybe_fix_spectrogram_shape(mel, n_mels)
    if mel.ndim != 2:
        raise ValueError(f"Expected 2D mel after shape fix, got {tuple(mel.shape)} for {path}")
    return mel


def pad_mels(mels: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, List[int]]:
    lengths = [int(m.shape[-1]) for m in mels]
    max_len = max(lengths)
    n_mels = int(mels[0].shape[0])
    out = torch.zeros((len(mels), n_mels, max_len), dtype=torch.float32)
    for i, mel in enumerate(mels):
        out[i, :, : int(mel.shape[-1])] = mel
    return out, lengths


def generate_batch(
    wm_model,
    batch_rows: Sequence[Dict[str, str]],
    args: argparse.Namespace,
    device: torch.device,
    target_bits: str,
    out_dir: Path,
) -> List[Dict[str, str]]:
    if not batch_rows:
        return []
    if len(batch_rows) > 1:
        try:
            return _generate_batch_once(wm_model, batch_rows, args, device, target_bits, out_dir)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            mid = len(batch_rows) // 2
            print(f"[OOM] split batch {len(batch_rows)} -> {mid}+{len(batch_rows)-mid}", flush=True)
            return generate_batch(wm_model, batch_rows[:mid], args, device, target_bits, out_dir) + generate_batch(
                wm_model, batch_rows[mid:], args, device, target_bits, out_dir
            )
    return _generate_batch_once(wm_model, batch_rows, args, device, target_bits, out_dir)


def _generate_batch_once(
    wm_model,
    batch_rows: Sequence[Dict[str, str]],
    args: argparse.Namespace,
    device: torch.device,
    target_bits: str,
    out_dir: Path,
) -> List[Dict[str, str]]:
    mels = [load_mel(r["mel_npy"], int(wm_model.params.n_mels)) for r in batch_rows]
    spec, lengths = pad_mels(mels)
    batch = int(spec.shape[0])
    gen._fingerprint_global = gen.bits_to_fingerprint(
        target_bits,
        device=device,
        finger_dim=len(target_bits),
        batch=batch,
    ).to(device)
    set_seed(int(args.seed))
    audio, sr = gen.predict_audio(wm_model, spec, device=device, fast_sampling=bool(args.fast))
    audio = audio.detach().cpu()
    outputs: List[Dict[str, str]] = []
    hop = int(wm_model.params.hop_samples)
    for i, row in enumerate(batch_rows):
        sid = gen.sanitize_name(row["id"])
        out_path = out_dir / "ours" / f"{sid}_ours.wav"
        n_audio = int(lengths[i]) * hop
        wav = audio[i : i + 1, :n_audio].clamp(-1.0, 1.0)
        torchaudio.save(str(out_path), wav, sample_rate=int(sr))
        outputs.append({"id": sid, "mel_npy": row["mel_npy"], "ours_wav": str(out_path.resolve())})
    return outputs


def main() -> None:
    args = parse_args()
    target_bits = "".join(ch for ch in str(args.target_bits) if ch in "01")
    if not target_bits:
        raise ValueError("target_bits must contain binary digits.")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA requested but not available.")

    device = torch.device(args.device)
    out_root = Path(args.output_dir).resolve()
    (out_root / "ours").mkdir(parents=True, exist_ok=True)
    Path(args.output_manifest_csv).resolve().parent.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(args.input_manifest_csv, args.col_id, args.col_mel)
    batches = make_batches(rows, int(args.batch_size), int(args.max_total_mel_frames))
    print(
        f"[Info] rows={len(rows)} batches={len(batches)} batch_size={args.batch_size} "
        f"max_total_mel_frames={args.max_total_mel_frames}",
        flush=True,
    )

    print("[Load] watermarked DiffWave checkpoint stack...", flush=True)
    wm_model = gen.build_watermarked_model(
        baseline_ckpt=args.baseline_ckpt,
        mask_ckpt=args.mask_ckpt,
        wm_ckpt=args.wm_ckpt,
        finger_dim=len(target_bits),
        device=device,
    )

    done: List[Dict[str, str]] = []
    skipped = 0
    for bi, batch_rows in enumerate(batches, start=1):
        pending = []
        for row in batch_rows:
            sid = gen.sanitize_name(row["id"])
            out_path = out_root / "ours" / f"{sid}_ours.wav"
            if out_path.exists() and not args.overwrite:
                skipped += 1
                done.append({"id": sid, "mel_npy": row["mel_npy"], "ours_wav": str(out_path.resolve())})
            else:
                pending.append(row)
        if pending:
            done.extend(generate_batch(wm_model, pending, args, device, target_bits, out_root))
        if bi % 5 == 0 or bi == len(batches):
            print(f"[Progress] batches {bi}/{len(batches)} rows_done={len(done)} skipped={skipped}", flush=True)

    done_sorted = sorted(done, key=lambda r: r["id"])
    with open(args.output_manifest_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "mel_npy", "ours_wav"])
        writer.writeheader()
        writer.writerows(done_sorted)

    meta = {
        "seed": int(args.seed),
        "fast": bool(args.fast),
        "batch_size": int(args.batch_size),
        "max_total_mel_frames": int(args.max_total_mel_frames),
        "num_rows": len(rows),
        "num_outputs": len(done_sorted),
    }
    with open(str(Path(args.output_manifest_csv).with_suffix(".meta.json")), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[Done] saved manifest: {Path(args.output_manifest_csv).resolve()}", flush=True)


if __name__ == "__main__":
    main()
