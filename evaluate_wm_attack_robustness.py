#!/usr/bin/env python3
"""Watermark robustness evaluation under fixed audio post-processing attacks.

Evaluate Ours vs Post-hoc watermark extraction performance after attacks:
- Gaussian noise (20 dB SNR)
- MP3 compression
- Resample 22.05k -> 16k -> 22.05k
- Low-pass filter (3 kHz)
- Temporal shift (+/- 4 ms)
- Gain (+15%)

Outputs:
- attack_utterance_metrics.csv
- attack_system_summary.csv
- attack_comparison_table.csv
- attack_comparison_table.md
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import subprocess
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio

try:
    import librosa
except Exception:
    librosa = None  # type: ignore

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None  # type: ignore

from evaluate_main_experiment import (
    _decode_posthoc_lsb_from_wav,
    _decode_posthoc_spread_from_wav,
    bits_from_prob_vector,
    compute_bit_metrics,
    extract_probs_with_ours_extractor,
    load_audio_mono,
    load_ours_extractor,
    normalize_bits,
    parse_posthoc_method,
    run_crossfit_calibration,
)


ATTACK_SR = 22050
DEFAULT_TARGET_BITS = "1011001010110010"
ATTACK_NAMES = [
    "gaussian_snr20db",
    "mp3_64k",
    "resample_22k_16k_22k",
    "lowpass_3khz",
    "shift_plus_4ms",
    "shift_minus_4ms",
    "gain_plus15pct",
]
FFMPEG_BIN: Optional[str] = None
FFPROBE_BIN: Optional[str] = None
TMP_WORK_DIR: str = str((Path(__file__).resolve().parent / "results" / ".tmp_attack").resolve())


@dataclass
class InputRecord:
    sample_id: str
    ours_wav: str
    posthoc_wav: str
    posthoc_result: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate watermark robustness under fixed attack configuration.")
    parser.add_argument(
        "--manifest_csv",
        type=str,
        default=r"E:\postgraduate\project\EMG-Speech\results\eval_manifest.csv",
        help="Input manifest with ours_wav/posthoc_wav/posthoc_result columns.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"E:\postgraduate\project\EMG-Speech\results\attack_eval_seed1234",
        help="Output directory for attack robustness reports.",
    )
    parser.add_argument(
        "--ours_extractor_ckpt",
        type=str,
        default=r"E:\postgraduate\project\EMG-Speech\checkpoints（80）\diffwave-watermark_extractor_epoch_61.pth",
        help="Checkpoint for Ours WatermarkExtractor.",
    )
    parser.add_argument(
        "--target_bits",
        type=str,
        default=DEFAULT_TARGET_BITS,
        help="Global target bits (fixed payload).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device for watermark extraction.",
    )
    parser.add_argument("--extractor_sr", type=int, default=22050, help="Extractor input sample rate.")
    parser.add_argument("--wm_decode_mode", type=str, default="windowed", choices=["single", "windowed"])
    parser.add_argument("--wm_window_sec", type=float, default=1.0)
    parser.add_argument("--wm_window_hop_sec", type=float, default=0.5)
    parser.add_argument("--wm_calibration", type=str, default="crossfit", choices=["none", "crossfit"])
    parser.add_argument("--wm_crossfit_folds", type=int, default=5)
    parser.add_argument(
        "--wm_enable_bit_flip",
        dest="wm_enable_bit_flip",
        action="store_true",
        help="Enable per-bit inversion in cross-fit calibration.",
    )
    parser.add_argument(
        "--wm_disable_bit_flip",
        dest="wm_enable_bit_flip",
        action="store_false",
        help="Disable per-bit inversion in cross-fit calibration.",
    )
    parser.set_defaults(wm_enable_bit_flip=True)
    parser.add_argument("--posthoc_spread_seed", type=int, default=1234)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--include_clean",
        action="store_true",
        default=False,
        help="Include clean/no-attack rows in addition to attacked rows.",
    )
    parser.add_argument("--ffmpeg_bin", type=str, default="", help="Optional explicit path to ffmpeg executable.")
    parser.add_argument("--ffprobe_bin", type=str, default="", help="Optional explicit path to ffprobe executable.")

    parser.add_argument("--col_id", type=str, default="id")
    parser.add_argument("--col_ours_wav", type=str, default="ours_wav")
    parser.add_argument("--col_posthoc_wav", type=str, default="posthoc_wav")
    parser.add_argument("--col_posthoc_result", type=str, default="posthoc_result")
    return parser.parse_args()


def _resolve_bin(provided: str, binary_name: str, extra_candidates: List[str]) -> Optional[str]:
    if provided:
        p = Path(provided)
        if p.exists():
            return str(p)
    found = shutil.which(binary_name)
    if found:
        return found
    for cand in extra_candidates:
        cp = Path(cand)
        if cp.exists():
            return str(cp)
    return None


def ensure_runtime_dependencies(args: argparse.Namespace) -> Tuple[str, str]:
    missing = []
    if AudioSegment is None:
        missing.append("pydub")
    if librosa is None:
        missing.append("librosa")

    ffmpeg_path = _resolve_bin(
        args.ffmpeg_bin,
        "ffmpeg",
        [
            r"E:\postgraduate\anaconda3\envs\hifigan\Library\bin\ffmpeg.exe",
            r"E:\postgraduate\anaconda3\Library\bin\ffmpeg.exe",
        ],
    )
    ffprobe_path = _resolve_bin(
        args.ffprobe_bin,
        "ffprobe",
        [
            r"E:\postgraduate\anaconda3\envs\hifigan\Library\bin\ffprobe.exe",
            r"E:\postgraduate\anaconda3\Library\bin\ffprobe.exe",
        ],
    )
    if not ffmpeg_path:
        missing.append("ffmpeg")
    if not ffprobe_path:
        missing.append("ffprobe")
    if missing:
        raise RuntimeError(
            "Missing required dependencies for attack evaluation: "
            + ", ".join(missing)
            + ". Please install them before running this script."
        )

    if AudioSegment is not None:
        AudioSegment.converter = ffmpeg_path
        AudioSegment.ffprobe = ffprobe_path
    assert ffmpeg_path is not None and ffprobe_path is not None
    return ffmpeg_path, ffprobe_path

def load_manifest(args: argparse.Namespace) -> List[InputRecord]:
    p = Path(args.manifest_csv)
    if not p.exists():
        raise FileNotFoundError(f"manifest_csv not found: {p}")
    df = pd.read_csv(p)

    required = [args.col_id, args.col_ours_wav, args.col_posthoc_wav, args.col_posthoc_result]
    miss_cols = [c for c in required if c not in df.columns]
    if miss_cols:
        raise ValueError(f"Manifest missing required columns: {miss_cols}")

    out: List[InputRecord] = []
    for _, row in df.iterrows():
        sid = str(row[args.col_id]).strip()
        ours_wav = str(row[args.col_ours_wav]).strip()
        posthoc_wav = str(row[args.col_posthoc_wav]).strip()
        posthoc_result = str(row[args.col_posthoc_result]).strip()
        if (not sid) or (not ours_wav) or (not posthoc_wav):
            continue
        if (not os.path.exists(ours_wav)) or (not os.path.exists(posthoc_wav)):
            continue
        out.append(
            InputRecord(
                sample_id=sid,
                ours_wav=ours_wav,
                posthoc_wav=posthoc_wav,
                posthoc_result=posthoc_result,
            )
        )
    if args.max_samples is not None:
        out = out[: max(0, int(args.max_samples))]
    if not out:
        raise RuntimeError("No valid records found in manifest.")
    return out


def _resample(wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return wav
    return torchaudio.functional.resample(wav, src_sr, dst_sr)


def _fix_len(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    cur = int(wav.shape[-1])
    if cur == target_len:
        return wav
    if cur > target_len:
        return wav[..., :target_len]
    pad = target_len - cur
    return torch.nn.functional.pad(wav, (0, pad))


def _to_attack_sr_and_len(wav: torch.Tensor, sr: int, target_len: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    x = _resample(wav, sr, ATTACK_SR).float().clamp(-1.0, 1.0)
    if target_len is not None:
        x = _fix_len(x, target_len)
    return x, ATTACK_SR


def _wav_to_bytes(wav: torch.Tensor, sr: int) -> io.BytesIO:
    buf = io.BytesIO()
    torchaudio.save(buf, wav.cpu(), sample_rate=int(sr), format="wav")
    buf.seek(0)
    return buf


def attack_clean(wav: torch.Tensor, sr: int, target_len: int) -> Tuple[torch.Tensor, int]:
    x, _ = _to_attack_sr_and_len(wav, sr, target_len=target_len)
    return x, ATTACK_SR


def attack_gaussian_20db(wav: torch.Tensor, sr: int, target_len: int) -> Tuple[torch.Tensor, int]:
    x, _ = _to_attack_sr_and_len(wav, sr, target_len=target_len)
    p_sig = float(torch.mean(x * x).item())
    if p_sig <= 1e-12:
        return x, ATTACK_SR
    p_noise = p_sig / (10.0 ** (20.0 / 10.0))
    noise_std = math.sqrt(max(p_noise, 0.0))
    noise = torch.randn_like(x) * noise_std
    y = (x + noise).clamp(-1.0, 1.0)
    return y, ATTACK_SR


def attack_mp3_64k(wav: torch.Tensor, sr: int, target_len: int) -> Tuple[torch.Tensor, int]:
    x, _ = _to_attack_sr_and_len(wav, sr, target_len=target_len)
    if not FFMPEG_BIN:
        raise RuntimeError("FFMPEG_BIN is not configured.")

    uid = uuid.uuid4().hex
    in_wav = os.path.join(TMP_WORK_DIR, f"{uid}_in.wav")
    mid_mp3 = os.path.join(TMP_WORK_DIR, f"{uid}_mid.mp3")
    out_wav = os.path.join(TMP_WORK_DIR, f"{uid}_out.wav")
    try:
        torchaudio.save(in_wav, x.cpu(), sample_rate=ATTACK_SR, format="wav")
        enc_cmd = [
            FFMPEG_BIN,
            "-y",
            "-loglevel",
            "error",
            "-i",
            in_wav,
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "64k",
            mid_mp3,
        ]
        dec_cmd = [
            FFMPEG_BIN,
            "-y",
            "-loglevel",
            "error",
            "-i",
            mid_mp3,
            "-ac",
            "1",
            "-ar",
            str(ATTACK_SR),
            out_wav,
        ]
        subprocess.run(enc_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        subprocess.run(dec_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        y, y_sr = torchaudio.load(out_wav)
    finally:
        for p in (in_wav, mid_mp3, out_wav):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    if y.ndim == 2 and y.shape[0] > 1:
        y = y.mean(dim=0, keepdim=True)
    if y.ndim == 1:
        y = y.unsqueeze(0)
    y = _resample(y.float().clamp(-1.0, 1.0), int(y_sr), ATTACK_SR)
    y = _fix_len(y, target_len)
    return y, ATTACK_SR


def attack_resample_22k_16k_22k(wav: torch.Tensor, sr: int, target_len: int) -> Tuple[torch.Tensor, int]:
    x, _ = _to_attack_sr_and_len(wav, sr, target_len=target_len)
    y = _resample(x, ATTACK_SR, 16000)
    y = _resample(y, 16000, ATTACK_SR)
    y = _fix_len(y, target_len).clamp(-1.0, 1.0)
    return y, ATTACK_SR


def attack_lowpass_3khz(wav: torch.Tensor, sr: int, target_len: int) -> Tuple[torch.Tensor, int]:
    x, _ = _to_attack_sr_and_len(wav, sr, target_len=target_len)
    y = torchaudio.functional.lowpass_biquad(x, ATTACK_SR, cutoff_freq=3000.0)
    y = _fix_len(y, target_len).clamp(-1.0, 1.0)
    return y, ATTACK_SR


def _shift_samples(wav: torch.Tensor, shift: int) -> torch.Tensor:
    if shift == 0:
        return wav
    t = int(wav.shape[-1])
    if abs(shift) >= t:
        return torch.zeros_like(wav)
    z = torch.zeros((wav.shape[0], abs(shift)), dtype=wav.dtype, device=wav.device)
    if shift > 0:
        return torch.cat([z, wav[..., : t - shift]], dim=-1)
    s = abs(shift)
    return torch.cat([wav[..., s:], z], dim=-1)


def attack_shift_plus_4ms(wav: torch.Tensor, sr: int, target_len: int) -> Tuple[torch.Tensor, int]:
    x, _ = _to_attack_sr_and_len(wav, sr, target_len=target_len)
    shift = int(round(0.004 * ATTACK_SR))
    y = _shift_samples(x, shift)
    y = _fix_len(y, target_len).clamp(-1.0, 1.0)
    return y, ATTACK_SR


def attack_shift_minus_4ms(wav: torch.Tensor, sr: int, target_len: int) -> Tuple[torch.Tensor, int]:
    x, _ = _to_attack_sr_and_len(wav, sr, target_len=target_len)
    shift = int(round(0.004 * ATTACK_SR))
    y = _shift_samples(x, -shift)
    y = _fix_len(y, target_len).clamp(-1.0, 1.0)
    return y, ATTACK_SR


def attack_gain_plus15pct(wav: torch.Tensor, sr: int, target_len: int) -> Tuple[torch.Tensor, int]:
    x, _ = _to_attack_sr_and_len(wav, sr, target_len=target_len)
    y = (x * 1.15).clamp(-1.0, 1.0)
    return y, ATTACK_SR


def get_attack_fns(include_clean: bool) -> List[Tuple[str, object]]:
    attacks = [
        ("gaussian_snr20db", attack_gaussian_20db),
        ("mp3_64k", attack_mp3_64k),
        ("resample_22k_16k_22k", attack_resample_22k_16k_22k),
        ("lowpass_3khz", attack_lowpass_3khz),
        ("shift_plus_4ms", attack_shift_plus_4ms),
        ("shift_minus_4ms", attack_shift_minus_4ms),
        ("gain_plus15pct", attack_gain_plus15pct),
    ]
    if include_clean:
        attacks = [("clean", attack_clean)] + attacks
    return attacks


def decode_posthoc_from_tensor(wav: torch.Tensor, method: str, target_len: int, spread_seed: int) -> Optional[str]:
    m = str(method).strip().lower()
    try:
        if m == "spread":
            bits = _decode_posthoc_spread_from_wav(wav, target_len, seed=int(spread_seed))
            return normalize_bits(bits, target_len)
        bits = _decode_posthoc_lsb_from_wav(wav, target_len)
        return normalize_bits(bits, target_len)
    except Exception:
        return None


def render_comparison_markdown(df: pd.DataFrame) -> str:
    lines = [
        "| Attack | Ours ACC(%) | Ours BER(%) | Post-hoc ACC(%) | Post-hoc BER(%) | Delta ACC(pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| {a} | {oa:.3f} | {ob:.3f} | {pa:.3f} | {pb:.3f} | {d:.3f} |".format(
                a=row["attack"],
                oa=row["ours_acc"] * 100.0 if math.isfinite(float(row["ours_acc"])) else float("nan"),
                ob=row["ours_ber"] * 100.0 if math.isfinite(float(row["ours_ber"])) else float("nan"),
                pa=row["posthoc_acc"] * 100.0 if math.isfinite(float(row["posthoc_acc"])) else float("nan"),
                pb=row["posthoc_ber"] * 100.0 if math.isfinite(float(row["posthoc_ber"])) else float("nan"),
                d=row["delta_acc"] * 100.0 if math.isfinite(float(row["delta_acc"])) else float("nan"),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(TMP_WORK_DIR, exist_ok=True)
    global FFMPEG_BIN, FFPROBE_BIN
    FFMPEG_BIN, FFPROBE_BIN = ensure_runtime_dependencies(args)

    target_bits = normalize_bits(args.target_bits, len("".join(ch for ch in str(args.target_bits) if ch in "01")))
    if len(target_bits) == 0:
        raise ValueError("target_bits must include at least one binary character.")
    n_bits = len(target_bits)

    records = load_manifest(args)
    attacks = get_attack_fns(include_clean=bool(args.include_clean))
    print(f"[Info] records={len(records)}, attacks={len(attacks)}")

    extractor = load_ours_extractor(args.ours_extractor_ckpt, args.device, target_bits_len=n_bits)

    rows: List[Dict[str, object]] = []
    ours_idx_by_attack: Dict[str, List[int]] = {k: [] for k, _ in attacks}
    ours_prob_by_attack: Dict[str, List[Optional[np.ndarray]]] = {k: [] for k, _ in attacks}
    diagnostics: Dict[str, object] = {}

    for i, rec in enumerate(records, start=1):
        ours_wav, ours_sr = load_audio_mono(rec.ours_wav)
        post_wav, post_sr = load_audio_mono(rec.posthoc_wav)
        ours_base, _ = _to_attack_sr_and_len(ours_wav, ours_sr)
        target_len = int(ours_base.shape[-1])

        method = parse_posthoc_method(rec.posthoc_result) or "lsb"
        method = method.lower()

        for attack_name, attack_fn in attacks:
            ours_att, ours_att_sr = attack_fn(ours_wav, ours_sr, target_len)
            post_att, _ = attack_fn(post_wav, post_sr, target_len)

            prob_vec = extract_probs_with_ours_extractor(
                extractor,
                wav=ours_att,
                sr=ours_att_sr,
                extractor_sr=args.extractor_sr,
                target_len=n_bits,
                device=args.device,
                decode_mode=args.wm_decode_mode,
                window_sec=args.wm_window_sec,
                window_hop_sec=args.wm_window_hop_sec,
            )
            ours_pred_raw = bits_from_prob_vector(prob_vec, thresholds=0.5, bit_flip=None)
            ours_acc, ours_ber, ours_emr = compute_bit_metrics(ours_pred_raw, target_bits)
            idx_row = len(rows)
            rows.append(
                {
                    "id": rec.sample_id,
                    "attack": attack_name,
                    "system": "Ours",
                    "wm_pred_bits_raw": ours_pred_raw,
                    "wm_pred_bits": ours_pred_raw,
                    "wm_acc": ours_acc,
                    "wm_ber": ours_ber,
                    "wm_emr": ours_emr,
                }
            )
            ours_idx_by_attack[attack_name].append(idx_row)
            ours_prob_by_attack[attack_name].append(prob_vec)

            post_pred = decode_posthoc_from_tensor(
                wav=post_att,
                method=method,
                target_len=n_bits,
                spread_seed=args.posthoc_spread_seed,
            )
            post_acc, post_ber, post_emr = compute_bit_metrics(post_pred, target_bits)
            rows.append(
                {
                    "id": rec.sample_id,
                    "attack": attack_name,
                    "system": "Post-hoc",
                    "wm_pred_bits_raw": post_pred,
                    "wm_pred_bits": post_pred,
                    "wm_acc": post_acc,
                    "wm_ber": post_ber,
                    "wm_emr": post_emr,
                }
            )

        if i % 10 == 0 or i == len(records):
            print(f"[Progress] {i}/{len(records)}")

    if str(args.wm_calibration).strip().lower() == "crossfit":
        for attack_name, _ in attacks:
            idxs = ours_idx_by_attack[attack_name]
            probs = ours_prob_by_attack[attack_name]
            valid_pairs: List[Tuple[int, np.ndarray]] = []
            for idx_row, pv in zip(idxs, probs):
                if pv is None:
                    continue
                valid_pairs.append((idx_row, np.asarray(pv, dtype=np.float32)))

            if len(valid_pairs) < 2:
                warnings.warn(
                    f"Attack={attack_name}: fewer than 2 valid Ours probability vectors, skip calibration.",
                    RuntimeWarning,
                )
                continue

            prob_matrix = np.stack([v for _, v in valid_pairs], axis=0).astype(np.float32)
            cal_bits, raw_bits, diag = run_crossfit_calibration(
                prob_matrix=prob_matrix,
                target_bits=target_bits,
                folds=int(args.wm_crossfit_folds),
                enable_bit_flip=bool(args.wm_enable_bit_flip),
                seed=1234,
            )

            for j, (idx_row, _) in enumerate(valid_pairs):
                rows[idx_row]["wm_pred_bits_raw"] = raw_bits[j]
                rows[idx_row]["wm_pred_bits"] = cal_bits[j]
                acc, ber, emr = compute_bit_metrics(cal_bits[j], target_bits)
                rows[idx_row]["wm_acc"] = acc
                rows[idx_row]["wm_ber"] = ber
                rows[idx_row]["wm_emr"] = emr

            diagnostics[attack_name] = diag

    out_dir = Path(args.output_dir)
    utter_df = pd.DataFrame(rows)
    utter_path = out_dir / "attack_utterance_metrics.csv"
    utter_df.to_csv(utter_path, index=False, encoding="utf-8")

    summary_df = (
        utter_df.groupby(["attack", "system"], as_index=False)[["wm_acc", "wm_ber", "wm_emr"]]
        .mean(numeric_only=True)
        .sort_values(["attack", "system"])
        .reset_index(drop=True)
    )
    summary_path = out_dir / "attack_system_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    ours_df = summary_df[summary_df["system"] == "Ours"][["attack", "wm_acc", "wm_ber"]].rename(
        columns={"wm_acc": "ours_acc", "wm_ber": "ours_ber"}
    )
    post_df = summary_df[summary_df["system"] == "Post-hoc"][["attack", "wm_acc", "wm_ber"]].rename(
        columns={"wm_acc": "posthoc_acc", "wm_ber": "posthoc_ber"}
    )
    comp_df = pd.merge(ours_df, post_df, on="attack", how="outer").sort_values("attack").reset_index(drop=True)
    comp_df["delta_acc"] = comp_df["ours_acc"] - comp_df["posthoc_acc"]
    comp_df["delta_ber"] = comp_df["ours_ber"] - comp_df["posthoc_ber"]
    comp_path = out_dir / "attack_comparison_table.csv"
    comp_df.to_csv(comp_path, index=False, encoding="utf-8")

    md_text = render_comparison_markdown(comp_df)
    md_path = out_dir / "attack_comparison_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    diag_path = out_dir / "attack_calibration_diagnostics.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print(f"[Saved] utterance: {utter_path}")
    print(f"[Saved] summary: {summary_path}")
    print(f"[Saved] comparison csv: {comp_path}")
    print(f"[Saved] comparison md: {md_path}")
    print(f"[Saved] calibration diagnostics: {diag_path}")


if __name__ == "__main__":
    main()


