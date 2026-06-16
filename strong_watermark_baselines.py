#!/usr/bin/env python3
"""Run official WavMark and AudioSeal baselines on generated EMG-to-speech audio.

The script uses local official source trees and optional local checkpoint paths.
It intentionally reports missing dependencies/checkpoints instead of silently
falling back, so the revision record is reproducible.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import tempfile
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torchaudio

from evaluate_main_experiment import compute_bit_metrics, normalize_bits
import evaluate_wm_attack_robustness as attack_mod
from evaluate_wm_attack_robustness import load_audio_mono


DEFAULT_TARGET_BITS = "1011001010110010"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official strong audio watermark baselines.")
    parser.add_argument("--manifest_csv", type=str, default="results/eval_manifest.csv")
    parser.add_argument("--output_dir", type=str, default="results/strong_watermark_baselines_seed1234")
    parser.add_argument("--wavmark_src", type=str, default="wavmark-main/src")
    parser.add_argument("--audioseal_src", type=str, default="audioseal-main/src")
    parser.add_argument("--wavmark_ckpt", type=str, default="")
    parser.add_argument("--audioseal_generator", type=str, default="audioseal_wm_16bits")
    parser.add_argument("--audioseal_detector", type=str, default="audioseal_detector_16bits")
    parser.add_argument("--target_bits", type=str, default=DEFAULT_TARGET_BITS)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--run_wavmark", action="store_true", default=False)
    parser.add_argument("--run_audioseal", action="store_true", default=False)
    parser.add_argument("--attacks", action="store_true", default=False)
    parser.add_argument("--audioseal_alpha", type=float, default=1.0)
    parser.add_argument("--wavmark_min_snr", type=float, default=20.0)
    parser.add_argument("--wavmark_max_snr", type=float, default=38.0)
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
        help="Optional read-only site-packages paths used only for missing pure-Python deps.",
    )
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def configure_attack_runtime(out_dir: Path) -> None:
    tmp_dir = out_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    os.environ["TMPDIR"] = str(tmp_dir)
    tempfile.tempdir = str(tmp_dir)

    candidates = [
        shutil.which("ffmpeg"),
        r"E:\postgraduate\anaconda3\envs\hifigan\Library\bin\ffmpeg.exe",
        r"E:\postgraduate\anaconda3\envs\diffwave\Library\bin\ffmpeg.exe",
        r"E:\postgraduate\anaconda3\Library\bin\ffmpeg.exe",
    ]
    for cand in candidates:
        if cand and Path(cand).exists():
            attack_mod.FFMPEG_BIN = str(cand)
            break
    attack_mod.TMP_WORK_DIR = str((out_dir / ".tmp_attack").resolve())
    Path(attack_mod.TMP_WORK_DIR).mkdir(parents=True, exist_ok=True)


def add_source_path(path: str) -> None:
    p = str(Path(path).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def add_extra_site_packages(paths: str) -> None:
    for raw in str(paths or "").split(";"):
        raw = raw.strip()
        if not raw:
            continue
        p = str(Path(raw).resolve())
        if Path(p).exists() and p not in sys.path:
            sys.path.append(p)


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def install_resampy_import_shim() -> None:
    if module_available("resampy"):
        return

    def _missing_resample(*_args, **_kwargs):
        raise ModuleNotFoundError("resampy is unavailable; this shim only permits wavmark import.")

    sys.modules.setdefault("resampy", types.SimpleNamespace(resample=_missing_resample))


def find_default_wavmark_ckpt() -> str:
    roots = [
        Path.home() / ".cache" / "huggingface" / "hub" / "models--M4869--WavMark" / "snapshots",
        Path("wavmark-main"),
    ]
    for root in roots:
        if not root.exists():
            continue
        matches = sorted(root.rglob("*.pkl"))
        if matches:
            return str(matches[0])
    return ""


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: Sequence[Dict[str, object]], cols: Sequence[str]) -> str:
    if not rows:
        return ""
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append("nan" if math.isnan(value) else f"{value:.4f}")
            else:
                vals.append(str(value))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def load_manifest(path: str, max_samples: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "baseline_wav" not in df.columns:
        raise ValueError("manifest must include baseline_wav column")
    if max_samples is not None:
        df = df.iloc[: max(0, int(max_samples))].copy()
    return df.reset_index(drop=True)


def to_numpy_16k(wav_path: str) -> np.ndarray:
    wav, sr = load_audio_mono(wav_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    x = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return np.clip(x, -1.0, 1.0)


def attack_numpy_16k(signal: np.ndarray, attack_name: str) -> np.ndarray:
    x = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0)
    target_len = int(x.shape[-1])
    fns = {
        "clean": lambda wav, sr, n: (wav, sr),
        "gaussian_snr20db": attack_mod.attack_gaussian_20db,
        "mp3_64k": attack_mod.attack_mp3_64k,
        "resample_22k_16k_22k": attack_mod.attack_resample_22k_16k_22k,
        "lowpass_3khz": attack_mod.attack_lowpass_3khz,
        "shift_plus_4ms": attack_mod.attack_shift_plus_4ms,
        "shift_minus_4ms": attack_mod.attack_shift_minus_4ms,
        "gain_plus15pct": attack_mod.attack_gain_plus15pct,
    }
    y, sr = fns[attack_name](x, 16000, target_len)
    if sr != 16000:
        y = torchaudio.functional.resample(y, sr, 16000)
    y = y[..., :target_len]
    if int(y.shape[-1]) < target_len:
        y = torch.nn.functional.pad(y, (0, target_len - int(y.shape[-1])))
    return np.clip(y.squeeze(0).detach().cpu().numpy().astype(np.float32), -1.0, 1.0)


def bit_array(bits: str, length: int) -> np.ndarray:
    bits = normalize_bits(bits, length)
    return np.asarray([1 if ch == "1" else 0 for ch in bits], dtype=np.int64)


def run_wavmark(args: argparse.Namespace, manifest: pd.DataFrame, out_dir: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    add_source_path(args.wavmark_src)
    install_resampy_import_shim()
    import wavmark  # type: ignore

    ckpt = args.wavmark_ckpt or find_default_wavmark_ckpt()
    if ckpt:
        model = wavmark.load_model(ckpt)
    else:
        model = wavmark.load_model()
    model = model.to(args.device)
    model.eval()
    payload = bit_array(args.target_bits, 16)
    attacks = [
        "clean",
        "gaussian_snr20db",
        "mp3_64k",
        "resample_22k_16k_22k",
        "lowpass_3khz",
        "shift_plus_4ms",
        "shift_minus_4ms",
        "gain_plus15pct",
    ] if args.attacks else ["clean"]

    rows: List[Dict[str, object]] = []
    for _, rec in manifest.iterrows():
        signal = to_numpy_16k(str(rec["baseline_wav"]))
        start = time.perf_counter()
        watermarked, info = wavmark.encode_watermark(
            model,
            signal,
            payload,
            min_snr=float(args.wavmark_min_snr),
            max_snr=float(args.wavmark_max_snr),
            show_progress=False,
        )
        encode_time = time.perf_counter() - start
        for attack_name in attacks:
            attacked = watermarked if attack_name == "clean" else attack_numpy_16k(watermarked, attack_name)
            start = time.perf_counter()
            decoded, dec_info = wavmark.decode_watermark(model, attacked, show_progress=False)
            decode_time = time.perf_counter() - start
            pred = "" if decoded is None else "".join("1" if int(v) else "0" for v in np.asarray(decoded).reshape(-1).tolist())
            pred = normalize_bits(pred, len(args.target_bits))
            acc, ber, emr = compute_bit_metrics(pred, args.target_bits)
            rows.append(
                {
                    "id": str(rec["id"]),
                    "baseline": "WavMark",
                    "attack": attack_name,
                    "pred_bits": pred,
                    "wm_acc": acc,
                    "wm_ber": ber,
                    "wm_emr": emr,
                    "encode_time_sec": encode_time,
                    "decode_time_sec": decode_time,
                    "snr": float(info.get("snr", float("nan"))) if isinstance(info, dict) else float("nan"),
                    "num_decode_hits": len(dec_info.get("results", [])) if isinstance(dec_info, dict) else 0,
                }
            )
    summary = summarize(rows)
    write_csv(out_dir / "wavmark_utterance_metrics.csv", rows)
    write_csv(out_dir / "wavmark_system_summary.csv", summary)
    return rows, summary


def run_audioseal(args: argparse.Namespace, manifest: pd.DataFrame, out_dir: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    add_source_path(args.audioseal_src)
    try:
        import torch._dynamo  # type: ignore

        torch._dynamo.config.suppress_errors = True
    except Exception:
        pass
    from audioseal import AudioSeal  # type: ignore

    generator = AudioSeal.load_generator(args.audioseal_generator).to(args.device)
    detector = AudioSeal.load_detector(args.audioseal_detector).to(args.device)
    generator.eval()
    detector.eval()
    message = torch.tensor(bit_array(args.target_bits, 16), dtype=torch.float32, device=args.device).unsqueeze(0)
    attacks = [
        "clean",
        "gaussian_snr20db",
        "mp3_64k",
        "resample_22k_16k_22k",
        "lowpass_3khz",
        "shift_plus_4ms",
        "shift_minus_4ms",
        "gain_plus15pct",
    ] if args.attacks else ["clean"]

    rows: List[Dict[str, object]] = []
    for _, rec in manifest.iterrows():
        signal = to_numpy_16k(str(rec["baseline_wav"]))
        wav = torch.from_numpy(signal).to(args.device).view(1, 1, -1)
        with torch.no_grad():
            start = time.perf_counter()
            watermarked = generator(wav, sample_rate=16000, message=message, alpha=float(args.audioseal_alpha))
            encode_time = time.perf_counter() - start
        watermarked_np = watermarked.detach().cpu().numpy().reshape(-1).astype(np.float32)
        for attack_name in attacks:
            attacked_np = watermarked_np if attack_name == "clean" else attack_numpy_16k(watermarked_np, attack_name)
            attacked = torch.from_numpy(attacked_np).to(args.device).view(1, 1, -1)
            with torch.no_grad():
                start = time.perf_counter()
                detect_logits, decoded = detector(attacked, sample_rate=16000)
                decode_time = time.perf_counter() - start
            pred_arr = (decoded.detach().cpu().numpy().reshape(-1) >= 0.5).astype(np.int64)
            pred = normalize_bits("".join("1" if int(v) else "0" for v in pred_arr.tolist()), len(args.target_bits))
            detect_prob = float(detect_logits[:, 1, :].detach().float().mean().cpu().item())
            acc, ber, emr = compute_bit_metrics(pred, args.target_bits)
            rows.append(
                {
                    "id": str(rec["id"]),
                    "baseline": "AudioSeal",
                    "attack": attack_name,
                    "pred_bits": pred,
                    "wm_acc": acc,
                    "wm_ber": ber,
                    "wm_emr": emr,
                    "detect_prob": detect_prob,
                    "encode_time_sec": encode_time,
                    "decode_time_sec": decode_time,
                }
            )
    summary = summarize(rows)
    write_csv(out_dir / "audioseal_utterance_metrics.csv", rows)
    write_csv(out_dir / "audioseal_system_summary.csv", summary)
    return rows, summary


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    grouped = df.groupby(["baseline", "attack"], as_index=False)
    summary = grouped.agg(
        wm_acc=("wm_acc", "mean"),
        wm_ber=("wm_ber", "mean"),
        wm_emr=("wm_emr", "mean"),
        encode_time_sec=("encode_time_sec", "mean"),
        decode_time_sec=("decode_time_sec", "mean"),
    )
    return summary.to_dict("records")


def dependency_report(args: argparse.Namespace) -> List[Dict[str, object]]:
    add_source_path(args.wavmark_src)
    add_source_path(args.audioseal_src)
    deps = ["resampy", "huggingface_hub", "einops", "omegaconf", "librosa", "soundfile"]
    rows = [{"dependency": dep, "available": module_available(dep)} for dep in deps]
    wavmark_ckpt = args.wavmark_ckpt or find_default_wavmark_ckpt()
    rows.append({"dependency": "wavmark_checkpoint_path", "available": bool(wavmark_ckpt and Path(wavmark_ckpt).exists())})
    rows.append({"dependency": "audioseal_generator_card_or_path", "available": bool(args.audioseal_generator)})
    rows.append({"dependency": "audioseal_detector_card_or_path", "available": bool(args.audioseal_detector)})
    rows.append({"dependency": "ffmpeg_for_mp3_attack", "available": bool(attack_mod.FFMPEG_BIN)})
    return rows


def main() -> None:
    args = parse_args()
    args.target_bits = normalize_bits(args.target_bits, len("".join(ch for ch in args.target_bits if ch in "01")))
    out_dir = ensure_dir(args.output_dir)
    configure_attack_runtime(out_dir)
    add_extra_site_packages(args.extra_site_packages)
    manifest = load_manifest(args.manifest_csv, args.max_samples)

    report = dependency_report(args)
    write_csv(out_dir / "dependency_report.csv", report)
    (out_dir / "dependency_report.md").write_text(
        markdown_table(report, ["dependency", "available"]) + "\n", encoding="utf-8"
    )

    all_summary: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []
    if args.run_wavmark:
        try:
            _, summary = run_wavmark(args, manifest, out_dir)
            all_summary.extend(summary)
        except Exception as exc:
            errors.append({"baseline": "WavMark", "error": repr(exc)})
    if args.run_audioseal:
        try:
            _, summary = run_audioseal(args, manifest, out_dir)
            all_summary.extend(summary)
        except Exception as exc:
            errors.append({"baseline": "AudioSeal", "error": repr(exc)})

    write_csv(out_dir / "strong_baseline_summary.csv", all_summary)
    if all_summary:
        (out_dir / "strong_baseline_summary.md").write_text(
            markdown_table(all_summary, ["baseline", "attack", "wm_acc", "wm_ber", "wm_emr", "encode_time_sec", "decode_time_sec"]) + "\n",
            encoding="utf-8",
        )
    write_csv(out_dir / "errors.csv", errors)
    if errors:
        (out_dir / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Saved] Strong baseline outputs to {out_dir}")
    if errors:
        print(json.dumps(errors, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
