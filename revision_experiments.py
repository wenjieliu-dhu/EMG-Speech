#!/usr/bin/env python3
"""Revision experiments for the SPL resubmission.

This script reuses existing generated audio and the trained watermark extractor
to produce tables requested by reviewers:
- decoding protocol ablation,
- provenance/traceability verification,
- copy-synthesis proxy robustness,
- bootstrap confidence intervals for existing metrics,
- local availability of stronger watermark baselines.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from evaluate_main_experiment import (
    bits_from_prob_vector,
    compute_bit_metrics,
    extract_probs_with_ours_extractor,
    load_audio_mono,
    load_ours_extractor,
    normalize_bits,
    run_crossfit_calibration,
)


DEFAULT_TARGET_BITS = "1011001010110010"
SYSTEMS_FOR_STATS = ("Baseline", "Post-hoc", "Ours")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reviewer-requested revision experiments.")
    parser.add_argument("--manifest_csv", type=str, default="results/eval_manifest.csv")
    parser.add_argument("--main_utterance_csv", type=str, default="results/main_eval_seed1234/utterance_metrics.csv")
    parser.add_argument("--attack_summary_csv", type=str, default="results/attack_eval_seed1234/attack_system_summary.csv")
    parser.add_argument("--output_dir", type=str, default="results/revision_experiments_seed1234")
    parser.add_argument(
        "--ours_extractor_ckpt",
        type=str,
        default=r"checkpoints（80）\16bits\diffwave-watermark_extractor_epoch_61.pth",
    )
    parser.add_argument("--target_bits", type=str, default=DEFAULT_TARGET_BITS)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--extractor_sr", type=int, default=22050)
    parser.add_argument("--window_sec", type=float, default=1.0)
    parser.add_argument("--window_hop_sec", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--bootstrap_iters", type=int, default=2000)
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def bits_to_label(bits: str, registry: Dict[str, str]) -> str:
    bits = normalize_bits(bits, len(next(iter(registry)))) if registry else bits
    return registry.get(bits, "unknown")


def summarize_bits(bits_list: Sequence[Optional[str]], target_bits: str) -> Dict[str, float]:
    accs: List[float] = []
    bers: List[float] = []
    emrs: List[float] = []
    for bits in bits_list:
        acc, ber, emr = compute_bit_metrics(bits, target_bits)
        accs.append(float(acc))
        bers.append(float(ber))
        emrs.append(float(emr))
    return {
        "n": float(len(bits_list)),
        "acc": float(np.nanmean(np.asarray(accs, dtype=np.float64))),
        "ber": float(np.nanmean(np.asarray(bers, dtype=np.float64))),
        "emr": float(np.nanmean(np.asarray(emrs, dtype=np.float64))),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                if math.isnan(val):
                    vals.append("nan")
                elif col.lower() in {"n", "samples"}:
                    vals.append(f"{val:.0f}")
                else:
                    vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body)


def load_manifest(path: Path, max_samples: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["id", "baseline_wav", "posthoc_wav", "ours_wav"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    if max_samples is not None:
        df = df.iloc[: max(0, int(max_samples))].copy()
    return df.reset_index(drop=True)


def collect_probs(
    manifest: pd.DataFrame,
    extractor,
    args: argparse.Namespace,
    target_len: int,
) -> Tuple[Dict[str, List[Optional[np.ndarray]]], List[Dict[str, object]], Dict[str, float]]:
    system_to_col = {
        "Baseline": "baseline_wav",
        "Post-hoc": "posthoc_wav",
        "Ours": "ours_wav",
    }
    decode_specs = [
        ("single", "single"),
        ("windowed", "windowed"),
    ]
    probs: Dict[str, List[Optional[np.ndarray]]] = {}
    rows: List[Dict[str, object]] = []
    timings: Dict[str, List[float]] = {f"{system}_{mode}": [] for system in system_to_col for mode, _ in decode_specs}

    for _, rec in manifest.iterrows():
        sample_id = str(rec["id"])
        for system, col in system_to_col.items():
            wav_path = str(rec[col])
            wav, sr = load_audio_mono(wav_path)
            for mode_name, decode_mode in decode_specs:
                key = f"{system}_{mode_name}"
                start = time.perf_counter()
                prob_vec = extract_probs_with_ours_extractor(
                    extractor,
                    wav=wav,
                    sr=sr,
                    extractor_sr=int(args.extractor_sr),
                    target_len=target_len,
                    device=str(args.device),
                    decode_mode=decode_mode,
                    window_sec=float(args.window_sec),
                    window_hop_sec=float(args.window_hop_sec),
                )
                timings[key].append(time.perf_counter() - start)
                probs.setdefault(key, []).append(prob_vec)
                raw_bits = bits_from_prob_vector(prob_vec, thresholds=0.5, bit_flip=None)
                acc, ber, emr = compute_bit_metrics(raw_bits, args.target_bits)
                rows.append(
                    {
                        "id": sample_id,
                        "system": system,
                        "decode_mode": mode_name,
                        "wav_path": wav_path,
                        "pred_bits_raw": raw_bits or "",
                        "wm_acc": acc,
                        "wm_ber": ber,
                        "wm_emr": emr,
                    }
                )
    timing_summary = {
        key: float(np.nanmean(np.asarray(vals, dtype=np.float64)))
        for key, vals in timings.items()
        if vals
    }
    return probs, rows, timing_summary


def prob_matrix(prob_list: Sequence[Optional[np.ndarray]], target_len: int) -> np.ndarray:
    valid = []
    for prob in prob_list:
        if prob is None:
            valid.append(np.zeros((target_len,), dtype=np.float32))
        else:
            arr = np.asarray(prob, dtype=np.float32).reshape(-1)
            out = np.zeros((target_len,), dtype=np.float32)
            n = min(target_len, arr.size)
            out[:n] = arr[:n]
            valid.append(out)
    return np.stack(valid, axis=0).astype(np.float32)


def run_decoding_ablation(
    probs: Dict[str, List[Optional[np.ndarray]]],
    args: argparse.Namespace,
    target_bits: str,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    target_len = len(target_bits)
    rows: List[Dict[str, object]] = []
    diagnostics: Dict[str, object] = {}

    # Main ablation requested by Reviewer 2: decode protocol over Ours.
    variants = [
        ("base extractor / single segment / threshold 0.5", "Ours_single", "raw", False),
        ("window voting / threshold 0.5", "Ours_windowed", "raw", False),
        ("window voting + cross-fit thresholds", "Ours_windowed", "crossfit", False),
        ("window voting + cross-fit thresholds + bit flip", "Ours_windowed", "crossfit", True),
    ]
    for name, key, mode, enable_flip in variants:
        prob_list = probs[key]
        if mode == "raw":
            bits = [bits_from_prob_vector(p, thresholds=0.5, bit_flip=None) for p in prob_list]
            diag = {}
        else:
            matrix = prob_matrix(prob_list, target_len)
            bits, raw_bits, diag = run_crossfit_calibration(
                prob_matrix=matrix,
                target_bits=target_bits,
                folds=int(args.folds),
                enable_bit_flip=bool(enable_flip),
                seed=int(args.seed),
            )
            diagnostics[name] = diag
        summary = summarize_bits(bits, target_bits)
        rows.append(
            {
                "variant": name,
                "n": summary["n"],
                "acc": summary["acc"],
                "ber": summary["ber"],
                "emr": summary["emr"],
            }
        )

    # Conventional post-hoc and clean-vocoder washout proxy for the same table.
    for label, key in [
        ("clean DiffWave re-vocoding proxy decoded by Ours extractor", "Baseline_windowed"),
        ("post-hoc audio decoded by Ours extractor", "Post-hoc_windowed"),
    ]:
        bits = [bits_from_prob_vector(p, thresholds=0.5, bit_flip=None) for p in probs[key]]
        summary = summarize_bits(bits, target_bits)
        rows.append(
            {
                "variant": label,
                "n": summary["n"],
                "acc": summary["acc"],
                "ber": summary["ber"],
                "emr": summary["emr"],
            }
        )

    return rows, diagnostics


def run_traceability(
    probs: Dict[str, List[Optional[np.ndarray]]],
    args: argparse.Namespace,
    target_bits: str,
) -> List[Dict[str, object]]:
    target_len = len(target_bits)
    ours_matrix = prob_matrix(probs["Ours_windowed"], target_len)
    ours_bits, _, _ = run_crossfit_calibration(
        prob_matrix=ours_matrix,
        target_bits=target_bits,
        folds=int(args.folds),
        enable_bit_flip=True,
        seed=int(args.seed),
    )
    baseline_bits = [bits_from_prob_vector(p, thresholds=0.5, bit_flip=None) for p in probs["Baseline_windowed"]]
    posthoc_bits = [bits_from_prob_vector(p, thresholds=0.5, bit_flip=None) for p in probs["Post-hoc_windowed"]]

    registry = {target_bits: "registered_source"}
    unknown_bits = "0" * target_len if target_bits != ("0" * target_len) else "1" * target_len
    registry_with_decoy = {target_bits: "registered_source", unknown_bits: "decoy_source"}

    def rates(bits_seq: Sequence[Optional[str]], expected_known: bool) -> Tuple[float, float, float]:
        labels = [bits_to_label(normalize_bits(b or "", target_len), registry_with_decoy) for b in bits_seq]
        source_hits = [1.0 if label == "registered_source" else 0.0 for label in labels]
        unknowns = [1.0 if label == "unknown" else 0.0 for label in labels]
        if expected_known:
            return float(np.mean(source_hits)), 1.0 - float(np.mean(source_hits)), float(np.mean(unknowns))
        return float(np.mean(source_hits)), float(np.mean(source_hits)), float(np.mean(unknowns))

    rows: List[Dict[str, object]] = []
    for condition, bits_seq, expected_known in [
        ("registered Ours audio", ours_bits, True),
        ("clean-vocoder copy-synthesis proxy", baseline_bits, False),
        ("post-hoc LSB/spread baseline audio", posthoc_bits, False),
    ]:
        hit, false_rate, unknown_rate = rates(bits_seq, expected_known)
        summary = summarize_bits(bits_seq, target_bits)
        rows.append(
            {
                "condition": condition,
                "n": float(len(bits_seq)),
                "id_attribution_or_false_accept": hit,
                "error_or_false_accept": false_rate,
                "unknown_rate": unknown_rate,
                "bit_acc_to_registered_id": summary["acc"],
                "emr_to_registered_id": summary["emr"],
            }
        )
    return rows


def bootstrap_ci(values: np.ndarray, iters: int, seed: int) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    means = np.empty((iters,), dtype=np.float64)
    n = int(values.size)
    for i in range(iters):
        sample = values[rng.randint(0, n, size=n)]
        means[i] = float(np.mean(sample))
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stats_ready_report(args: argparse.Namespace, out_dir: Path) -> List[Dict[str, object]]:
    src = Path(args.main_utterance_csv)
    if not src.exists():
        return []
    df = pd.read_csv(src)
    metrics = ["wer", "stoi", "pesq", "mcd", "ssim", "wm_acc", "wm_ber", "wm_emr"]
    rows: List[Dict[str, object]] = []
    for system in SYSTEMS_FOR_STATS:
        sys_df = df[df["system"] == system]
        for metric in metrics:
            if metric not in sys_df.columns:
                continue
            mean, lo, hi = bootstrap_ci(sys_df[metric].to_numpy(), int(args.bootstrap_iters), int(args.seed))
            rows.append(
                {
                    "system": system,
                    "metric": metric,
                    "n": float(len(sys_df)),
                    "mean": mean,
                    "bootstrap_ci95_low": lo,
                    "bootstrap_ci95_high": hi,
                    "utterance_std": float(np.nanstd(sys_df[metric].to_numpy(dtype=np.float64), ddof=1)),
                }
            )

    seed_dirs = sorted(Path("results").glob("main_eval*seed*"))
    seed_summary = []
    for d in seed_dirs:
        p = d / "system_summary.csv"
        if p.exists():
            seed_summary.append({"result_dir": str(d), "system_summary": str(p)})
    (out_dir / "available_seed_runs.json").write_text(
        json.dumps(seed_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows


def attack_report(args: argparse.Namespace) -> List[Dict[str, object]]:
    p = Path(args.attack_summary_csv)
    if not p.exists():
        return []
    df = pd.read_csv(p)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "attack": str(row["attack"]),
                "system": str(row["system"]),
                "acc": float(row["wm_acc"]),
                "ber": float(row["wm_ber"]),
                "emr": float(row["wm_emr"]),
            }
        )
    return rows


def check_strong_baselines() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    wavmark_spec = importlib.util.find_spec("wavmark")
    rows.append(
        {
            "baseline": "WavMark",
            "python_package_available": bool(wavmark_spec),
            "model_status": "not checked" if wavmark_spec else "package missing",
            "notes": "Needed for reviewer-requested strong multi-bit watermark baseline.",
        }
    )

    audioseal_spec = importlib.util.find_spec("audioseal")
    status = "package missing"
    notes = "AudioSeal can be reported as detection/provenance baseline if model weights are available."
    if audioseal_spec:
        try:
            from audioseal import AudioSeal  # type: ignore

            # Do not force download here; local cards point to Hugging Face checkpoints.
            status = "package available; weights may require Hugging Face download"
            notes = "Local model cards exist for audioseal_wm_16bits and audioseal_detector_16bits."
            _ = AudioSeal
        except Exception as exc:
            status = f"import failed: {exc}"
    rows.append(
        {
            "baseline": "AudioSeal",
            "python_package_available": bool(audioseal_spec),
            "model_status": status,
            "notes": notes,
        }
    )
    return rows


def main() -> None:
    args = parse_args()
    target_bits = normalize_bits(args.target_bits, len("".join(ch for ch in str(args.target_bits) if ch in "01")))
    if not target_bits:
        raise ValueError("target_bits must contain at least one binary digit.")
    args.target_bits = target_bits

    out_dir = ensure_dir(args.output_dir)
    manifest = load_manifest(Path(args.manifest_csv), args.max_samples)

    extractor = load_ours_extractor(args.ours_extractor_ckpt, args.device, target_bits_len=len(target_bits))
    extractor.eval()

    probs, raw_rows, timing_summary = collect_probs(manifest, extractor, args, len(target_bits))
    write_csv(out_dir / "raw_extractor_predictions.csv", raw_rows)
    (out_dir / "extractor_timing_seconds.json").write_text(
        json.dumps(timing_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ablation_rows, ablation_diag = run_decoding_ablation(probs, args, target_bits)
    write_csv(out_dir / "decoding_ablation.csv", ablation_rows)
    (out_dir / "decoding_ablation.md").write_text(
        markdown_table(ablation_rows, ["variant", "n", "acc", "ber", "emr"]) + "\n", encoding="utf-8"
    )
    (out_dir / "decoding_ablation_diagnostics.json").write_text(
        json.dumps(ablation_diag, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    trace_rows = run_traceability(probs, args, target_bits)
    write_csv(out_dir / "traceability_verification.csv", trace_rows)
    (out_dir / "traceability_verification.md").write_text(
        markdown_table(
            trace_rows,
            [
                "condition",
                "n",
                "id_attribution_or_false_accept",
                "error_or_false_accept",
                "unknown_rate",
                "bit_acc_to_registered_id",
                "emr_to_registered_id",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    stats_rows = stats_ready_report(args, out_dir)
    write_csv(out_dir / "bootstrap_metric_ci.csv", stats_rows)

    attack_rows = attack_report(args)
    write_csv(out_dir / "attack_summary_for_revision.csv", attack_rows)

    baseline_rows = check_strong_baselines()
    write_csv(out_dir / "strong_baseline_availability.csv", baseline_rows)
    (out_dir / "strong_baseline_availability.md").write_text(
        markdown_table(baseline_rows, ["baseline", "python_package_available", "model_status", "notes"]) + "\n",
        encoding="utf-8",
    )

    print(f"[Saved] Revision experiment outputs to {out_dir}")
    print("\n=== Decoding ablation ===")
    print(markdown_table(ablation_rows, ["variant", "n", "acc", "ber", "emr"]))
    print("\n=== Traceability verification ===")
    print(
        markdown_table(
            trace_rows,
            [
                "condition",
                "n",
                "id_attribution_or_false_accept",
                "error_or_false_accept",
                "unknown_rate",
                "bit_acc_to_registered_id",
                "emr_to_registered_id",
            ],
        )
    )


if __name__ == "__main__":
    main()
