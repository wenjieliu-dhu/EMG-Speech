#!/usr/bin/env python3
"""Aggregate system summaries across random seeds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


METRICS = ["wer", "stoi", "pesq", "mcd", "ssim", "wm_acc", "wm_ber", "wm_emr"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate evaluate_main_experiment outputs over seeds.")
    parser.add_argument("--summary_csvs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--system", type=str, default="Ours")
    parser.add_argument("--output_dir", type=str, default="results/seed_aggregate")
    return parser.parse_args()


def fmt_mean_std(mean: float, std: float, metric: str) -> str:
    if metric in {"wer", "wm_ber", "wm_emr"}:
        return f"{mean * 100:.3f} ± {std * 100:.3f}"
    if metric == "wm_acc":
        return f"{mean:.4f} ± {std:.4f}"
    return f"{mean:.3f} ± {std:.3f}"


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: Sequence[Dict[str, object]]) -> str:
    cols = list(rows[0].keys())
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if len(args.summary_csvs) != len(args.seeds):
        raise ValueError("--summary_csvs and --seeds must have the same length.")

    per_seed_rows: List[Dict[str, object]] = []
    for seed, csv_path in zip(args.seeds, args.summary_csvs):
        df = pd.read_csv(csv_path)
        row = df[df["system"].astype(str) == str(args.system)]
        if row.empty:
            raise ValueError(f"System {args.system!r} not found in {csv_path}")
        rec: Dict[str, object] = {"seed": seed, "system": args.system}
        for metric in METRICS:
            if metric in row.columns:
                rec[metric] = float(row.iloc[0][metric])
        per_seed_rows.append(rec)

    agg_rows: List[Dict[str, object]] = []
    for metric in METRICS:
        vals = np.asarray([float(r[metric]) for r in per_seed_rows if metric in r], dtype=np.float64)
        if vals.size == 0:
            continue
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0
        agg_rows.append(
            {
                "system": args.system,
                "metric": metric,
                "n_seeds": int(vals.size),
                "mean": mean,
                "std": std,
                "mean ± std": fmt_mean_std(mean, std, metric),
            }
        )

    out_dir = Path(args.output_dir)
    write_csv(out_dir / "per_seed_summary.csv", per_seed_rows)
    write_csv(out_dir / "mean_std_summary.csv", agg_rows)
    (out_dir / "mean_std_summary.md").write_text(markdown_table(agg_rows) + "\n", encoding="utf-8")
    print(f"[Saved] {out_dir / 'mean_std_summary.md'}")
    print(markdown_table(agg_rows))


if __name__ == "__main__":
    main()
