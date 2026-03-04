#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List


def _load_jsonl(path: Path) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            step = int(obj["step"])
            rows[step] = obj
    return rows


def _series(rows: Dict[int, dict], steps: List[int], key: str) -> List[float]:
    out: List[float] = []
    for s in steps:
        val = float(rows[s].get(key, 0.0))
        out.append(val)
    return out


def _summary(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"sum": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0}
    sorted_vals = sorted(vals)
    p90_idx = min(len(sorted_vals) - 1, int(0.9 * (len(sorted_vals) - 1)))
    return {
        "sum": float(sum(vals)),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "p90": float(sorted_vals[p90_idx]),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper-style OPUS overhead report from end-to-end step timing logs.")
    p.add_argument("--baseline-log", required=True, type=str, help="JSONL from random baseline run")
    p.add_argument("--opus-log", required=True, type=str, help="JSONL from OPUS run")
    p.add_argument("--warmup-steps", type=int, default=20, help="Ignore steps < warmup_steps")
    p.add_argument("--output-json", type=str, default=None, help="Optional output summary JSON path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    baseline_rows = _load_jsonl(Path(args.baseline_log))
    opus_rows = _load_jsonl(Path(args.opus_log))

    baseline_steps = {s for s in baseline_rows.keys() if s >= int(args.warmup_steps)}
    opus_steps = {s for s in opus_rows.keys() if s >= int(args.warmup_steps)}
    common_steps = sorted(baseline_steps.intersection(opus_steps))
    if not common_steps:
        raise RuntimeError("No common steps found between baseline and OPUS logs after warmup filter.")

    base_step_total = _series(baseline_rows, common_steps, "step_total_s")
    opus_step_total = _series(opus_rows, common_steps, "step_total_s")
    base_summary = _summary(base_step_total)
    opus_summary = _summary(opus_step_total)

    base_total = max(base_summary["sum"], 1e-12)
    overhead_frac = (opus_summary["sum"] - base_summary["sum"]) / base_total

    # Extra attribution for OPUS run only.
    opus_scoring = _summary(_series(opus_rows, common_steps, "scoring_pass_s"))
    opus_features = _summary(_series(opus_rows, common_steps, "feature_build_s"))
    opus_selector = _summary(_series(opus_rows, common_steps, "selector_total_s"))
    opus_train = _summary(_series(opus_rows, common_steps, "train_pass_s"))

    out = {
        "steps_compared": int(len(common_steps)),
        "warmup_steps": int(args.warmup_steps),
        "baseline": {
            "step_total_s": base_summary,
            "selection_mode": str(baseline_rows[common_steps[-1]].get("selection_mode", "unknown")),
        },
        "opus": {
            "step_total_s": opus_summary,
            "selection_mode": str(opus_rows[common_steps[-1]].get("selection_mode", "unknown")),
            "scoring_pass_s": opus_scoring,
            "feature_build_s": opus_features,
            "selector_total_s": opus_selector,
            "train_pass_s": opus_train,
        },
        "paper_style_overhead_fraction": float(overhead_frac),
        "paper_style_overhead_percent": float(overhead_frac * 100.0),
    }

    print(json.dumps(out, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
