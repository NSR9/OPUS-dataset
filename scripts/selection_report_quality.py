#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize OPUS selection report quality")
    p.add_argument("--report-json", type=str, default="selection_report.json")
    p.add_argument("--top-k", type=int, default=5)
    return p.parse_args()


def safe_mean(xs: list[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def safe_median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return float(statistics.median(xs))


def main() -> None:
    args = parse_args()
    report = json.loads(Path(args.report_json).read_text(encoding="utf-8"))
    rows = report.get("records", [])
    if not rows:
        raise RuntimeError("No records in report")

    selected = [r for r in rows if bool(r.get("selected", False))]
    removed = [r for r in rows if not bool(r.get("selected", False))]

    sel_j = [float(r.get("golden_match_jaccard", 0.0)) for r in selected]
    rem_j = [float(r.get("golden_match_jaccard", 0.0)) for r in removed]

    sel_tags = Counter(str(r.get("golden_match_tag", "")) for r in selected)
    rem_tags = Counter(str(r.get("golden_match_tag", "")) for r in removed)

    print(f"Report: {args.report_json}")
    print(f"Rows: {len(rows)} | Selected: {len(selected)} | Removed: {len(removed)}")
    print(
        "Golden Jaccard: "
        f"selected_mean={safe_mean(sel_j):.6f} removed_mean={safe_mean(rem_j):.6f} "
        f"delta={safe_mean(sel_j) - safe_mean(rem_j):.6f}"
    )
    print(
        "Golden Jaccard median: "
        f"selected={safe_median(sel_j):.6f} removed={safe_median(rem_j):.6f}"
    )

    if "round_summaries" in report:
        print("\nPer-round deltas:")
        for rs in report["round_summaries"]:
            s = float(rs.get("selected_avg_golden_jaccard", 0.0))
            r = float(rs.get("removed_avg_golden_jaccard", 0.0))
            print(
                f"  round={int(rs.get('round', -1))} "
                f"selected_avg={s:.6f} removed_avg={r:.6f} delta={s-r:.6f} "
                f"capture_nonfinite={int(rs.get('capture_nonfinite_values', 0))} "
                f"feature_nonfinite={int(rs.get('feature_nonfinite_values', 0))}"
            )

    print("\nSelected tag distribution:")
    for tag, count in sel_tags.most_common(args.top_k):
        print(f"  {tag or '<empty>'}: {count}")

    print("\nRemoved tag distribution:")
    for tag, count in rem_tags.most_common(args.top_k):
        print(f"  {tag or '<empty>'}: {count}")

    ranked = sorted(rows, key=lambda x: float(x.get("round0_score", 0.0)), reverse=True)
    print("\nTop by round0_score:")
    for r in ranked[: max(1, int(args.top_k))]:
        snippet = str(r.get("text_snippet", "")).replace("\n", " ")
        print(
            f"  round={int(r.get('round', 0))} idx={int(r.get('candidate_index', -1))} "
            f"selected={bool(r.get('selected', False))} score={float(r.get('round0_score', 0.0)):.6e} "
            f"tag={r.get('golden_match_tag', '')}"
        )
        print(f"    {snippet[:180]}")


if __name__ == "__main__":
    main()
