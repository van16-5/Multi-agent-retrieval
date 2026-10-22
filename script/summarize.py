"""Compare result files side by side.

    python scripts/summarize.py results/eval_closed_book.jsonl results/eval_single_rag.jsonl results/eval_full.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiagent_rag.metrics import calibration_table, summarize  # noqa: E402


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt(v):
    return f"{v:.1f}" if isinstance(v, float) else str(v)


def table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=None, help="optional markdown report path")
    args = ap.parse_args()

    overall_rows, type_rows, report = [], [], []
    for path in args.files:
        name = os.path.basename(path).replace(".jsonl", "")
        recs = load(path)
        s = summarize(recs)
        overall_rows.append({"system": name, **s["overall"]})
        for t, m in s["by_type"].items():
            type_rows.append({"system": name, "type": t, **m})
        if any("confidence" in r for r in recs):
            report.append(f"\n### Confidence calibration: {name}\n")
            report.append(table(calibration_table(recs), ["confidence", "n", "EM", "F1"]))

    cols = ["system", "n", "EM", "F1", "Cover", "recall_all", "recall_hybrid", "recall_wikipedia", "recall_graph",
            "conflict_rate", "retry_rate", "EM_when_retried", "avg_confidence", "avg_llm_calls", "avg_latency_s"]
    cols = [c for c in cols if any(c in r for r in overall_rows)]
    text = "## Overall\n\n" + table(overall_rows, cols)
    text += "\n\n## By question type\n\n" + table(type_rows, ["system", "type", "n", "EM", "F1", "Cover"])
    text += "\n" + "\n".join(report)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
