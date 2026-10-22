"""Run a system over the dev or eval split, appending one JSON line per question (resumable).

Examples:
    python scripts/run_eval.py --split dev --mode full
    python scripts/run_eval.py --split eval --mode single_rag
    python scripts/run_eval.py --split eval --mode full --no-graph --tag no_graph
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiagent_rag.builder import MODES, Components  # noqa: E402
from multiagent_rag.config import Config  # noqa: E402
from multiagent_rag.metrics import summarize  # noqa: E402
from multiagent_rag.pipeline import evaluate_example, serialize  # noqa: E402


def run(components: Components, split: str, mode: str, cfg: Config, out_path: str, limit=None):
    examples = components.data.split(split)[:limit]
    done = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["id"]] = r
    todo = [ex for ex in examples if ex.id not in done]
    print(f"[run] {mode} on {split}: {len(done)} done, {len(todo)} to go -> {out_path}")
    system = components.system(mode, cfg)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, ex in enumerate(todo, 1):
            try:
                rec = evaluate_example(system, ex)
            except Exception as e:
                print(f"[run] {ex.id} failed: {e}")
                continue
            f.write(json.dumps(serialize(rec), ensure_ascii=False) + "\n")
            f.flush()
            done[ex.id] = rec
            if i % 10 == 0 or i == len(todo):
                em = 100 * sum(r["em"] for r in done.values()) / len(done)
                rate = (time.time() - t0) / i
                print(f"  {i}/{len(todo)}  EM so far {em:.1f}  ({rate:.1f}s/question)")
    records = [done[ex.id] for ex in examples if ex.id in done]
    return summarize(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "eval"], default="dev")
    ap.add_argument("--mode", choices=MODES, default="full")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="", help="suffix for the results file, e.g. an ablation name")
    ap.add_argument("--no-hybrid", action="store_true")
    ap.add_argument("--no-wikipedia", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--no-conflict", action="store_true")
    ap.add_argument("--n-distractors", type=int, default=None)
    args = ap.parse_args()

    cfg = Config(
        use_hybrid=not args.no_hybrid,
        use_wikipedia=not args.no_wikipedia,
        use_graph=not args.no_graph,
        use_conflict_agent=not args.no_conflict,
    )
    if args.n_distractors is not None:
        cfg.n_distractor_questions = args.n_distractors
    name = f"{args.split}_{args.mode}" + (f"_{args.tag}" if args.tag else "")
    out_path = os.path.join(cfg.results_dir, name + ".jsonl")
    summary = run(Components(cfg), args.split, args.mode, cfg, out_path, args.limit)
    print(json.dumps(summary, indent=2))
    with open(os.path.join(cfg.results_dir, name + "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
