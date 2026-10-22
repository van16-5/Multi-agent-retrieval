"""Answer one question and print the full agent trace (for the presentation).

    python scripts/demo.py --question "Who is the mother of the director of film Move (1970 Film)?"
    python scripts/demo.py --dev-index 3
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiagent_rag.builder import Components  # noqa: E402
from multiagent_rag.config import Config  # noqa: E402
from multiagent_rag.trace import print_trace  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default=None)
    ap.add_argument("--dev-index", type=int, default=0)
    args = ap.parse_args()

    comps = Components(Config())
    gold = None
    if args.question:
        q = args.question
    else:
        ex = comps.data.dev[args.dev_index]
        q, gold = ex.question, ex.answer
    print_trace(comps.system("full").answer(q), gold)


if __name__ == "__main__":
    main()
