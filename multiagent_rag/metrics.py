from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List


def normalize_answer(s: str) -> str:
    """SQuAD/HotpotQA-style normalisation used by the official 2Wiki evaluation."""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    # yes/no answers must match exactly, as in the official script
    if p in {"yes", "no"} or g in {"yes", "no"}:
        return float(p == g)
    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pt), same / len(gt)
    return 2 * precision * recall / (precision + recall)


def cover_match(pred: str, gold: str) -> float:
    """Gold answer appears inside the prediction (lenient accuracy common for LLM outputs)."""
    g = normalize_answer(gold)
    return float(bool(g) and g in normalize_answer(pred))


def answers_compatible(a: str, b: str) -> bool:
    na, nb = normalize_answer(a), normalize_answer(b)
    if not na or not nb:
        return True
    if na == nb or na in nb or nb in na:
        return True
    return f1_score(a, b) >= 0.5


def title_recall(gold_titles: Iterable[str], retrieved_titles: Iterable[str]) -> float:
    gold = {normalize_answer(t) for t in gold_titles}
    if not gold:
        return 0.0
    got = {normalize_answer(t) for t in retrieved_titles}
    return len(gold & got) / len(gold)


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-question records (see pipeline output) into overall and per-type metrics."""
    if not records:
        return {}

    def agg(rows):
        n = len(rows)
        out = {
            "n": n,
            "EM": 100 * sum(r["em"] for r in rows) / n,
            "F1": 100 * sum(r["f1"] for r in rows) / n,
            "Cover": 100 * sum(r["cover"] for r in rows) / n,
            "avg_llm_calls": sum(r.get("llm_calls", 0) for r in rows) / n,
            "avg_latency_s": sum(r.get("latency_s", 0) for r in rows) / n,
        }
        if any("retrieval_recall" in r for r in rows):
            for src in ("all", "hybrid", "wikipedia", "graph"):
                vals = [r["retrieval_recall"].get(src) for r in rows if r.get("retrieval_recall", {}).get(src) is not None]
                if vals:
                    out[f"recall_{src}"] = 100 * sum(vals) / len(vals)
        if any("conflicts_detected" in r for r in rows):
            out["conflict_rate"] = 100 * sum(1 for r in rows if r.get("conflicts_detected", 0) > 0) / n
            out["retry_rate"] = 100 * sum(1 for r in rows if r.get("retries", 0) > 0) / n
            retried = [r for r in rows if r.get("retries", 0) > 0]
            if retried:
                out["EM_when_retried"] = 100 * sum(r["em"] for r in retried) / len(retried)
            out["avg_confidence"] = sum(r.get("confidence", 0) for r in rows) / n
            out["avg_subquestions"] = sum(len(r.get("sub_questions", [])) for r in rows) / n
        return out

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_type[r["type"]].append(r)
    return {"overall": agg(records), "by_type": {t: agg(rs) for t, rs in sorted(by_type.items())}}


def calibration_table(records: List[Dict[str, Any]], bins=(0.0, 0.4, 0.6, 0.8, 1.01)) -> List[Dict[str, Any]]:
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [r for r in records if lo <= r.get("confidence", 0) < hi]
        if sel:
            rows.append(
                {
                    "confidence": f"[{lo:.1f}, {min(hi, 1.0):.1f}]",
                    "n": len(sel),
                    "EM": 100 * sum(r["em"] for r in sel) / len(sel),
                    "F1": 100 * sum(r["f1"] for r in sel) / len(sel),
                }
            )
    return rows
