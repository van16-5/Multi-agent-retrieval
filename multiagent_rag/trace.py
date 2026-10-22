from __future__ import annotations

from typing import Any, Dict, Optional


def _get(ev, key):
    return ev.get(key) if isinstance(ev, dict) else getattr(ev, key)


def print_trace(result: Dict[str, Any], gold: Optional[str] = None, max_chars: int = 160) -> None:
    """Human-readable view of one pipeline run: plan, per-step evidence/verdicts, conflicts, final answer."""
    bar = "=" * 90
    print(bar)
    print(f"QUESTION   : {result['question']}")
    if gold is not None:
        print(f"GOLD       : {gold}")
    print(f"PLANNER    : {result.get('complexity')} -> {result.get('sub_questions')}")
    for i, st in enumerate(result.get("steps", []), 1):
        print("-" * 90)
        print(f"STEP {i}: {st.get('resolved_question', '')}")
        if len(st.get("queries", [])) > 1:
            print(f"  queries: {st['queries']}")
        relevant = set(st.get("relevant_ids", []))
        for ev in st.get("evidence", []):
            eid = _get(ev, "eid")
            mark = "*" if eid in relevant else " "
            meta = _get(ev, "meta") or {}
            head = meta.get("path") if _get(ev, "source") == "graph" and meta.get("path") else _get(ev, "title")
            text = _get(ev, "text").replace("\n", " ")
            print(f"  {mark}[{eid}] {head}: {text[:max_chars]}{'...' if len(text) > max_chars else ''}")
        init = st.get("initial_verdict", {})
        print(f"  source answers : {init.get('source_answers', st.get('source_answers'))}")
        if init.get("conflict"):
            print(f"  CONFLICT       : {init.get('conflict_reason')}")
            if st.get("retried"):
                print(f"  verification Q : {st.get('verification_question')}")
                print(f"  after retry    : {st.get('source_answers')}  unresolved={st.get('conflict_unresolved')}")
        print(f"  => answer: {st.get('answer')}   (step confidence {st.get('confidence')})")
    print("-" * 90)
    print(f"ANSWER     : {result['answer']}")
    print(f"REASONING  : {result.get('reasoning', '')}")
    print("PATH       : " + "  |  ".join(result.get("reasoning_path", [])))
    print(f"CONFIDENCE : {result.get('confidence')}   LLM calls: {result.get('llm_calls')}   "
          f"latency: {result.get('latency_s')}s")
    print(bar)
