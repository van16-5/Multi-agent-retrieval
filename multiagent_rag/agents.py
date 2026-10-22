from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from . import prompts as P
from .config import Config
from .llm import as_list, parse_json
from .metrics import answers_compatible
from .retrievers.base import Evidence

SOURCES = ("hybrid", "wikipedia", "graph")
PLACEHOLDER_RE = re.compile(r"#(\d+)")


def format_evidence(evidence: List[Evidence], max_chars: int) -> str:
    if not evidence:
        return "(no evidence retrieved)"
    lines = []
    for ev in evidence:
        text = ev.text if len(ev.text) <= max_chars else ev.text[:max_chars].rsplit(" ", 1)[0] + " ..."
        if ev.source == "graph" and ev.meta.get("path"):
            lines.append(f"[{ev.eid}] (graph path: {ev.meta['path']}) {text}")
        else:
            lines.append(f"[{ev.eid}] ({ev.title}) {text}")
    return "\n".join(lines)


def _clean_answer(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        x = ", ".join(str(i) for i in x if i is not None)
    s = str(x).strip().strip('"').strip()
    if not s or s.lower() in {"null", "none", "unknown", "n/a", "not found", "insufficient evidence"}:
        return None
    return s


class PlannerAgent:
    def __init__(self, llm, cfg: Config):
        self.llm, self.cfg = llm, cfg

    def run(self, question: str) -> Dict[str, Any]:
        raw = self.llm.generate(
            P.PLANNER_SYSTEM.format(max_sub=self.cfg.max_subquestions),
            P.PLANNER_USER.format(question=question),
            self.cfg.planner_max_tokens,
        )
        obj = parse_json(raw) or {}
        subs = [str(s).strip() for s in as_list(obj.get("sub_questions")) if str(s).strip()]
        subs = subs[: self.cfg.max_subquestions]
        # Drop placeholders that point forward or to missing steps; they cannot be resolved.
        subs = [s for i, s in enumerate(subs) if all(1 <= int(n) <= i for n in PLACEHOLDER_RE.findall(s))]
        complexity = obj.get("complexity", "multi_hop" if len(subs) > 1 else "simple")
        if not subs:
            subs, complexity = [question], "simple"
        return {"complexity": complexity, "sub_questions": subs, "raw": raw}


class VerifierAgent:
    def __init__(self, llm, cfg: Config):
        self.llm, self.cfg = llm, cfg

    def run(
        self,
        sub_question: str,
        evidence: List[Evidence],
        conflict_reason: Optional[str] = None,
        verification_question: Optional[str] = None,
    ) -> Dict[str, Any]:
        note = ""
        if conflict_reason:
            note = P.VERIFIER_CONFLICT_NOTE.format(reason=conflict_reason, verification_question=verification_question)
        raw = self.llm.generate(
            P.VERIFIER_SYSTEM,
            P.VERIFIER_USER.format(
                sub_question=sub_question,
                conflict_note=note,
                evidence=format_evidence(evidence, self.cfg.max_passage_chars),
            ),
            self.cfg.verifier_max_tokens,
        )
        obj = parse_json(raw) or {}
        valid_ids = {ev.eid for ev in evidence}
        relevant = [str(i) for i in as_list(obj.get("relevant_ids")) if str(i) in valid_ids]
        src_raw = obj.get("source_answers") if isinstance(obj.get("source_answers"), dict) else {}
        source_answers = {s: _clean_answer(src_raw.get(s)) for s in SOURCES}
        answer = _clean_answer(obj.get("answer"))
        if answer is None:
            answer = next((a for a in source_answers.values() if a), None)

        llm_conflict = bool(obj.get("conflict")) and str(obj.get("conflict")).lower() != "false"
        rule_conflict = self.detect_conflict(source_answers)
        return {
            "answer": answer,
            "relevant_ids": relevant,
            "source_answers": source_answers,
            "conflict": llm_conflict or rule_conflict,
            "conflict_reason": str(obj.get("conflict_reason") or "") or (
                "sources disagree: " + "; ".join(f"{k}={v}" for k, v in source_answers.items() if v)
                if rule_conflict else ""
            ),
            "raw": raw,
        }

    @staticmethod
    def detect_conflict(source_answers: Dict[str, Optional[str]]) -> bool:
        """Deterministic check on top of the LLM's judgement: any pair of non-null, incompatible answers."""
        vals = [v for v in source_answers.values() if v]
        return any(not answers_compatible(a, b) for i, a in enumerate(vals) for b in vals[i + 1:])


class ConflictAgent:
    def __init__(self, llm, cfg: Config):
        self.llm, self.cfg = llm, cfg

    def run(self, question: str, sub_question: str, verdict: Dict[str, Any], evidence: List[Evidence]) -> Dict[str, Any]:
        key_ids = set(verdict.get("relevant_ids") or [])
        key_ev = [ev for ev in evidence if ev.eid in key_ids] or evidence[:6]
        raw = self.llm.generate(
            P.CONFLICT_SYSTEM,
            P.CONFLICT_USER.format(
                question=question,
                sub_question=sub_question,
                source_answers={k: v for k, v in verdict["source_answers"].items() if v},
                reason=verdict.get("conflict_reason", ""),
                evidence=format_evidence(key_ev, self.cfg.max_passage_chars),
            ),
            self.cfg.conflict_max_tokens,
        )
        obj = parse_json(raw) or {}
        vq = str(obj.get("verification_question") or "").strip() or sub_question
        return {"verification_question": vq, "candidates": as_list(obj.get("candidates")), "raw": raw}


class SynthesisAgent:
    def __init__(self, llm, cfg: Config):
        self.llm, self.cfg = llm, cfg

    def run(self, question: str, steps: List[Dict[str, Any]], evidence: List[Evidence]) -> Dict[str, Any]:
        chain = "\n".join(
            f"{i}. {st['resolved_question']} -> {st['answer'] or 'unknown'}"
            + (" (conflict unresolved)" if st.get("conflict_unresolved") else "")
            for i, st in enumerate(steps, 1)
        )
        raw = self.llm.generate(
            P.SYNTH_SYSTEM,
            P.SYNTH_USER.format(
                question=question, chain=chain, evidence=format_evidence(evidence, self.cfg.max_passage_chars)
            ),
            self.cfg.synth_max_tokens,
        )
        obj = parse_json(raw) or {}
        answer = _clean_answer(obj.get("answer"))
        if answer is None:
            answer = steps[-1]["answer"] if steps and steps[-1]["answer"] else (raw.strip().split("\n")[0][:100])
        return {"answer": answer, "reasoning": str(obj.get("reasoning") or ""), "raw": raw}
