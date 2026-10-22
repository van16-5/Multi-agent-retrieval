from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Dict, List, Optional

from . import prompts as P
from .agents import PLACEHOLDER_RE, ConflictAgent, PlannerAgent, SynthesisAgent, VerifierAgent, format_evidence
from .config import Config
from .data import QAExample
from .llm import parse_json
from .metrics import answers_compatible, cover_match, exact_match, f1_score, title_recall
from .retrievers.base import Evidence

PREFIX = {"hybrid": "H", "wikipedia": "W", "graph": "G"}


class MultiAgentPipeline:
    """Planner -> (hybrid | wikipedia | graph) retrieval -> verifier -> [conflict agent + 1 retry] -> synthesis.

    One shared LLM backs every agent. Sub-questions are answered in order so that later ones can use
    earlier answers through #k placeholders (needed for compositional / bridge questions).
    """

    def __init__(self, cfg: Config, llm, retrievers: Dict[str, Any]):
        self.cfg = cfg
        self.llm = llm
        self.retrievers = {k: v for k, v in retrievers.items() if k in cfg.enabled_sources and v is not None}
        self.planner = PlannerAgent(llm, cfg)
        self.verifier = VerifierAgent(llm, cfg)
        self.conflict = ConflictAgent(llm, cfg)
        self.synthesizer = SynthesisAgent(llm, cfg)

    # ------------------------------------------------------------- retrieval
    def retrieve(self, queries: List[str]) -> List[Evidence]:
        """Run every enabled retriever on every query; interleave per source so each query is represented."""
        # Each retriever already returns its own top-k; with several queries allow two extra passages
        # per source so every entity gets coverage without blowing up the verifier prompt.
        cap = None if len(queries) == 1 else self.cfg.fused_top_k + 2
        pooled: List[Evidence] = []
        for name, retriever in self.retrievers.items():
            per_query = []
            for q in queries:
                try:
                    per_query.append(retriever.retrieve(q))
                except Exception as e:  # one failing source must not sink the whole question
                    print(f"[{name}] retrieval failed for {q!r}: {e}")
                    per_query.append([])
            merged, seen = [], set()
            for rank in range(max((len(r) for r in per_query), default=0)):
                for res in per_query:
                    if rank < len(res):
                        key = (res[rank].title, res[rank].text)
                        if key not in seen:
                            seen.add(key)
                            merged.append(res[rank])
            pooled.extend(merged[:cap])
        return self._relabel(pooled)

    @staticmethod
    def _relabel(evidence: List[Evidence]) -> List[Evidence]:
        counters: Dict[str, int] = {}
        out = []
        for ev in evidence:
            counters[ev.source] = counters.get(ev.source, 0) + 1
            out.append(replace(ev, eid=f"{PREFIX[ev.source]}{counters[ev.source]}"))
        return out

    @staticmethod
    def _merge(a: List[Evidence], b: List[Evidence]) -> List[Evidence]:
        seen = {(e.source, e.title, e.text) for e in a}
        merged = list(a) + [e for e in b if (e.source, e.title, e.text) not in seen]
        merged.sort(key=lambda e: list(PREFIX).index(e.source))
        return MultiAgentPipeline._relabel(merged)

    # ---------------------------------------------------------- sub-questions
    @staticmethod
    def resolve(sub_question: str, steps: List[Dict[str, Any]]) -> str:
        def sub(m):
            n = int(m.group(1))
            if 1 <= n <= len(steps):
                return steps[n - 1]["answer"] or steps[n - 1]["resolved_question"]
            return m.group(0)

        return PLACEHOLDER_RE.sub(sub, sub_question)

    def build_queries(self, sub_question: str, steps: List[Dict[str, Any]]) -> List[str]:
        resolved = self.resolve(sub_question, steps)
        refs = list(dict.fromkeys(int(n) for n in PLACEHOLDER_RE.findall(sub_question)))
        if len(refs) <= 1:
            return [resolved]
        # "When did #1 and #2 die?" -> one focused query per entity, plus the combined query.
        template = " ".join(PLACEHOLDER_RE.sub(" ", sub_question).split())
        queries = [resolved]
        for n in refs:
            ans = steps[n - 1]["answer"] if n <= len(steps) else None
            if ans:
                queries.append(f"{ans}: {template}")
        return queries

    def _step_confidence(self, step: Dict[str, Any]) -> float:
        if not step["answer"]:
            return 0.1
        answered = [v for v in step["source_answers"].values() if v]
        agree = sum(1 for v in answered if answers_compatible(v, step["answer"]))
        conf = 0.4 + 0.6 * agree / max(1, len(self.retrievers))
        if step["conflict_unresolved"]:
            conf *= 0.6
        elif step["retried"]:
            conf *= 0.9
        return round(min(conf, 1.0), 3)

    # ------------------------------------------------------------------- main
    def answer(self, question: str) -> Dict[str, Any]:
        t0, calls0 = time.time(), self.llm.calls
        plan = self.planner.run(question)
        steps: List[Dict[str, Any]] = []
        retries_left = self.cfg.max_conflict_retries if self.cfg.use_conflict_agent else 0
        conflicts_detected = 0

        for sq in plan["sub_questions"]:
            resolved = self.resolve(sq, steps)
            queries = self.build_queries(sq, steps)
            evidence = self.retrieve(queries)
            verdict = self.verifier.run(resolved, evidence)
            step = {
                "sub_question": sq,
                "resolved_question": resolved,
                "queries": queries,
                "evidence": evidence,
                "initial_verdict": {k: v for k, v in verdict.items() if k != "raw"},
                "retried": False,
                "verification_question": None,
            }
            if verdict["conflict"]:
                conflicts_detected += 1
                if retries_left > 0:
                    retries_left -= 1
                    c = self.conflict.run(question, resolved, verdict, evidence)
                    extra = self.retrieve([c["verification_question"]])
                    evidence = self._merge(evidence, extra)
                    verdict = self.verifier.run(
                        resolved, evidence, verdict["conflict_reason"], c["verification_question"]
                    )
                    step.update(
                        evidence=evidence,
                        retried=True,
                        verification_question=c["verification_question"],
                        conflict_candidates=c["candidates"],
                    )
            step.update(
                answer=verdict["answer"],
                relevant_ids=verdict["relevant_ids"],
                source_answers=verdict["source_answers"],
                conflict_unresolved=bool(verdict["conflict"]),
                conflict_reason=verdict["conflict_reason"],
            )
            step["confidence"] = self._step_confidence(step)
            steps.append(step)

        final_evidence = []
        for i, st in enumerate(steps, 1):
            rel = [e for e in st["evidence"] if e.eid in set(st["relevant_ids"])]
            if not rel:
                rel = [e for e in st["evidence"] if e.eid[1:] in {"1", "2"}]
            final_evidence += [replace(e, eid=f"{i}.{e.eid}") for e in rel]
        synth = self.synthesizer.run(question, steps, final_evidence)

        step_conf = [s["confidence"] for s in steps] or [0.1]
        confidence = round(0.5 * min(step_conf) + 0.5 * sum(step_conf) / len(step_conf), 3)
        reasoning_path = [
            f"{s['resolved_question']} -> {s['answer']}" + (" [verified after conflict]" if s["retried"] else "")
            for s in steps
        ]
        return {
            "question": question,
            "answer": synth["answer"],
            "reasoning": synth["reasoning"],
            "reasoning_path": reasoning_path,
            "confidence": confidence,
            "complexity": plan["complexity"],
            "sub_questions": plan["sub_questions"],
            "steps": steps,
            "final_evidence": final_evidence,
            "conflicts_detected": conflicts_detected,
            "retries": sum(1 for s in steps if s["retried"]),
            "llm_calls": self.llm.calls - calls0,
            "latency_s": round(time.time() - t0, 2),
        }


class ClosedBookBaseline:
    """LLM only, no retrieval."""

    def __init__(self, cfg: Config, llm):
        self.cfg, self.llm = cfg, llm

    def answer(self, question: str) -> Dict[str, Any]:
        t0, c0 = time.time(), self.llm.calls
        raw = self.llm.generate(P.CLOSED_BOOK_SYSTEM, f"Question: {question}", 64)
        obj = parse_json(raw) or {}
        ans = str(obj.get("answer") or raw.split("\n")[0]).strip()
        return {"question": question, "answer": ans, "llm_calls": self.llm.calls - c0,
                "latency_s": round(time.time() - t0, 2), "steps": [], "final_evidence": []}


class SingleRAGBaseline:
    """Standard single-shot RAG: hybrid retrieval on the full question, one LLM call."""

    def __init__(self, cfg: Config, llm, hybrid):
        self.cfg, self.llm, self.hybrid = cfg, llm, hybrid

    def answer(self, question: str) -> Dict[str, Any]:
        t0, c0 = time.time(), self.llm.calls
        evidence = MultiAgentPipeline._relabel(self.hybrid.retrieve(question))
        raw = self.llm.generate(
            P.SINGLE_RAG_SYSTEM,
            P.SINGLE_RAG_USER.format(question=question, evidence=format_evidence(evidence, self.cfg.max_passage_chars)),
            self.cfg.synth_max_tokens,
        )
        obj = parse_json(raw) or {}
        ans = str(obj.get("answer") or raw.split("\n")[0]).strip()
        return {"question": question, "answer": ans, "reasoning": str(obj.get("reasoning") or ""),
                "llm_calls": self.llm.calls - c0, "latency_s": round(time.time() - t0, 2),
                "steps": [{"evidence": evidence}], "final_evidence": evidence}


def _titles(evidence: List[Evidence], source: Optional[str] = None) -> List[str]:
    return [e.title for e in evidence if source is None or e.source == source]


def evaluate_example(system, ex: QAExample) -> Dict[str, Any]:
    out = system.answer(ex.question)
    all_ev = [e for st in out.get("steps", []) for e in st.get("evidence", [])]
    recall = {"all": title_recall(ex.supporting_titles, _titles(all_ev))}
    for src in PREFIX:
        if any(e.source == src for e in all_ev):
            recall[src] = title_recall(ex.supporting_titles, _titles(all_ev, src))
    record = {
        "id": ex.id,
        "type": ex.type,
        "gold": ex.answer,
        "pred": out["answer"],
        "em": exact_match(out["answer"], ex.answer),
        "f1": f1_score(out["answer"], ex.answer),
        "cover": cover_match(out["answer"], ex.answer),
        "retrieval_recall": recall,
        "supporting_titles": ex.supporting_titles,
    }
    record.update(out)
    return record


def serialize(record: Dict[str, Any]) -> Dict[str, Any]:
    """Make a pipeline record JSON-safe (Evidence objects -> dicts)."""

    def conv(x):
        if isinstance(x, Evidence):
            return x.to_dict()
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        return x

    return conv(record)
