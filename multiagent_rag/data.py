from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .config import Config


@dataclass
class QAExample:
    id: str
    question: str
    answer: str
    type: str
    supporting_titles: List[str]
    supporting_facts: List[Tuple[str, int]]
    evidences: List[List[str]] = field(default_factory=list)  # gold Wikidata triples, eval only


@dataclass
class Document:
    doc_id: int
    title: str
    sentences: List[str]

    @property
    def text(self) -> str:
        return " ".join(s.strip() for s in self.sentences).strip()


def _context_pairs(context) -> List[Tuple[str, List[str]]]:
    """Normalise the two context layouts found in 2Wiki mirrors."""
    if isinstance(context, dict):
        return list(zip(context["title"], context.get("sentences") or context.get("content")))
    pairs = []
    for item in context:
        if isinstance(item, dict):
            pairs.append((item["title"], item.get("sentences") or item.get("content")))
        else:
            pairs.append((item[0], item[1]))
    return pairs


def _supporting_pairs(sf) -> List[Tuple[str, int]]:
    if isinstance(sf, dict):
        return list(zip(sf["title"], [int(i) for i in sf["sent_id"]]))
    return [(x["title"], int(x["sent_id"])) if isinstance(x, dict) else (x[0], int(x[1])) for x in sf]


def _to_example(row) -> QAExample:
    sf = _supporting_pairs(row["supporting_facts"])
    titles = list(dict.fromkeys(t for t, _ in sf))
    return QAExample(
        id=row.get("id") or row.get("_id"),
        question=row["question"],
        answer=row["answer"],
        type=row["type"],
        supporting_titles=titles,
        supporting_facts=sf,
        evidences=[list(e) if not isinstance(e, dict) else [e["fact"], e["relation"], e["entity"]]
                   for e in (row.get("evidences") or [])],
    )


class TwoWikiData:
    """Samples dev/eval questions and builds the shared retrieval corpus.

    The corpus is the union of the context paragraphs (gold + distractors) of
    all sampled questions plus extra distractor questions. Retrievers never see
    which paragraph belongs to which question.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dev: List[QAExample] = []
        self.eval: List[QAExample] = []
        self.documents: List[Document] = []

    def load(self) -> "TwoWikiData":
        os.makedirs(self.cfg.cache_dir, exist_ok=True)
        cache_path = os.path.join(
            self.cfg.cache_dir,
            f"data_{self.cfg.seed}_{self.cfg.n_dev}_{self.cfg.n_eval}_{self.cfg.n_distractor_questions}.json",
        )
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                blob = json.load(f)
            self.dev = [QAExample(**x) for x in blob["dev"]]
            self.eval = [QAExample(**x) for x in blob["eval"]]
            self.documents = [Document(**d) for d in blob["documents"]]
            return self

        from datasets import load_dataset

        ds = load_dataset(self.cfg.dataset_name, split=self.cfg.dataset_split)
        idx = list(range(len(ds)))
        random.Random(self.cfg.seed).shuffle(idx)
        n_dev, n_eval, n_dis = self.cfg.n_dev, self.cfg.n_eval, self.cfg.n_distractor_questions
        dev_idx = idx[:n_dev]
        eval_idx = idx[n_dev:n_dev + n_eval]
        dis_idx = idx[n_dev + n_eval:n_dev + n_eval + n_dis]

        self.dev = [_to_example(ds[i]) for i in dev_idx]
        self.eval = [_to_example(ds[i]) for i in eval_idx]

        seen: Dict[str, int] = {}
        docs: List[Document] = []
        for i in dev_idx + eval_idx + dis_idx:
            for title, sents in _context_pairs(ds[i]["context"]):
                if title in seen:
                    continue
                seen[title] = len(docs)
                docs.append(Document(doc_id=len(docs), title=title, sentences=list(sents)))
        self.documents = docs

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dev": [x.__dict__ for x in self.dev],
                    "eval": [x.__dict__ for x in self.eval],
                    "documents": [d.__dict__ for d in self.documents],
                },
                f,
            )
        return self

    def split(self, name: str) -> List[QAExample]:
        return {"dev": self.dev, "eval": self.eval}[name]
