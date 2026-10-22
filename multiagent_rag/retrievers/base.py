from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np

from ..config import Config

STOPWORDS = set(
    """a an the of in on at to for from by with and or is are was were be been being who whom whose what
    which when where why how did do does that this these those as it its his her their he she they them
    film movie song album book than same both either more most first earlier later born died""".split()
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str, drop_stopwords: bool = True) -> List[str]:
    toks = _WORD_RE.findall(text.lower())
    return [t for t in toks if t not in STOPWORDS] if drop_stopwords else toks


def normalize_key(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


_QUESTION_WORDS = {"Which", "Who", "Whom", "Whose", "What", "When", "Where", "Why", "How", "Are", "Is", "Was",
                   "Were", "Did", "Do", "Does", "In", "The", "A", "An"}
# Lowercase "and" is deliberately not a connector: "X and Y" in questions almost always joins two entities.
_CONNECTORS = {"of", "the", "de", "la", "le", "du", "von", "van", "der", "di", "da", "del", "y", "in", "on",
               "for", "to", "a", "an", "&"}


def extract_entity_spans(text: str) -> List[str]:
    """Capitalised name spans such as "Held Einer Nacht" or "Move (1970 Film)" from a question."""
    tokens = re.findall(r"\([^)]*\)|[^\s(]+", text)
    spans, cur = [], []

    def flush():
        while cur and cur[-1].lower() in _CONNECTORS:
            cur.pop()
        if cur:
            spans.append(" ".join(cur))
        cur.clear()

    for i, tok in enumerate(tokens):
        word = tok.strip(",?!.;:\"'")
        if tok.startswith("(") and cur:
            cur.append(tok)
            flush()
        elif word[:1].isupper() or (word[:1].isdigit() and cur):
            if not cur and word in _QUESTION_WORDS and (i == 0 or word not in {"The", "A", "An", "In"}):
                continue
            cur.append(word)
            if tok != word and tok.rstrip("\"'")[-1:] in ",?!.;:":
                flush()
        elif cur and word.lower() in _CONNECTORS:
            cur.append(word)
        else:
            flush()
    flush()
    return list(dict.fromkeys(s for s in spans if len(s) > 2))


@dataclass
class Evidence:
    source: str  # "hybrid" | "wikipedia" | "graph"
    title: str
    text: str
    score: float
    eid: str = ""  # short id shown to agents, e.g. H1 / W2 / G3
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Embedder:
    """bge-small-en-v1.5 wrapper shared by all retrievers."""

    def __init__(self, cfg: Config):
        from sentence_transformers import SentenceTransformer

        self.cfg = cfg
        self.model = SentenceTransformer(cfg.embed_model)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype="float32")
        return self.model.encode(
            list(texts),
            batch_size=self.cfg.embed_batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 2000,
            convert_to_numpy=True,
        ).astype("float32")

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_passages([self.cfg.bge_query_instruction + t for t in texts])


def assign_ids(evidence: List[Evidence], prefix: str) -> List[Evidence]:
    for i, ev in enumerate(evidence, 1):
        ev.eid = f"{prefix}{i}"
    return evidence
