from __future__ import annotations

import hashlib
import os
from typing import Dict, List

import numpy as np

from ..config import Config
from ..data import Document
from .base import Embedder, Evidence, tokenize


def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60) -> List[tuple]:
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class HybridRetriever:
    """Local corpus retriever: BM25 (lexical) + FAISS over bge embeddings (dense), fused with RRF."""

    name = "hybrid"

    def __init__(self, cfg: Config, documents: List[Document], embedder: Embedder):
        import faiss
        from rank_bm25 import BM25Okapi

        self.cfg = cfg
        self.docs = documents
        self.embedder = embedder
        self.passages = [f"{d.title}. {d.text}" for d in documents]

        self.bm25 = BM25Okapi([tokenize(p) for p in self.passages])

        emb = self._load_or_embed()
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def _load_or_embed(self) -> np.ndarray:
        os.makedirs(self.cfg.cache_dir, exist_ok=True)
        h = hashlib.md5(("\n".join(self.passages) + self.cfg.embed_model).encode("utf-8")).hexdigest()[:12]
        path = os.path.join(self.cfg.cache_dir, f"doc_emb_{h}.npy")
        if os.path.exists(path):
            return np.load(path)
        emb = self.embedder.encode_passages(self.passages)
        np.save(path, emb)
        return emb

    def bm25_search(self, query: str, k: int) -> List[int]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [int(i) for i in top if scores[i] > 0]

    def dense_search(self, query: str, k: int) -> List[int]:
        q = self.embedder.encode_queries([query])
        _, idx = self.index.search(q, k)
        return [int(i) for i in idx[0] if i >= 0]

    def retrieve(self, query: str) -> List[Evidence]:
        depth = max(self.cfg.candidate_depth, self.cfg.bm25_top_k, self.cfg.dense_top_k)
        bm25_ids = self.bm25_search(query, depth)
        dense_ids = self.dense_search(query, depth)
        fused = reciprocal_rank_fusion([bm25_ids, dense_ids], k=self.cfg.rrf_k)[: self.cfg.fused_top_k]
        bm25_rank = {d: r + 1 for r, d in enumerate(bm25_ids)}
        dense_rank = {d: r + 1 for r, d in enumerate(dense_ids)}
        out = []
        for doc_id, score in fused:
            d = self.docs[doc_id]
            out.append(
                Evidence(
                    source=self.name,
                    title=d.title,
                    text=d.text,
                    score=float(score),
                    meta={"doc_id": doc_id, "bm25_rank": bm25_rank.get(doc_id), "dense_rank": dense_rank.get(doc_id)},
                )
            )
        return out
