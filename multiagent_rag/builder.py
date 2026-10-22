from __future__ import annotations

from typing import Any, Dict, Optional

from .config import Config
from .data import TwoWikiData
from .pipeline import ClosedBookBaseline, MultiAgentPipeline, SingleRAGBaseline

MODES = ("full", "single_rag", "closed_book")


class Components:
    """Loads data, embedder, retrievers and the LLM once so several modes/ablations can reuse them."""

    def __init__(self, cfg: Config, llm=None):
        self.cfg = cfg
        self.data = TwoWikiData(cfg).load()
        print(f"[data] dev={len(self.data.dev)} eval={len(self.data.eval)} corpus={len(self.data.documents)} docs")
        self._llm = llm
        self._embedder = None
        self._retrievers: Dict[str, Any] = {}

    @property
    def llm(self):
        if self._llm is None:
            from .llm import LLM

            self._llm = LLM(self.cfg)
        return self._llm

    @property
    def embedder(self):
        if self._embedder is None:
            from .retrievers import Embedder

            self._embedder = Embedder(self.cfg)
        return self._embedder

    def retriever(self, name: str):
        if name not in self._retrievers:
            from .retrievers import GraphRetriever, HybridRetriever, WikipediaRetriever

            if name == "hybrid":
                self._retrievers[name] = HybridRetriever(self.cfg, self.data.documents, self.embedder)
            elif name == "wikipedia":
                self._retrievers[name] = WikipediaRetriever(self.cfg, self.embedder)
            elif name == "graph":
                self._retrievers[name] = GraphRetriever(self.cfg, self.data.documents, self.embedder)
            else:
                raise ValueError(name)
        return self._retrievers[name]

    def system(self, mode: str = "full", cfg: Optional[Config] = None):
        """`cfg` may override ablation switches (use_graph=False, ...) while sharing loaded components."""
        cfg = cfg or self.cfg
        if mode == "closed_book":
            return ClosedBookBaseline(cfg, self.llm)
        if mode == "single_rag":
            return SingleRAGBaseline(cfg, self.llm, self.retriever("hybrid"))
        if mode == "full":
            return MultiAgentPipeline(cfg, self.llm, {s: self.retriever(s) for s in cfg.enabled_sources})
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
