from __future__ import annotations

import json
import os
import re
import threading
from typing import Dict, List

import numpy as np

from ..config import Config
from .base import STOPWORDS, Embedder, Evidence, extract_entity_spans

API_URL = "https://en.wikipedia.org/w/api.php"
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


class WikipediaRetriever:
    """External search: live Wikipedia (MediaWiki API) -> top pages -> intro chunks ranked by bge similarity.

    Responses are cached on disk so evaluation runs are reproducible and polite to the API.
    """

    name = "wikipedia"

    def __init__(self, cfg: Config, embedder: Embedder):
        import requests

        self.cfg = cfg
        self.embedder = embedder
        self.session = requests.Session()
        self.session.headers["User-Agent"] = cfg.wiki_user_agent
        os.makedirs(cfg.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cfg.cache_dir, "wikipedia_cache.json")
        self._lock = threading.Lock()
        self.cache: Dict[str, List[Dict[str, str]]] = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, encoding="utf-8") as f:
                self.cache = json.load(f)

    def _save_cache(self):
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.cache, f)
        os.replace(tmp, self.cache_path)

    def _api_search(self, query: str) -> List[Dict[str, str]]:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": self.cfg.wiki_top_pages,
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "exlimit": self.cfg.wiki_top_pages,
            "redirects": 1,
        }
        r = self.session.get(API_URL, params=params, timeout=self.cfg.wiki_timeout)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        ranked = sorted(pages.values(), key=lambda p: p.get("index", 99))
        return [{"title": p["title"], "extract": p.get("extract", "")} for p in ranked if p.get("extract")]

    def candidate_pages(self, query: str) -> List[Dict[str, str]]:
        """Union of pages found for each named entity in the query and for the full query.

        Searching a whole multi-entity question tends to return loosely related pages, so each entity
        span is also searched on its own; results are interleaved and capped at `wiki_top_pages`.
        """
        lists = [self.search_pages(span)[:2] for span in extract_entity_spans(query)[:3]]
        lists.append(self.search_pages(query))
        out, seen = [], set()
        for rank in range(max(len(x) for x in lists)):
            for res in lists:
                if rank < len(res) and res[rank]["title"] not in seen:
                    seen.add(res[rank]["title"])
                    out.append(res[rank])
        return out[: self.cfg.wiki_top_pages]

    def search_pages(self, query: str) -> List[Dict[str, str]]:
        if query in self.cache:
            return self.cache[query]
        pages: List[Dict[str, str]] = []
        try:
            pages = self._api_search(query)
            if not pages:
                keywords = " ".join(w for w in re.findall(r"[\w()'-]+", query) if w.lower() not in STOPWORDS)
                if keywords and keywords != query:
                    pages = self._api_search(keywords)
        except Exception as e:  # network failures must not crash an evaluation run
            print(f"[wikipedia] search failed for {query!r}: {e}")
            return []
        with self._lock:
            self.cache[query] = pages
            self._save_cache()
        return pages

    def _chunks(self, page: Dict[str, str]) -> List[str]:
        sents = [s.strip() for s in _SENT_SPLIT.split(page["extract"].replace("\n", " ")) if s.strip()]
        n = self.cfg.wiki_sentences_per_chunk
        return [" ".join(sents[i:i + n]) for i in range(0, len(sents), n)]

    def retrieve(self, query: str) -> List[Evidence]:
        pages = self.candidate_pages(query)
        cands = []
        for rank, page in enumerate(pages):
            for ci, chunk in enumerate(self._chunks(page)):
                cands.append((page["title"], chunk, rank, ci))
        if not cands:
            return []
        q = self.embedder.encode_queries([query])[0]
        p = self.embedder.encode_passages([f"{t}. {c}" for t, c, _, _ in cands])
        sims = p @ q
        # Small prior for higher-ranked pages and for the opening chunk, which holds most infobox-like facts.
        scores = sims + np.array([0.03 * (len(pages) - r) + (0.05 if ci == 0 else 0.0) for _, _, r, ci in cands])
        order = np.argsort(-scores)[: self.cfg.wiki_top_chunks]
        return [
            Evidence(
                source=self.name,
                title=cands[i][0],
                text=cands[i][1],
                score=float(scores[i]),
                meta={"page_rank": cands[i][2] + 1, "chunk": cands[i][3]},
            )
            for i in order
        ]
