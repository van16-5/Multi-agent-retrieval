from __future__ import annotations

import hashlib
import os
import pickle
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..config import Config
from ..data import Document
from .base import STOPWORDS, Embedder, Evidence, normalize_key

# Lightweight relation lexicon covering the Wikidata relations used by 2WikiMultiHopQA.
# Used both to label graph edges (from passage sentences) and to detect the relation a question asks about.
RELATION_PATTERNS: Dict[str, str] = {
    "director": r"\bdirect(?:ed|or|s|ing)\b",
    "producer": r"\bproduc(?:ed|er|ers)\b",
    "composer": r"\bcompos(?:ed|er)\b|\bmusic (?:by|was)\b|\bscore\b",
    "performer": r"\bperform(?:ed|er)\b|\bsung\b|\bsinger\b|\bstarr(?:ing|ed)\b|\brecorded by\b",
    "writer": r"\bwrit(?:ten|er)\b|\bscreenplay\b|\bauthor\b|\bnovel by\b",
    "date_of_birth": r"\bborn\b|\bbirth\b|\([^()]*\d{4}\s*[–-][^()]*\d{4}\s*\)",
    "place_of_birth": r"\bborn in\b|\bbirthplace\b|\bnative of\b|\bborn at\b",
    "date_of_death": r"\bdied\b|\bdeath\b|\bpassed away\b|\([^()]*\d{4}\s*[–-][^()]*\d{4}\s*\)",
    "place_of_death": r"\bdied (?:in|at)\b|\bplace of death\b",
    "burial": r"\bburi(?:ed|al)\b|\binterred\b|\bcemetery\b",
    "cause_of_death": r"\bcause of death\b|\bdied (?:of|from)\b|\bcancer\b|\bheart attack\b|\bdisease\b|\bsuicide\b",
    "spouse": r"\bmarri(?:ed|age)\b|\bwife\b|\bhusband\b|\bspouse\b|\bwed\b",
    "father": r"\bfather\b|\bson of\b|\bdaughter of\b",
    "mother": r"\bmother\b|\bson of\b|\bdaughter of\b",
    "child": r"\bson\b|\bdaughter\b|\bchild(?:ren)?\b",
    "sibling": r"\bbrother\b|\bsister\b|\bsibling\b",
    "nationality": r"\bnationality\b|\bcitizen\b|\bcountry\b|\bfrom the same country\b",
    "education": r"\bstudied\b|\bgraduat(?:ed|e)\b|\beducated\b|\buniversity\b|\bcollege\b|\bschool\b|\battend(?:ed)?\b",
    "award": r"\baward(?:ed|s)?\b|\bprize\b|\bwon\b|\bhonou?r\b",
    "founder": r"\bfound(?:ed|er)\b|\bestablished\b",
    "publication_date": r"\breleased\b|\bpublished\b|\bpremiered\b|\bcame out\b|\bfirst\b|\bearlier\b|\bolder\b|\bnewer\b|\byear\b",
    "employer": r"\bwork(?:ed|s)? (?:for|at)\b|\bemploy(?:ed|er)\b",
    "occupation": r"\boccupation\b|\bprofession\b|\bwas an?\b",
    "location": r"\blocated\b|\bheadquarter(?:ed|s)\b|\bbased in\b",
}
_REL_RE = {k: re.compile(v, re.I) for k, v in RELATION_PATTERNS.items()}

# Questions phrased about age/order ("who died first", "who was born later") map to date relations.
_QUESTION_HINTS = [
    (re.compile(r"\b(older|younger|born first|born later|born earlier)\b", re.I), "date_of_birth"),
    (re.compile(r"\b(died first|died later|died earlier|lived longer|die first)\b", re.I), "date_of_death"),
    (re.compile(r"\b(where|place)\b.*\bborn\b|\bborn\b.*\bwhere\b", re.I), "place_of_birth"),
    (re.compile(r"\b(where|place)\b.*\bdie[d]?\b", re.I), "place_of_death"),
    (re.compile(r"\bsame (country|nationality)\b|\bnationality\b|\bwhich country\b", re.I), "nationality"),
]

_PERSON = {"PERSON", "PAGE", "MISC"}
_PLACE = {"GPE", "LOC", "NORP", "FAC", "PAGE", "MISC"}
RELATION_TARGET_TYPES: Dict[str, Set[str]] = {
    **{r: _PERSON for r in ("director", "producer", "composer", "performer", "writer", "spouse", "father",
                            "mother", "child", "sibling", "founder")},
    **{r: {"DATE"} for r in ("date_of_birth", "date_of_death", "publication_date")},
    **{r: _PLACE for r in ("place_of_birth", "place_of_death", "burial", "nationality", "location")},
    "education": {"ORG", "PAGE", "MISC"},
    "employer": {"ORG", "PAGE", "MISC"},
    "award": {"WORK_OF_ART", "EVENT", "ORG", "PAGE", "MISC"},
}

KEEP_LABELS = {"PERSON", "ORG", "GPE", "LOC", "NORP", "DATE", "WORK_OF_ART", "FAC", "EVENT", "LANGUAGE"}
_CAPS_RE = re.compile(r"\b(?:[A-Z][\w'.-]+(?:\s+(?:of|de|la|von|van|the|and)?\s*[A-Z][\w'.-]+)*)\b")
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_YEAR_RE = re.compile(rf"\b(?:(?:\d{{1,2}} (?:{_MONTHS})|(?:{_MONTHS}) \d{{1,2}},?) )?\d{{4}}\b")


def detect_relations(text: str) -> Set[str]:
    return {name for name, rx in _REL_RE.items() if rx.search(text)}


def question_relations(question: str) -> Set[str]:
    rels = detect_relations(question)
    for rx, rel in _QUESTION_HINTS:
        if rx.search(question):
            rels.add(rel)
    return rels


def strip_parenthetical(title: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()


class GraphRetriever:
    """Graph RAG over an entity graph built from the corpus with NER + a relation lexicon.

    Nodes: one per document title (entity page) plus literal entities (dates, nationalities, people
    without a page). Edges: "page --relation--> entity" extracted from each sentence, carrying the
    sentence as provenance. Retrieval links question entities to nodes, then walks up to
    `graph_max_depth` hops, scoring each traversed edge sentence by semantic similarity to the
    question, relation match, and hop distance.
    """

    name = "graph"

    def __init__(self, cfg: Config, documents: List[Document], embedder: Embedder):
        import networkx as nx

        self.cfg = cfg
        self.docs = documents
        self.embedder = embedder
        self.nx = nx
        self.alias_to_nodes: Dict[str, Set[str]] = defaultdict(set)
        for d in documents:
            node = self._page_node(d.title)
            self.alias_to_nodes[normalize_key(d.title)].add(node)
            short = normalize_key(strip_parenthetical(d.title))
            if short and len(short) > 2 and short not in STOPWORDS:
                self.alias_to_nodes[short].add(node)

        self.graph, self.sentences = self._load_or_build()
        self.sent_emb = self._load_or_embed_sentences()
        self.title_list = [d.title for d in documents]
        self.title_emb: Optional[np.ndarray] = None

    @staticmethod
    def _page_node(title: str) -> str:
        return "page::" + title

    def _cache_key(self) -> str:
        blob = "\n".join(d.title + "|" + d.text for d in self.docs) + str(self.cfg.use_spacy)
        return hashlib.md5(blob.encode("utf-8")).hexdigest()[:12]

    # ------------------------------------------------------------------ build
    def _load_or_build(self):
        os.makedirs(self.cfg.cache_dir, exist_ok=True)
        path = os.path.join(self.cfg.cache_dir, f"graph_{self._cache_key()}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        result = self._build()
        with open(path, "wb") as f:
            pickle.dump(result, f)
        return result

    def _entity_extractor(self):
        if self.cfg.use_spacy:
            try:
                import spacy

                nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])

                def extract_batch(texts: List[str]):
                    for doc in nlp.pipe(texts, batch_size=256):
                        yield [(e.text, e.label_) for e in doc.ents if e.label_ in KEEP_LABELS]

                return extract_batch
            except Exception as e:
                print(f"[graph] spaCy unavailable ({e}); falling back to regex entity extraction")

        def regex_batch(texts: List[str]):
            for t in texts:
                ents = [(m.group(0), "DATE") for m in _YEAR_RE.finditer(t)]
                ents += [(m.group(0), "MISC") for m in _CAPS_RE.finditer(t)]
                yield ents

        return regex_batch

    def _build(self):
        G = self.nx.MultiGraph()
        for d in self.docs:
            G.add_node(self._page_node(d.title), kind="page", label=d.title)

        sentences: List[Tuple[str, int, str]] = []  # (doc title, sentence index, text)
        for d in self.docs:
            for i, s in enumerate(d.sentences):
                if s.strip():
                    sentences.append((d.title, i, s.strip()))

        extract = self._entity_extractor()
        print(f"[graph] extracting entities from {len(sentences)} sentences ...")
        for sid, ((title, _, text), ents) in enumerate(zip(sentences, extract([s[2] for s in sentences]))):
            src = self._page_node(title)
            rels = sorted(detect_relations(text))
            src_key = " " + normalize_key(title) + " "
            seen: Set[str] = set()
            for ent_text, label in ents:
                key = normalize_key(ent_text)
                # Skip mentions of the page's own subject, including fragments like "Nacht" in "Held einer Nacht".
                if not key or f" {key} " in src_key or key in STOPWORDS or len(key) < 2:
                    continue
                targets = self.alias_to_nodes.get(key)
                if targets:
                    dst_nodes = list(targets)[:3]
                else:
                    dst = "ent::" + key
                    if dst not in G:
                        G.add_node(dst, kind="literal", label=ent_text.strip(), ner=label)
                    dst_nodes = [dst]
                for dst in dst_nodes:
                    if dst == src or dst in seen:
                        continue
                    seen.add(dst)
                    G.add_edge(src, dst, sid=sid, relations=rels, ner=label)
        print(f"[graph] built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G, sentences

    def _load_or_embed_sentences(self) -> np.ndarray:
        path = os.path.join(self.cfg.cache_dir, f"graph_sent_emb_{self._cache_key()}.npy")
        if os.path.exists(path):
            return np.load(path)
        emb = self.embedder.encode_passages([f"{t}. {s}" for t, _, s in self.sentences])
        np.save(path, emb)
        return emb

    # --------------------------------------------------------------- retrieve
    def link_entities(self, question: str) -> List[str]:
        q = " " + normalize_key(question) + " "
        matches = []
        for alias, nodes in self.alias_to_nodes.items():
            if f" {alias} " in q:
                matches.append((alias, nodes))
        # Prefer longer (more specific) aliases and drop aliases nested inside a longer match.
        matches.sort(key=lambda x: -len(x[0]))
        chosen: List[str] = []
        kept_aliases: List[str] = []
        for alias, nodes in matches:
            if any(f" {alias} " in f" {k} " for k in kept_aliases):
                continue
            kept_aliases.append(alias)
            for n in sorted(nodes):
                if n not in chosen:
                    chosen.append(n)
        if not chosen:
            chosen = self._dense_link(question, k=2)
        return chosen[: self.cfg.graph_max_seed_entities]

    def _dense_link(self, question: str, k: int) -> List[str]:
        if self.title_emb is None:
            self.title_emb = self.embedder.encode_passages(self.title_list)
        q = self.embedder.encode_queries([question])[0]
        top = np.argsort(-(self.title_emb @ q))[:k]
        return [self._page_node(self.title_list[i]) for i in top]

    def _expandable(self, node: str) -> bool:
        data = self.graph.nodes[node]
        if data.get("kind") == "page":
            return True
        # Literal people without their own page can still bridge two documents, but hubs like
        # "American" or "1970" would explode the search, so they are treated as leaves.
        return data.get("ner") == "PERSON" and self.graph.degree(node) <= 20

    def retrieve(self, query: str) -> List[Evidence]:
        seeds = self.link_entities(query)
        if not seeds:
            return []
        q_rels = question_relations(query)
        q_emb = self.embedder.encode_queries([query])[0]

        # BFS keeping the best path to every edge sentence.
        best: Dict[int, Tuple[float, List[str], Set[str], int]] = {}
        frontier = [(s, [self.graph.nodes[s]["label"]]) for s in seeds]
        visited = set(seeds)
        for hop in range(1, self.cfg.graph_max_depth + 1):
            nxt = []
            for node, path in frontier:
                if not self._expandable(node):
                    continue
                for _, nbr, data in self.graph.edges(node, data=True):
                    sid = data["sid"]
                    rels = set(data["relations"])
                    sim = float(self.sent_emb[sid] @ q_emb)
                    score = sim + (0.15 if rels & q_rels else 0.0) - 0.05 * (hop - 1)
                    # Several entities share one sentence; prefer the one whose type fits the asked relation
                    # (e.g. a PERSON for "director", a DATE for "date_of_birth") so the shown path is meaningful.
                    nbr_type = "PAGE" if self.graph.nodes[nbr].get("kind") == "page" else data.get("ner", "")
                    if any(nbr_type in RELATION_TARGET_TYPES.get(r, ()) for r in (rels & q_rels or rels)):
                        score += 0.02
                    nbr_label = self.graph.nodes[nbr]["label"]
                    rel_label = "/".join(sorted(rels & q_rels) or sorted(rels)[:1]) or "mentions"
                    new_path = path + [f"--{rel_label}-->", nbr_label]
                    if sid not in best or score > best[sid][0]:
                        best[sid] = (score, new_path, rels, hop)
                    if nbr not in visited:
                        visited.add(nbr)
                        nxt.append((nbr, new_path))
            frontier = nxt

        ranked = sorted(best.items(), key=lambda x: -x[1][0])[: self.cfg.graph_top_k]
        out = []
        for sid, (score, path, rels, hop) in ranked:
            title, sent_idx, text = self.sentences[sid]
            out.append(
                Evidence(
                    source=self.name,
                    title=title,
                    text=text,
                    score=score,
                    meta={
                        "path": " ".join(path),
                        "hops": hop,
                        "relations": sorted(rels),
                        "sent_idx": sent_idx,
                        "seeds": [self.graph.nodes[s]["label"] for s in seeds],
                    },
                )
            )
        return out
