from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List


@dataclass
class Config:
    # Dataset
    dataset_name: str = "framolfese/2WikiMultihopQA"
    dataset_split: str = "validation"  # the official test split has no answers
    seed: int = 42
    n_dev: int = 100
    n_eval: int = 500
    # Extra questions whose context paragraphs are added to the corpus purely as
    # distractors, so retrieval is harder than the per-question distractor setting.
    n_distractor_questions: int = 1000

    # Generator
    llm_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    load_in_4bit: bool = True
    temperature: float = 0.0
    max_new_tokens: int = 384
    planner_max_tokens: int = 200
    verifier_max_tokens: int = 320
    conflict_max_tokens: int = 160
    synth_max_tokens: int = 320

    # Dense retrieval
    embed_model: str = "BAAI/bge-small-en-v1.5"
    bge_query_instruction: str = "Represent this sentence for searching relevant passages: "
    embed_batch_size: int = 64

    # Hybrid retrieval
    dense_top_k: int = 5
    bm25_top_k: int = 5
    candidate_depth: int = 20  # candidates taken from each retriever before fusion
    rrf_k: int = 60
    fused_top_k: int = 5

    # Graph retrieval
    graph_max_depth: int = 2
    graph_top_k: int = 5
    graph_max_seed_entities: int = 4
    use_spacy: bool = True

    # External search
    wiki_top_pages: int = 3
    wiki_top_chunks: int = 5
    wiki_sentences_per_chunk: int = 3
    wiki_timeout: float = 10.0
    wiki_user_agent: str = "MultiAgentRAG-StudentProject/1.0 (educational use)"

    # Agents
    max_subquestions: int = 3
    max_conflict_retries: int = 1  # per question; hard cap, no agent loops
    max_passage_chars: int = 600

    # Ablation switches
    use_hybrid: bool = True
    use_wikipedia: bool = True
    use_graph: bool = True
    use_conflict_agent: bool = True

    # Paths
    cache_dir: str = "cache"
    results_dir: str = "results"

    @property
    def enabled_sources(self) -> List[str]:
        return [
            name
            for name, on in (
                ("hybrid", self.use_hybrid),
                ("wikipedia", self.use_wikipedia),
                ("graph", self.use_graph),
            )
            if on
        ]

    def to_dict(self):
        return asdict(self)
