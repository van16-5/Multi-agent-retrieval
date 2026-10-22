# Multi-Agent Retrieval for Multi-Hop Question Answering

A multi-agent retrieval-augmented generation (RAG) system for multi-hop questions on **2WikiMultiHopQA**.
A planner decomposes the question; three complementary retrievers (local hybrid search, live Wikipedia,
and a knowledge graph) gather evidence for each sub-question; a verifier checks relevance, entailment, and
contradictions; a conflict agent runs **one** focused re-retrieval when sources disagree; and a synthesis
agent produces a grounded answer with a reasoning path, evidence, and a confidence score.
All agents share one small open model, **Qwen3-4B-Instruct-2507 in 4-bit**, so the whole system runs on a free Colab T4.

```
                            QUESTION
                               │
                     ┌─────────▼────────┐
                     │  PLANNER AGENT   │  complexity + decomposition (0–3 sub-questions, #k placeholders)
                     └─────────┬────────┘
                   sub-questions, answered in order (#1 resolved before #2 is retrieved)
            ┌─────────────────┼─────────────────┐
    HYBRID RETRIEVER    EXTERNAL RETRIEVER   GRAPH RETRIEVER
    BM25 + FAISS (RRF)   Wikipedia API       NetworkX entity graph, ≤2 hops
            └─────────────────┼─────────────────┘
                        evidence pool  (H1..H5, W1..W5, G1..G5)
                     ┌────────▼────────┐
                     │ VERIFIER AGENT  │  relevance, per-source answers, contradiction
                     └────────┬────────┘
                   conflict? (LLM judgement OR deterministic answer-disagreement check)
                     NO /           \ YES (max 1 retry per question)
                       │      ┌──────▼───────┐
                       │      │CONFLICT AGENT│ focused verification question
                       │      └──────┬───────┘
                       │     re-run all 3 retrievers, merge, re-verify
                       └──────┬──────┘
                     ┌────────▼────────┐
                     │ SYNTHESIS AGENT │  grounded short answer
                     └────────┬────────┘
            answer + reasoning path + evidence + confidence
```

## How the NLP methods are used

| Component | Method |
|---|---|
| Planner | LLM few-shot question decomposition into atomic sub-questions with `#k` references (compositional reasoning) |
| Hybrid retriever | BM25 lexical ranking + dense bi-encoder retrieval (`bge-small-en-v1.5`, FAISS inner-product index) fused with Reciprocal Rank Fusion |
| External retriever | MediaWiki search for the full query and for each named-entity span, then sentence chunking and dense re-ranking |
| Graph retriever | spaCy NER + relation lexicon to build a page→entity graph from the corpus; entity linking of question mentions; ≤2-hop traversal scored by semantic similarity, relation match, and hop distance |
| Verifier | LLM natural-language-inference-style judgement (relevance / entailment / contradiction) plus a deterministic answer-compatibility check (normalised containment / token F1) |
| Conflict agent | LLM query reformulation targeted at the disagreement, followed by one re-retrieval |
| Synthesis | Grounded generation constrained to verified evidence; confidence from cross-source agreement and conflict status |

## Setup

### Colab (recommended)
1. Copy this folder to Google Drive (for example `MyDrive/Multi-agent Retrieval`).
2. Open `notebooks/MultiAgent_RAG_Colab.ipynb` in Colab and choose **Runtime → Change runtime type → T4 GPU**.
3. Run the cells in order. Indexes, the Wikipedia cache, and results are written to `cache/` and `results/` on Drive.

### Command line (any CUDA machine)
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python scripts/demo.py --dev-index 0                              # one question with the full agent trace
python scripts/run_eval.py --split dev  --mode full               # 100 dev questions
python scripts/run_eval.py --split eval --mode closed_book        # baselines
python scripts/run_eval.py --split eval --mode single_rag
python scripts/run_eval.py --split eval --mode full               # 500 eval questions
python scripts/run_eval.py --split dev  --mode full --no-graph --tag no_graph   # ablation
python scripts/summarize.py results/eval_closed_book.jsonl results/eval_single_rag.jsonl results/eval_full.jsonl
```
Runs are **resumable**: each finished question is appended to `results/<split>_<mode>[_tag].jsonl`, and a rerun skips questions that are already done.

`tests/smoke_test.py` runs the real data and retrievers with a scripted mock LLM, so the orchestration logic can be checked without a GPU.

## Experimental protocol

| Setting | Value |
|---|---|
| Dataset | `framolfese/2WikiMultihopQA`, validation split (the official test split has no answers) |
| Development / evaluation | 100 / 500 questions, disjoint, sampled with seed 42 |
| Local corpus | Union of context paragraphs (gold + distractors) of the dev and eval questions, plus 1,000 extra questions' paragraphs as distractors (about 16k paragraphs). Retrievers never see which paragraph belongs to which question. |
| Generator | Qwen3-4B-Instruct-2507, 4-bit NF4, greedy decoding (temperature 0), 200–384 max new tokens per agent |
| Dense / lexical / fused top-k | 5 / 5 / 5 (20 candidates from each retriever before fusion, RRF k = 60) |
| Graph | NetworkX, max depth 2, top 5 edge sentences |
| Wikipedia | top 3 pages, top 5 chunks |
| Conflict retry | at most 1 per question (no agent loops) |

**Systems compared:** `closed_book` (LLM only), `single_rag` (hybrid retrieval with the full question and one LLM call), and `full` (the multi-agent system). Ablations switch off one source or the conflict agent.

**Metrics:**
- Answer quality: EM, F1 (official 2Wiki/HotpotQA normalisation) and Cover (the gold answer appears in the prediction).
- Retrieval: supporting-page recall, overall and per source.
- Agent behaviour: conflict rate, retry rate, EM on retried questions, average LLM calls and latency.
- Confidence calibration: EM per confidence bucket.

All of these are reported overall and per question type (comparison, inference, compositional, bridge_comparison).

## Design notes and limitations
- **No label leakage.** The dataset's gold Wikidata `evidences` triples and supporting-fact labels are never used at inference time; they are only used for evaluation. The knowledge graph is extracted automatically from raw paragraph text.
- **Live Wikipedia differs from the 2020 dataset snapshot.** Titles get renamed (for example "Held einer Nacht" became "Hero for a Night") and facts get updated. This is a realistic source of cross-source conflict that the conflict agent has to resolve. Responses are cached in `cache/wikipedia_cache.json` so reruns are reproducible.
- **Sub-question cap.** With at most 3 sub-questions, bridge-comparison questions use a combined final sub-question ("What are the nationalities of #1 and #2?"). The pipeline then issues one retrieval query per entity so both entities are covered.
- **Confidence** is heuristic (cross-source agreement, conflict status, and the weakest step in the chain), not a learned probability. Check the calibration table before trusting it.
- **Runtime.** On a T4 the full system takes roughly 20–30 s per question (about 5 LLM calls). Plan for several Colab sessions for the 500-question evaluation; resumable runs make this safe.

## Project structure
```
multiagent_rag/
  config.py          all hyper-parameters and ablation switches
  data.py            2Wiki loading, dev/eval sampling, corpus construction
  llm.py             shared 4-bit Qwen generator, MockLLM, robust JSON parsing
  prompts.py         prompts for every agent and baseline
  agents.py          Planner, Verifier, Conflict, Synthesis agents
  pipeline.py        orchestration, baselines, per-question evaluation
  metrics.py         EM / F1 / Cover, retrieval recall, aggregation, calibration
  trace.py           readable agent trace for demos
  builder.py         loads components once and builds systems / ablations
  retrievers/
    base.py          Evidence type, bge embedder, tokenisation, entity-span extraction
    hybrid.py        BM25 + FAISS + Reciprocal Rank Fusion
    wikipedia.py     MediaWiki search + dense chunk re-ranking (cached)
    graph.py         NER/relation-based entity graph + multi-hop traversal
scripts/             run_eval.py, summarize.py, demo.py
notebooks/           Colab notebook (setup → retrieval analysis → dev → ablations → eval → charts)
tests/smoke_test.py  GPU-free end-to-end test
```
