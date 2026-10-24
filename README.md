# RAG Evaluation Lab — measuring retrieval and answers

A small science question-answering RAG system (30 passages written for this project) built so that it can be **measured**:

- **Retrieval:** Recall@1, Recall@3 and MRR for BM25, dense and hybrid retrievers.
- **Generation:** answer accuracy (expected keywords), **faithfulness** (a Gemini judge checks every fact against the passages), and **refusal** on questions the passages cannot answer.

Course: Advanced NLP (AML23702) · Global Academy of Technology

## How it works

```
Question ──> retriever (BM25 / dense / hybrid RRF) ──> top-3 passages
         ──> Gemini answers ONLY from passages, or says "I don't know"
         ──> Gemini judge: SUPPORTED / UNSUPPORTED
Test set (20 answerable + 4 unanswerable) ──> metrics ──> outputs/ + dashboard
```

## Files

| File | What it does |
|---|---|
| `rag_core.py` | Retrievers, answer prompt, judge prompt, scoring helpers |
| `evaluate.py` | Part A retrieval metrics, Part B generation metrics |
| `llm.py` | Calls Gemini (key from `.env`), retries on rate limits |
| `app.py` | Streamlit dashboard: results, live ask with judge, test set |
| `main.py` | Command-line ask |
| `data/passages.json` | 30 science passages (biology, physics, chemistry, environment) |
| `data/test_set.json` | 24 questions with gold passage and expected keywords |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then paste your Gemini key into .env
```

## Run

```bash
python evaluate.py --retrieval-only   # Part A only, no key needed
python evaluate.py                    # Part A + Part B (about 45 Gemini calls)
python evaluate.py --max-seconds 150  # same, but stops after 150 s; run again to continue
streamlit run app.py                  # dashboard
```

## Results (from `outputs/`)

Retrieval, 20 answerable questions:

| Retriever | Recall@1 | Recall@3 | MRR |
|---|---|---|---|
| BM25 | 0.85 | 0.90 | 0.891 |
| Dense | 1.00 | 1.00 | 1.000 |
| Hybrid (RRF) | 0.95 | 0.95 | 0.960 |

Generation with the hybrid retriever (top 3), gemini-3.5-flash-lite:

| Metric | Value |
|---|---|
| Answer accuracy (20 answerable) | 90% |
| Faithfulness (18 non-refused answers, judged) | 100% |
| Correct refusals (4 unanswerable) | 100% |
| Wrong refusals (20 answerable) | 10% (Q11, Q13) |

All results are saved in `outputs/`.
