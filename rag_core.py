"""
rag_core.py - a small RAG system over science passages, built so that it can
be MEASURED.

Retrievers: BM25, dense (embeddings) and hybrid (Reciprocal Rank Fusion).
Generator : Gemini, answering only from the retrieved passages.
Judge     : Gemini again, checking whether an answer is supported by the
            passages (used to measure faithfulness).
"""

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from llm import generate

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60
TOP_K = 3
NOT_FOUND = "I don't know based on the given passages."

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "to", "in", "on",
    "for", "and", "or", "by", "with", "what", "which", "who", "how", "why",
    "does", "do", "can", "it", "its", "this", "that", "my", "me", "from", "at",
    "when", "there", "called", "much", "many",
}


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def load_json(name):
    with open(Path("data") / name, encoding="utf-8") as f:
        return json.load(f)


class Retriever:
    def __init__(self, passages, model_name=EMBED_MODEL):
        self.passages = passages
        docs = [f"{p['title']}. {p['text']}" for p in passages]
        self.bm25 = BM25Okapi([tokenize(d) for d in docs])
        self.model = SentenceTransformer(model_name)
        self.vectors = self.model.encode(docs, normalize_embeddings=True,
                                         show_progress_bar=False)

    def _bm25_order(self, question):
        return list(np.argsort(-self.bm25.get_scores(tokenize(question)), kind="stable"))

    def _dense_order(self, question):
        q = self.model.encode([question], normalize_embeddings=True)[0]
        return list(np.argsort(-(self.vectors @ q), kind="stable"))

    def search(self, question, k=TOP_K, mode="hybrid"):
        if mode == "bm25":
            order = self._bm25_order(question)
        elif mode == "dense":
            order = self._dense_order(question)
        else:
            bm25_rank = {int(i): r for r, i in enumerate(self._bm25_order(question), 1)}
            dense_rank = {int(i): r for r, i in enumerate(self._dense_order(question), 1)}
            fused = {i: 1 / (RRF_K + bm25_rank[i]) + 1 / (RRF_K + dense_rank[i])
                     for i in bm25_rank}
            order = sorted(fused, key=fused.get, reverse=True)
        return [self.passages[int(i)] for i in order[:k]]


def build_context(passages):
    return "\n\n".join(f"[{p['id']}] {p['title']}: {p['text']}" for p in passages)


def answer(question, passages):
    prompt = f"""Answer the question using ONLY the passages below.

Rules:
1. Do not use outside knowledge.
2. Give a short answer (1 to 3 sentences) and cite the passage id, like [S05].
3. If the passages do not contain the answer, reply exactly: "{NOT_FOUND}"

Passages:
{build_context(passages)}

Question: {question}

Answer:"""
    return generate(prompt)


def judge_faithfulness(question, passages, answer_text):
    """Ask Gemini whether every claim in the answer is supported by the passages.
    Returns (is_faithful: bool, raw_verdict: str)."""
    prompt = f"""You are a strict checker for a question-answering system.

Read the passages and the answer. Decide if EVERY fact stated in the answer is
directly supported by the passages. Ignore the passage id citations themselves.

Reply with exactly one word on the first line: SUPPORTED or UNSUPPORTED.
On the second line give a short reason (one sentence).

Passages:
{build_context(passages)}

Question: {question}
Answer: {answer_text}"""
    verdict = generate(prompt)
    first_line = verdict.strip().splitlines()[0].upper() if verdict.strip() else ""
    return first_line.startswith("SUPPORTED"), verdict.strip()


def keyword_correct(answer_text, keyword_groups):
    """Correct if, for every group, at least one keyword appears in the answer."""
    text = answer_text.lower()
    return all(any(k.lower() in text for k in group) for group in keyword_groups)


def is_refusal(answer_text):
    return "don't know" in answer_text.lower() or "do not know" in answer_text.lower()
