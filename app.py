"""
app.py - Streamlit dashboard for the RAG Evaluation Lab.

Run:  streamlit run app.py
(Run `python evaluate.py` first to fill the Results tab.)
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from llm import get_model_name
from rag_core import (TOP_K, Retriever, answer, is_refusal, judge_faithfulness,
                      load_json)

st.set_page_config(page_title="RAG Evaluation Lab", page_icon="🧪", layout="wide")


@st.cache_resource(show_spinner="Loading passages and building indexes...")
def get_retriever():
    return Retriever(load_json("passages.json"))


retriever = get_retriever()
tests = load_json("test_set.json")

with st.sidebar:
    st.header("Data")
    st.metric("Science passages", len(retriever.passages))
    st.metric("Test questions", len(tests))
    st.dataframe(pd.Series([p["topic"] for p in retriever.passages]).value_counts()
                 .rename("passages"), use_container_width=True)
    st.caption(f"LLM and judge: {get_model_name()} · Embeddings: all-MiniLM-L6-v2")

st.title("RAG Evaluation Lab")
st.write("A small science Q&A system built to be **measured**: retrieval is scored with "
         "Recall@k and MRR, and answers are scored for accuracy, faithfulness and correct "
         "refusal.")

tab_results, tab_ask, tab_tests = st.tabs(["Results", "Ask", "Test set"])

with tab_results:
    r_file = Path("outputs") / "retrieval_results.json"
    g_file = Path("outputs") / "generation_results.json"
    if r_file.exists():
        st.subheader("Part A · Retrieval (20 answerable questions)")
        data = json.loads(r_file.read_text())
        rows = [{"retriever": mode, "question type": group, "n": m["n"],
                 "Recall@1": m["recall_at_1"], "Recall@3": m["recall_at_3"], "MRR": m["mrr"]}
                for mode, groups in data["summary"].items() for group, m in groups.items()]
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.bar_chart(df[df["question type"] != "all"].pivot(
            index="question type", columns="retriever", values="MRR"), stack=False)
    else:
        st.info("Run `python evaluate.py --retrieval-only` to create the retrieval results.")

    if g_file.exists():
        g = json.loads(g_file.read_text())
        s = g["summary"]
        st.subheader(f"Part B · Generation ({s.get('retriever', 'hybrid')} retriever, top {TOP_K})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Answer accuracy", f"{s['answer_accuracy']:.0%}",
                  help=f"Expected keywords found, {s['num_answerable']} answerable questions")
        c2.metric("Faithfulness", f"{s['faithfulness']:.0%}",
                  help=f"Judge says SUPPORTED, {s['num_judged']} non-refused answers")
        c3.metric("Correct refusals", f"{s['correct_refusal_rate']:.0%}",
                  help=f"{s['num_unanswerable']} questions not covered by the passages")
        c4.metric("Wrong refusals", f"{s['wrong_refusal_rate']:.0%}",
                  help="Answerable questions where the system said it did not know")
        table = pd.DataFrame(g["rows"])[["qid", "type", "question", "gold", "retrieved",
                                         "answer", "keyword_correct", "faithful", "refused"]]
        st.dataframe(table, hide_index=True, use_container_width=True)
    else:
        st.info("Run `python evaluate.py` (needs the Gemini key) to create generation results.")

with tab_ask:
    mode = st.radio("Retriever", ["hybrid", "dense", "bm25"], horizontal=True)
    question = st.text_input("Ask a science question",
                             "Why does a rifle kick back against the shoulder?")
    check = st.checkbox("Also run the faithfulness judge", value=True)
    if st.button("Ask", type="primary") and question.strip():
        passages = retriever.search(question, k=TOP_K, mode=mode)
        with st.spinner("Asking Gemini..."):
            ans = answer(question, passages)
        st.subheader("Answer")
        st.success(ans)
        if check and not is_refusal(ans):
            with st.spinner("Running the judge..."):
                ok, verdict = judge_faithfulness(question, passages, ans)
            (st.info if ok else st.error)(f"Judge: {verdict}")
        st.subheader("Retrieved passages")
        for p in passages:
            with st.expander(f"[{p['id']}] {p['title']} ({p['topic']})"):
                st.write(p["text"])

with tab_tests:
    st.dataframe(pd.DataFrame(tests), hide_index=True, use_container_width=True)
