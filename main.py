"""
main.py - ask single questions from the command line.

Usage:
  python main.py                   # interactive, hybrid retriever
  python main.py --mode bm25       # try another retriever
For the full evaluation, run:  python evaluate.py
"""

import argparse

from rag_core import TOP_K, Retriever, answer, is_refusal, judge_faithfulness, load_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="hybrid", choices=["bm25", "dense", "hybrid"])
    args = parser.parse_args()
    retriever = Retriever(load_json("passages.json"))
    print(f"Loaded {len(retriever.passages)} passages. Retriever: {args.mode}")
    while True:
        q = input("\nQuestion (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit", ""}:
            break
        passages = retriever.search(q, k=TOP_K, mode=args.mode)
        print("Retrieved:", [p["id"] + " " + p["title"] for p in passages])
        ans = answer(q, passages)
        print("Answer:", ans)
        if not is_refusal(ans):
            ok, verdict = judge_faithfulness(q, passages, ans)
            print("Judge:", verdict.splitlines()[0])


if __name__ == "__main__":
    main()
