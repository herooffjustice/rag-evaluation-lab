"""
evaluate.py - measures the RAG system in two parts.

Part A. Retrieval (no LLM, same result every run)
  For the 20 answerable questions we know the gold passage.
  Recall@1, Recall@3 and MRR are computed for BM25, dense and hybrid.

Part B. Generation (uses Gemini)
  With the hybrid retriever by default (top 3 passages), Gemini answers all 24 questions.
  - Answer accuracy : expected keywords appear in the answer (answerable only)
  - Faithfulness    : a Gemini judge says every fact is supported by the passages
  - Refusal         : for the 4 unanswerable questions, did the system say
                      "I don't know"? (and for answerable ones, did it wrongly refuse?)

Run:
  python evaluate.py                  # both parts
  python evaluate.py --retrieval-only # part A only (no API key needed)
"""

import argparse
import csv
import json
import time
from pathlib import Path

from rag_core import (TOP_K, Retriever, answer, is_refusal, judge_faithfulness,
                      keyword_correct, load_json)

MODES = ["bm25", "dense", "hybrid"]
PAUSE = 4   # seconds between Gemini calls (free-tier rate limit)


def retrieval_eval(retriever, tests):
    answerable = [t for t in tests if t["gold"]]
    n_passages = len(retriever.passages)
    summary, rows = {}, []
    for mode in MODES:
        ranks = {"all": [], "keyword": [], "paraphrase": []}
        for t in answerable:
            ids = [p["id"] for p in retriever.search(t["question"], k=n_passages, mode=mode)]
            rank = ids.index(t["gold"]) + 1
            ranks["all"].append(rank)
            ranks[t["type"]].append(rank)
            rows.append({"mode": mode, "qid": t["qid"], "gold": t["gold"],
                         "rank": rank, "top3": ids[:3]})
        summary[mode] = {
            group: {"recall_at_1": round(sum(r == 1 for r in rs) / len(rs), 3),
                    "recall_at_3": round(sum(r <= 3 for r in rs) / len(rs), 3),
                    "mrr": round(sum(1 / r for r in rs) / len(rs), 3),
                    "n": len(rs)}
            for group, rs in ranks.items()
        }
    return summary, rows


CACHE = Path("outputs") / "generation_cache.json"


def generation_eval(retriever, tests, mode="hybrid", max_seconds=None):
    """Answers and judges every question. Finished questions are saved to a
    cache file, so if the run stops (time limit, rate limit) the next run
    continues from where it stopped. Returns None until all are done."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    if cache.get("_mode") != mode:
        cache = {"_mode": mode}
    start = time.time()
    for t in tests:
        if t["qid"] in cache:
            continue
        if max_seconds and time.time() - start > max_seconds:
            done = len(cache) - 1
            print(f"Stopped after the time limit: {done}/{len(tests)} done. Run again to continue.")
            return None, None
        passages = retriever.search(t["question"], k=TOP_K, mode=mode)
        ans = answer(t["question"], passages)
        time.sleep(PAUSE)
        refused = is_refusal(ans)
        if refused:
            faithful, verdict = True, "SKIPPED (refusal makes no factual claim)"
        else:
            faithful, verdict = judge_faithfulness(t["question"], passages, ans)
            time.sleep(PAUSE)
        row = {
            "qid": t["qid"], "type": t["type"], "question": t["question"],
            "gold": t["gold"], "retrieved": [p["id"] for p in passages],
            "gold_retrieved": t["gold"] in [p["id"] for p in passages] if t["gold"] else None,
            "answer": ans, "refused": refused,
            "keyword_correct": keyword_correct(ans, t["keywords"]) if t["gold"] else None,
            "faithful": faithful, "judge_verdict": verdict,
        }
        cache[t["qid"]] = row
        CACHE.write_text(json.dumps(cache, indent=2))
        print(f"{t['qid']}  refused={refused!s:<5} correct={row['keyword_correct']!s:<5} "
              f"faithful={faithful!s:<5} | {ans[:70]}")

    rows = [cache[t["qid"]] for t in tests]
    answerable = [r for r in rows if r["gold"]]
    unanswerable = [r for r in rows if not r["gold"]]
    judged = [r for r in rows if not r["refused"]]
    summary = {
        "answer_accuracy": round(sum(r["keyword_correct"] for r in answerable) / len(answerable), 3),
        "faithfulness": round(sum(r["faithful"] for r in judged) / len(judged), 3) if judged else None,
        "num_judged": len(judged),
        "correct_refusal_rate": round(sum(r["refused"] for r in unanswerable) / len(unanswerable), 3),
        "wrong_refusal_rate": round(sum(r["refused"] for r in answerable) / len(answerable), 3),
        "num_answerable": len(answerable),
        "num_unanswerable": len(unanswerable),
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--mode", default="hybrid", choices=MODES,
                        help="retriever used for Part B")
    parser.add_argument("--max-seconds", type=int, default=None,
                        help="stop Part B after this many seconds (progress is saved)")
    args = parser.parse_args()

    passages, tests = load_json("passages.json"), load_json("test_set.json")
    retriever = Retriever(passages)
    Path("outputs").mkdir(exist_ok=True)

    print("=== Part A: Retrieval ===")
    r_summary, r_rows = retrieval_eval(retriever, tests)
    print(f"{'mode':<8}{'group':<12}{'R@1':>6}{'R@3':>6}{'MRR':>7}")
    for mode, groups in r_summary.items():
        for group, m in groups.items():
            print(f"{mode:<8}{group:<12}{m['recall_at_1']:>6}{m['recall_at_3']:>6}{m['mrr']:>7}")
    with open("outputs/retrieval_results.json", "w") as f:
        json.dump({"summary": r_summary, "rows": r_rows}, f, indent=2)

    if args.retrieval_only:
        return

    print(f"\n=== Part B: Generation ({args.mode}, top {TOP_K}) ===")
    g_summary, g_rows = generation_eval(retriever, tests, mode=args.mode,
                                        max_seconds=args.max_seconds)
    if g_summary is None:
        return
    g_summary["retriever"] = args.mode
    print(json.dumps(g_summary, indent=2))
    with open("outputs/generation_results.json", "w") as f:
        json.dump({"summary": g_summary, "rows": g_rows}, f, indent=2)
    with open("outputs/generation_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(g_rows[0].keys()))
        writer.writeheader()
        writer.writerows(g_rows)
    print("\nSaved outputs/retrieval_results.json, outputs/generation_results.json/.csv")


if __name__ == "__main__":
    main()
