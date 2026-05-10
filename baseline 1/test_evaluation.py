"""
Test harness for the Ayurvedic pipeline.

Reads test_conversations.txt and, for each conversation, simulates the
pipeline turn-by-turn — feeding user messages exactly as the real
pipeline would receive them. Compares the architecture's predicted
problems against the "Expected:" ground-truth labels and reports
accuracy across three match levels.

Modes:
  default  : Phase 0 only (semantic search). Fast, no API key needed.
  --full   : Full pipeline — incremental persona update each turn +
             final Phase 2 recommendation. Requires GROQ_API_KEY and
             consumes API quota.

Match levels reported:
  Top-1   : the single highest-scoring detected ID is in expected
  Any@K   : at least one detected (top-K) ID is in expected (lenient)
  All@K   : every expected ID appears in detected top-K  (strict —
            relevant for the cross-cutting conversations 22-30)

Run:
  python test_evaluation.py
  python test_evaluation.py --full
  python test_evaluation.py --top-k 5
"""

import argparse
import json
import os
import re
import time

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from pipeline import (
    build_problem_index,
    cosine_similarity,
    get_simplified_map,
    load_json,
    recommend_plan,
    update_persona,
)

load_dotenv()


# ============================================================
# CONVERSATION PARSING
# ============================================================
def parse_test_conversations(filepath):
    """
    Parse test_conversations.txt into a list of dicts:
      { conv_num, user_messages (excluding 'done'), expected_str, expected_ids }
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    parts = re.split(r"-----\s*Conversation\s*(\d+)\s*-----", content)[1:]
    conversations = []

    for i in range(0, len(parts), 2):
        conv_num = int(parts[i])
        block = parts[i + 1]

        user_messages = []
        expected_str = None

        for raw_line in block.splitlines():
            line = raw_line.strip()
            if line.startswith("You:"):
                msg = line[len("You:"):].strip()
                if msg and msg.lower() != "done":
                    user_messages.append(msg)
            elif line.startswith("Expected:"):
                expected_str = line[len("Expected:"):].strip()

        if user_messages and expected_str:
            conversations.append({
                "conv_num": conv_num,
                "user_messages": user_messages,
                "expected_str": expected_str,
                "expected_ids": re.findall(r"[A-Z]+_\d{3}", expected_str),
            })

    return conversations


# ============================================================
# OPTIMIZED SEARCH (uses pre-built index)
# ============================================================
def search_with_index(user_messages, problem_index, embed_model, top_k=3):
    """
    Same logic as pipeline.search_top_problems but takes a pre-built
    problem_index so we don't re-embed 200+ synonyms for every test.
    """
    if not user_messages:
        return []

    user_embs = embed_model.encode(user_messages, convert_to_numpy=True)

    best_per_problem = {}
    for problem, synonym, syn_emb in problem_index:
        pid = problem["id"]
        for u_emb, u_msg in zip(user_embs, user_messages):
            score = cosine_similarity(u_emb, syn_emb)
            if pid not in best_per_problem or score > best_per_problem[pid]["score"]:
                best_per_problem[pid] = {
                    "problem": problem,
                    "best_synonym": synonym,
                    "best_message": u_msg,
                    "score": score,
                }

    ranked = sorted(best_per_problem.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


# ============================================================
# MATCH EVALUATION
# ============================================================
def evaluate_match(detected_ids, expected_ids):
    """
    Returns three booleans:
      top1     : highest-score detected ID is one of the expected IDs
      any_at_k : at least one detected ID overlaps with expected
      all_at_k : every expected ID is found in detected top-K
    """
    if not detected_ids or not expected_ids:
        return {"top1": False, "any": False, "all": False}
    return {
        "top1": detected_ids[0] in expected_ids,
        "any": any(d in expected_ids for d in detected_ids),
        "all": all(e in detected_ids for e in expected_ids),
    }


# ============================================================
# TURN-BY-TURN SIMULATION
# ============================================================
def simulate_phase0(user_messages, problem_index, embed_model, top_k):
    """Phase 0 only: semantic search across all messages."""
    return search_with_index(user_messages, problem_index, embed_model, top_k=top_k)


def simulate_full(user_messages, problem_index, problems_db, tasks_db,
                  simplified_schema, embed_model, llm_json, top_k):
    """
    Full simulation:
      1. Update persona ONE TURN AT A TIME (LLM call per turn)
      2. Run Phase 0 semantic search at the end (top_k problems)
      3. Run Phase 2 final plan generation
    Returns (top_problems, persona, plan)
    """
    persona = {}
    for msg in user_messages:
        persona = update_persona(persona, msg, simplified_schema, llm_json)

    top_problems = search_with_index(user_messages, problem_index, embed_model, top_k=top_k)
    plan = recommend_plan(persona, top_problems, tasks_db, llm_json)

    return top_problems, persona, plan


# ============================================================
# REPORTING
# ============================================================
def print_results_table(results, top_k):
    """Pretty-print the per-conversation results."""
    print("\n" + "=" * 110)
    print("DETAILED RESULTS")
    print("=" * 110)
    header = (
        f"{'#':<4}"
        f"{'Expected':<22}"
        f"{f'Detected top-{top_k} (id : score)':<54}"
        f"{'Top1':<6}"
        f"{f'Any@{top_k}':<7}"
        f"{f'All@{top_k}':<7}"
    )
    print(header)
    print("-" * 110)
    for r in results:
        print(
            f"{r['idx']:<4}"
            f"{r['expected']:<22}"
            f"{r['detected']:<54}"
            f"{('YES' if r['top1'] else 'no'):<6}"
            f"{('YES' if r['any'] else 'no'):<7}"
            f"{('YES' if r['all'] else 'no'):<7}"
        )


def print_summary(counts, n, top_k, mode_label, total_time):
    print("\n" + "=" * 110)
    print(f"SUMMARY  |  Mode: {mode_label}  |  Total time: {total_time:.1f}s")
    print("=" * 110)
    print(f"  Total conversations evaluated:    {n}")
    print(f"  Top-1 match  (highest detected in expected):              "
          f"{counts['top1']:>3}/{n}  ({counts['top1']/n*100:.1f}%)")
    print(f"  Any@{top_k}  (at least one expected in top-{top_k}):              "
          f"{counts['any']:>3}/{n}  ({counts['any']/n*100:.1f}%)")
    print(f"  All@{top_k}  (every expected appears in top-{top_k}):            "
          f"{counts['all']:>3}/{n}  ({counts['all']/n*100:.1f}%)")
    print("=" * 110)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Ayurvedic pipeline against test_conversations.txt."
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run full pipeline (turn-by-turn persona updates + Phase 2 plan). "
             "Requires GROQ_API_KEY. Slower and uses API quota."
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of top problems to consider (default: 3)."
    )
    parser.add_argument(
        "--conversations-file", default="test_conversations.txt",
        help="Path to the test conversations file (default: test_conversations.txt)."
    )
    parser.add_argument(
        "--show-plan", action="store_true",
        help="Print Phase 2 analysis summary for each conversation (only meaningful with --full)."
    )
    args = parser.parse_args()

    mode_label = "FULL pipeline (LLM)" if args.full else "Phase 0 only (semantic search)"
    print("=" * 110)
    print(f"PIPELINE EVALUATION  |  Mode: {mode_label}  |  Top-K: {args.top_k}")
    print("=" * 110)

    # ---- Load resources ----
    print("\n[1/5] Loading embedding model (all-MiniLM-L6-v2)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("[2/5] Loading problems.json and pre-building synonym index...")
    problems_db = load_json("problems.json")
    problem_index = build_problem_index(problems_db, embed_model)
    print(f"      → indexed {len(problem_index)} synonym embeddings "
          f"across {len(problems_db.get('problems', []))} problems")

    llm_json = None
    tasks_db = None
    simplified_schema = None
    if args.full:
        if not os.environ.get("GROQ_API_KEY"):
            print("\nERROR: GROQ_API_KEY is not set. --full mode requires it.")
            print("Either set the env var or remove --full to run Phase 0 only.")
            return

        print("[3/5] Loading tasks.json + user_attributes.json + initializing Groq LLM...")
        from langchain_groq import ChatGroq
        tasks_db = load_json("tasks.json")
        schema = load_json("user_attributes.json")
        simplified_schema = get_simplified_map(schema)
        llm_json = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    else:
        print("[3/5] Skipping LLM init (Phase 0 only mode)")

    print(f"[4/5] Parsing conversations from {args.conversations_file}...")
    conversations = parse_test_conversations(args.conversations_file)
    print(f"      → {len(conversations)} conversations parsed")

    print("[5/5] Running evaluation...\n")

    # ---- Run evaluation ----
    results = []
    counts = {"top1": 0, "any": 0, "all": 0}
    t_start = time.time()

    for conv in conversations:
        idx = conv["conv_num"]
        user_msgs = conv["user_messages"]
        expected_ids = conv["expected_ids"]
        expected_str = conv["expected_str"]

        # Print live progress
        preview = expected_str if len(expected_str) <= 55 else expected_str[:52] + "..."
        n_turns = len(user_msgs)
        print(f"  Conv {idx:>2}/{len(conversations)}  "
              f"({n_turns} turn{'s' if n_turns != 1 else ''})  →  Expected: {preview}")

        if args.full:
            top_problems, persona, plan = simulate_full(
                user_msgs, problem_index, problems_db, tasks_db,
                simplified_schema, embed_model, llm_json, top_k=args.top_k
            )
            if args.show_plan and isinstance(plan, dict):
                summary = plan.get("analysis_summary", "")
                if summary:
                    print(f"           Phase 2 summary: {summary[:120]}...")
        else:
            top_problems = simulate_phase0(
                user_msgs, problem_index, embed_model, top_k=args.top_k
            )

        detected_ids = [p["problem"]["id"] for p in top_problems]
        detected_display = "  ".join(
            f"{p['problem']['id']}:{p['score']:.2f}" for p in top_problems
        )

        match = evaluate_match(detected_ids, expected_ids)
        for key in counts:
            if match[key]:
                counts[key] += 1

        results.append({
            "idx": idx,
            "expected": " + ".join(expected_ids) if expected_ids else "?",
            "detected": detected_display,
            **match,
        })

    total_time = time.time() - t_start

    # ---- Report ----
    print_results_table(results, args.top_k)
    print_summary(counts, len(conversations), args.top_k, mode_label, total_time)


if __name__ == "__main__":
    main()
