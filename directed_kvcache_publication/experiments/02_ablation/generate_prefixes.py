#!/usr/bin/env python3
"""Generate AI-derived and TF-IDF prefixes for all documents.

Pre-computes per-document prefixes using Gemini and TF-IDF, stores them in a
JSON file that the ablation sweep notebook loads. Run once before the sweep.

Generates for each document:
  - summary_ai: one-sentence summary (Gemini)
  - instruction_ai: customized reading instruction (Gemini)
  - keywords_tfidf: top-10 TF-IDF terms

Usage:
    cd /home/jupyter/research/directed_kvcache_publication
    python3 experiments/02_ablation/generate_prefixes.py
"""

import os
os.umask(0o000)
import json
import gc
import time
import random as pyrandom
from pathlib import Path
from collections import Counter
import math
import re

from datasets import load_dataset
from google import genai

# ── Config ────────────────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 400
OUTPUT_PATH = Path(__file__).resolve().parent / "generated_prefixes.json"

client = genai.Client(vertexai=True, project="anagram-442822", location="us-central1")
MODEL = "gemini-2.5-flash"


# ── Dataset loading (identical to main sweep) ─────────────────────────

def count_words(text):
    return len(text.split())


def load_all_datasets():
    all_samples = {}

    print("Loading datasets...")
    ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
    candidates = []
    for item in ds:
        passages = item.get("passages", {}).get("passage_text", [])
        passage = " ".join(passages) if passages else ""
        query = item.get("query", "")
        answers = item.get("answers", [])
        answer = answers[0] if answers and answers[0] != "No Answer Present." else ""
        if passage and query and answer:
            wc = count_words(passage)
            if 30 <= wc <= 500:
                candidates.append({"passage": passage, "query": query, "answer": answer,
                                   "passage_words": wc})
    pyrandom.seed(SEED + 100); pyrandom.shuffle(candidates)
    all_samples["ms_marco"] = candidates[:N_SAMPLES]
    print(f"  ms_marco: {len(all_samples['ms_marco'])}")
    del ds, candidates; gc.collect()

    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    candidates = []
    for item in ds:
        passage = item.get("context", ""); query = item.get("question", "")
        answers = item.get("answers", {}).get("text", [])
        answer = answers[0] if answers else ""
        if passage and query and answer:
            wc = count_words(passage)
            if 30 <= wc <= 500:
                candidates.append({"passage": passage, "query": query, "answer": answer,
                                   "passage_words": wc})
    pyrandom.seed(SEED + 200); pyrandom.shuffle(candidates)
    all_samples["squad_v2"] = candidates[:N_SAMPLES]
    print(f"  squad_v2: {len(all_samples['squad_v2'])}")
    del ds, candidates; gc.collect()

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation")
    candidates = []
    for item in ds:
        entity_pages = item.get("entity_pages", {})
        wiki_contexts = entity_pages.get("wiki_context", [])
        if not wiki_contexts or not wiki_contexts[0]: continue
        passage = " ".join(wiki_contexts[0].split()[:500])
        query = item["question"]; answer_val = item["answer"]["value"]
        aliases = item["answer"].get("aliases", [])
        passage_lower = passage.lower()
        found = answer_val.lower() in passage_lower
        if not found:
            for alias in aliases:
                if alias.lower() in passage_lower: found = True; break
        if not found: continue
        wc = count_words(passage)
        if 30 <= wc <= 500 and count_words(answer_val) >= 1:
            candidates.append({"passage": passage, "query": query, "answer": answer_val,
                               "passage_words": wc})
    pyrandom.seed(SEED + 300); pyrandom.shuffle(candidates)
    all_samples["triviaqa"] = candidates[:N_SAMPLES]
    print(f"  triviaqa: {len(all_samples['triviaqa'])}")
    del ds, candidates; gc.collect()

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    candidates = []
    for item in ds:
        context = item.get("context", {}); sf = item.get("supporting_facts", {})
        title_to_sents = {t: s for t, s in zip(context.get("title", []),
                                                 context.get("sentences", []))}
        parts = [title_to_sents[t][sid]
                 for t, sid in zip(sf.get("title", []), sf.get("sent_id", []))
                 if t in title_to_sents and sid < len(title_to_sents[t])]
        if not parts: continue
        passage = " ".join(parts); query = item["question"]; answer = item["answer"]
        wc = count_words(passage)
        if 30 <= wc <= 500 and count_words(answer) >= 1:
            candidates.append({"passage": passage, "query": query, "answer": answer,
                               "passage_words": wc})
    pyrandom.seed(SEED + 400); pyrandom.shuffle(candidates)
    all_samples["hotpotqa"] = candidates[:N_SAMPLES]
    print(f"  hotpotqa: {len(all_samples['hotpotqa'])}")
    del ds, candidates; gc.collect()

    ds = load_dataset("ucinlp/drop", split="validation")
    candidates = []
    for item in ds:
        spans = item.get("answers_spans", {}).get("spans", [])
        if not spans or not spans[0]: continue
        passage = item["passage"]; answer = spans[0]
        wc = count_words(passage)
        if 30 <= wc <= 500 and count_words(answer) >= 1:
            candidates.append({"passage": passage, "query": item["question"],
                               "answer": answer, "passage_words": wc})
    pyrandom.seed(SEED + 500); pyrandom.shuffle(candidates)
    all_samples["drop"] = candidates[:N_SAMPLES]
    print(f"  drop: {len(all_samples['drop'])}")
    del ds, candidates; gc.collect()

    ds = load_dataset("openai/gsm8k", "main", split="test")
    candidates = []
    for item in ds:
        if "####" not in item["answer"]: continue
        answer = item["answer"].split("####")[-1].strip()
        if not answer: continue
        passage = item["question"]; wc = count_words(passage)
        if 10 <= wc <= 500:
            candidates.append({"passage": passage, "query": "What is the answer?",
                               "answer": answer, "passage_words": wc})
    pyrandom.seed(SEED + 600); pyrandom.shuffle(candidates)
    all_samples["gsm8k"] = candidates[:N_SAMPLES]
    print(f"  gsm8k: {len(all_samples['gsm8k'])}")
    del ds, candidates; gc.collect()

    return all_samples


# ── TF-IDF ────────────────────────────────────────────────────────────

def tokenize_simple(text):
    """Simple whitespace + lowercase tokenizer, strip punctuation."""
    return [w.strip(".,;:!?\"'()[]{}") for w in text.lower().split() if len(w) > 2]


def compute_tfidf_prefixes(all_samples, top_k=10):
    """Compute TF-IDF top-k keywords per document within each dataset."""
    result = {}
    for ds_key, samples in all_samples.items():
        # Build document frequency across this dataset
        doc_freq = Counter()
        doc_tokens = []
        for s in samples:
            tokens = set(tokenize_simple(s["passage"]))
            doc_tokens.append(tokens)
            for t in tokens:
                doc_freq[t] += 1

        N = len(samples)
        ds_result = []
        for idx, s in enumerate(samples):
            tf = Counter(tokenize_simple(s["passage"]))
            max_tf = max(tf.values()) if tf else 1
            scores = {}
            for term, count in tf.items():
                if doc_freq[term] < 2:
                    continue  # skip hapax legomena
                tfidf = (count / max_tf) * math.log(N / doc_freq[term])
                scores[term] = tfidf
            top_terms = sorted(scores, key=scores.get, reverse=True)[:top_k]
            ds_result.append(" ".join(top_terms))
        result[ds_key] = ds_result
    return result


# ── Gemini generation ─────────────────────────────────────────────────

SUMMARY_PROMPT = """Read the following text and write a one-sentence summary that captures the key information. Output ONLY the summary sentence, nothing else.

Text:
{passage}"""

INSTRUCTION_PROMPT = """Read the following text. Then write a short instruction (1-2 sentences) that would help someone read this text more effectively and retain the key information. The instruction should be specific to this particular text's content and structure. Output ONLY the instruction, nothing else.

Text:
{passage}"""


def generate_ai_prefixes(all_samples, batch_size=20):
    """Generate AI summaries and instructions for all documents using Gemini."""
    summaries = {}
    instructions = {}

    total = sum(len(v) for v in all_samples.values())
    done = 0

    for ds_key, samples in all_samples.items():
        ds_summaries = []
        ds_instructions = []

        for idx, s in enumerate(samples):
            passage = s["passage"][:2000]  # truncate for API limits

            # Generate summary
            for attempt in range(3):
                try:
                    resp = client.models.generate_content(
                        model=MODEL,
                        contents=SUMMARY_PROMPT.format(passage=passage),
                        config={"temperature": 0.0, "max_output_tokens": 100},
                    )
                    summary = (resp.text or "").strip()
                    if not summary:
                        summary = "This text contains important information."
                    break
                except Exception as e:
                    if attempt == 2:
                        summary = "This text contains important information."
                        print(f"    WARN: summary failed for {ds_key}[{idx}]: {e}")
                    time.sleep(2 ** attempt)

            # Generate instruction
            for attempt in range(3):
                try:
                    resp = client.models.generate_content(
                        model=MODEL,
                        contents=INSTRUCTION_PROMPT.format(passage=passage),
                        config={"temperature": 0.0, "max_output_tokens": 100},
                    )
                    instruction = (resp.text or "").strip()
                    if not instruction:
                        instruction = "Read this text carefully and note the key details."
                    break
                except Exception as e:
                    if attempt == 2:
                        instruction = "Read this text carefully and note the key details."
                        print(f"    WARN: instruction failed for {ds_key}[{idx}]: {e}")
                    time.sleep(2 ** attempt)

            ds_summaries.append(summary)
            ds_instructions.append(instruction)
            done += 1

            if (idx + 1) % 50 == 0:
                print(f"  {ds_key}: {idx+1}/{len(samples)} ({done}/{total} total)")

        summaries[ds_key] = ds_summaries
        instructions[ds_key] = ds_instructions
        print(f"  {ds_key}: done ({len(ds_summaries)} summaries, {len(ds_instructions)} instructions)")

    return summaries, instructions


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("GENERATE PREFIXES FOR ABLATION SWEEP")
    print("=" * 70)

    # Load datasets
    all_samples = load_all_datasets()

    # TF-IDF keywords
    print("\nComputing TF-IDF keywords...")
    tfidf = compute_tfidf_prefixes(all_samples, top_k=10)
    for ds_key in tfidf:
        print(f"  {ds_key}: sample keywords = '{tfidf[ds_key][0][:80]}'")

    # AI-generated prefixes
    print("\nGenerating AI prefixes with Gemini...")
    summaries, instructions = generate_ai_prefixes(all_samples)

    # Save everything
    output = {
        "metadata": {
            "model": MODEL,
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "tfidf_keywords": tfidf,
        "ai_summaries": summaries,
        "ai_instructions": instructions,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Total: {sum(len(v) for v in tfidf.values())} TF-IDF, "
          f"{sum(len(v) for v in summaries.values())} summaries, "
          f"{sum(len(v) for v in instructions.values())} instructions")


if __name__ == "__main__":
    main()
