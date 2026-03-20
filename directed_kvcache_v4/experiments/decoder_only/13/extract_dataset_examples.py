#!/usr/bin/env python3
"""Extract passage text for the first hard sample of each dataset.

Replays the exact loading/filtering/shuffling pipeline from build_exp13.py
to recover the passage text that matches the checkpoint query/answer pairs.
Writes results to results/decoder_only/exp13/dataset_examples.json.

Usage:
    cd experiments/decoder_only/13
    python extract_dataset_examples.py
"""

import os
os.umask(0o000)
os.environ['HF_HOME'] = '/home/petergrabowski_google_com/.cache/huggingface'

import sys
sys.path.insert(0, "../../..")

import json
import numpy as np
from pathlib import Path
from datasets import load_dataset
from lib.data import count_words

SEED = 42
N_SAMPLES = 400
HARD_FRAC = 0.4
N_HARD = int(N_SAMPLES * HARD_FRAC)  # 160

DS_SEEDS = {
    'squad_v2': SEED + 100,
    'triviaqa': SEED + 200,
    'hotpotqa': SEED + 300,
    'drop': SEED + 400,
    'boolq': SEED + 500,
    'gsm8k': SEED + 700,
}

RESULTS_DIR = Path("../../../results/decoder_only/exp13")
EXP02_DIR = Path("../../../results/decoder_only/exp02")
EXP03_DIR = Path("../../../results/decoder_only/exp03")
EXP05_DIR = Path("../../../results/decoder_only/exp05")
EXP06_DIR = Path("../../../results/decoder_only/exp06")

examples = {}

# ================================================================
# MS MARCO
# ================================================================
print("--- MS MARCO ---")
exp02_ckpt = json.loads((EXP02_DIR / "checkpoint.json").read_text())
exp02_results = exp02_ckpt['results']
msmarco_bare = np.array([r['nll_bare'] for r in exp02_results])
sorted_idx = np.argsort(msmarco_bare)[::-1]
msmarco_hard_idx = np.sort(sorted_idx[:N_HARD])

ds_msmarco = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
msmarco_candidates = []
for item in ds_msmarco:
    if len(msmarco_candidates) >= 3 * N_SAMPLES:
        break
    passages = item.get('passages', {})
    ptexts = passages.get('passage_text', [])
    is_sel = passages.get('is_selected', [])
    query = item.get('query', '')
    answers = item.get('answers', [])
    well_formed = item.get('wellFormedAnswers', [])
    answer = None
    if well_formed and len(well_formed) > 0 and well_formed[0] not in ('[]', ''):
        answer = well_formed[0]
    elif answers and len(answers) > 0 and answers[0] != 'No Answer Present.':
        answer = answers[0]
    if not answer:
        continue
    for pt, sel in zip(ptexts, is_sel):
        wc = count_words(pt)
        if sel == 1 and 30 <= wc <= 300:
            msmarco_candidates.append({
                'passage': pt, 'query': query, 'answer': answer,
                'word_count': wc,
            })
            break

np.random.seed(SEED)
indices = np.random.permutation(len(msmarco_candidates))[:N_SAMPLES]
msmarco_all = [msmarco_candidates[i] for i in indices]

first_hard = msmarco_all[msmarco_hard_idx[0]]
# Verify against checkpoint
ckpt = json.loads((RESULTS_DIR / "checkpoint_ms_marco.json").read_text())
assert ckpt['results'][0]['query'] == first_hard['query'], "MS MARCO query mismatch!"
examples['ms_marco'] = {
    'passage': first_hard['passage'],
    'query': first_hard['query'],
    'answer': first_hard['answer'],
    'word_count': first_hard['word_count'],
}
print(f"  query: {first_hard['query']}")
print(f"  passage: {first_hard['passage'][:80]}...")
del ds_msmarco, msmarco_candidates, msmarco_all

# ================================================================
# Helper: load dataset, filter, shuffle, get first hard sample
# ================================================================
BARE_NLL_SOURCES = {
    'squad_v2': EXP03_DIR / "bare_squad_v2.json",
    'triviaqa': EXP03_DIR / "bare_triviaqa.json",
    'hotpotqa': EXP03_DIR / "bare_hotpotqa.json",
    'drop': EXP05_DIR / "bare_drop.json",
    'boolq': EXP05_DIR / "bare_boolq.json",
    'gsm8k': EXP06_DIR / "bare_gsm8k.json",
}


def get_first_hard(ds_name, all_samples_ds):
    bare_ckpt = json.loads(BARE_NLL_SOURCES[ds_name].read_text())
    bare_nlls_all = bare_ckpt['bare_nlls'][:N_SAMPLES]
    bare_arr = np.array(bare_nlls_all)
    sorted_idx = np.argsort(bare_arr)[::-1]
    h_idx = np.sort(sorted_idx[:N_HARD])
    first = all_samples_ds[h_idx[0]]
    # Verify
    ckpt = json.loads((RESULTS_DIR / f"checkpoint_{ds_name}.json").read_text())
    assert ckpt['results'][0]['query'] == first['query'], f"{ds_name} query mismatch!"
    return first


# ================================================================
# SQuAD 2.0
# ================================================================
print("\n--- SQuAD 2.0 ---")
ds_squad = load_dataset("rajpurkar/squad_v2", split="validation")
squad_candidates = []
for item in ds_squad:
    answers = item.get('answers', {})
    answer_texts = answers.get('text', [])
    if not answer_texts:
        continue
    passage = item['context']
    query = item['question']
    answer = answer_texts[0]
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        squad_candidates.append({
            'passage': passage, 'query': query, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['squad_v2'])
sq_indices = np.random.permutation(len(squad_candidates))[:N_SAMPLES]
squad_all = [squad_candidates[i] for i in sq_indices]
first = get_first_hard('squad_v2', squad_all)
examples['squad_v2'] = {
    'passage': first['passage'],
    'query': first['query'],
    'answer': first['answer'],
    'word_count': first['word_count'],
}
print(f"  query: {first['query']}")
print(f"  passage: {first['passage'][:80]}...")
del ds_squad, squad_candidates, squad_all

# ================================================================
# TriviaQA
# ================================================================
print("\n--- TriviaQA ---")
ds_trivia = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation")
trivia_candidates = []
for item in ds_trivia:
    entity_pages = item.get('entity_pages', {})
    wiki_contexts = entity_pages.get('wiki_context', [])
    if not wiki_contexts or not wiki_contexts[0]:
        continue
    words = wiki_contexts[0].split()[:500]
    passage = ' '.join(words)
    query = item['question']
    answer_val = item['answer']['value']
    aliases = item['answer'].get('aliases', [])
    passage_lower = passage.lower()
    found = answer_val.lower() in passage_lower
    if not found:
        for alias in aliases:
            if alias.lower() in passage_lower:
                found = True
                break
    if not found:
        continue
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer_val) >= 1:
        trivia_candidates.append({
            'passage': passage, 'query': query, 'answer': answer_val,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['triviaqa'])
tr_indices = np.random.permutation(len(trivia_candidates))[:N_SAMPLES]
trivia_all = [trivia_candidates[i] for i in tr_indices]
first = get_first_hard('triviaqa', trivia_all)
examples['triviaqa'] = {
    'passage': first['passage'],
    'query': first['query'],
    'answer': first['answer'],
    'word_count': first['word_count'],
}
print(f"  query: {first['query']}")
print(f"  passage: {first['passage'][:80]}...")
del ds_trivia, trivia_candidates, trivia_all

# ================================================================
# HotpotQA
# ================================================================
print("\n--- HotpotQA ---")
ds_hotpot = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
hotpot_candidates = []
for item in ds_hotpot:
    context = item.get('context', {})
    sf = item.get('supporting_facts', {})
    ctx_titles = context.get('title', [])
    ctx_sentences = context.get('sentences', [])
    sf_titles = sf.get('title', [])
    sf_sent_ids = sf.get('sent_id', [])
    title_to_sents = {}
    for title, sents in zip(ctx_titles, ctx_sentences):
        title_to_sents[title] = sents
    passage_parts = []
    for title, sid in zip(sf_titles, sf_sent_ids):
        if title in title_to_sents and sid < len(title_to_sents[title]):
            passage_parts.append(title_to_sents[title][sid])
    if not passage_parts:
        continue
    passage = ' '.join(passage_parts)
    query = item['question']
    answer = item['answer']
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        hotpot_candidates.append({
            'passage': passage, 'query': query, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['hotpotqa'])
hp_indices = np.random.permutation(len(hotpot_candidates))[:N_SAMPLES]
hotpot_all = [hotpot_candidates[i] for i in hp_indices]
first = get_first_hard('hotpotqa', hotpot_all)
examples['hotpotqa'] = {
    'passage': first['passage'],
    'query': first['query'],
    'answer': first['answer'],
    'word_count': first['word_count'],
}
print(f"  query: {first['query']}")
print(f"  passage: {first['passage'][:80]}...")
del ds_hotpot, hotpot_candidates, hotpot_all

# ================================================================
# DROP
# ================================================================
print("\n--- DROP ---")
ds_drop = load_dataset("ucinlp/drop", split="validation")
drop_candidates = []
for item in ds_drop:
    passage = item['passage']
    question = item['question']
    answers_spans = item.get('answers_spans', {})
    spans = answers_spans.get('spans', [])
    if not spans or not spans[0]:
        continue
    answer = spans[0]
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        drop_candidates.append({
            'passage': passage, 'query': question, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['drop'])
drop_indices = np.random.permutation(len(drop_candidates))[:N_SAMPLES]
drop_all = [drop_candidates[i] for i in drop_indices]
first = get_first_hard('drop', drop_all)
examples['drop'] = {
    'passage': first['passage'],
    'query': first['query'],
    'answer': first['answer'],
    'word_count': first['word_count'],
}
print(f"  query: {first['query']}")
print(f"  passage: {first['passage'][:80]}...")
del ds_drop, drop_candidates, drop_all

# ================================================================
# BoolQ
# ================================================================
print("\n--- BoolQ ---")
ds_boolq = load_dataset("google/boolq", split="validation")
boolq_candidates = []
for item in ds_boolq:
    passage = item['passage']
    question = item['question']
    answer = "Yes" if item['answer'] else "No"
    wc = count_words(passage)
    if 30 <= wc <= 500:
        boolq_candidates.append({
            'passage': passage, 'query': question, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['boolq'])
boolq_indices = np.random.permutation(len(boolq_candidates))[:N_SAMPLES]
boolq_all = [boolq_candidates[i] for i in boolq_indices]
first = get_first_hard('boolq', boolq_all)
examples['boolq'] = {
    'passage': first['passage'],
    'query': first['query'],
    'answer': first['answer'],
    'word_count': first['word_count'],
}
print(f"  query: {first['query']}")
print(f"  passage: {first['passage'][:80]}...")
del ds_boolq, boolq_candidates, boolq_all

# ================================================================
# GSM8K
# ================================================================
print("\n--- GSM8K ---")
ds_gsm8k = load_dataset("openai/gsm8k", "main", split="test")
gsm8k_candidates = []
for item in ds_gsm8k:
    passage = item['question']
    raw_answer = item['answer']
    if '####' not in raw_answer:
        continue
    answer = raw_answer.split('####')[-1].strip()
    if not answer:
        continue
    query = "What is the answer?"
    wc = count_words(passage)
    if 30 <= wc <= 500:
        gsm8k_candidates.append({
            'passage': passage, 'query': query, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['gsm8k'])
gsm8k_indices = np.random.permutation(len(gsm8k_candidates))[:N_SAMPLES]
gsm8k_all = [gsm8k_candidates[i] for i in gsm8k_indices]
first = get_first_hard('gsm8k', gsm8k_all)
examples['gsm8k'] = {
    'passage': first['passage'],
    'query': first['query'],
    'answer': first['answer'],
    'word_count': first['word_count'],
}
print(f"  query: {first['query']}")
print(f"  passage: {first['passage'][:80]}...")
del ds_gsm8k, gsm8k_candidates, gsm8k_all

# ================================================================
# Write output
# ================================================================
out_path = RESULTS_DIR / "dataset_examples.json"
out_path.write_text(json.dumps(examples, indent=2, ensure_ascii=False))
print(f"\nWrote {out_path}")
for ds_name, ex in examples.items():
    print(f"  {ds_name}: {ex['word_count']} words, query={ex['query'][:50]}")
