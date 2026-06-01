#!/usr/bin/env python3
"""Build the contrastive answer-discrimination sweep.

Motivation
----------
Prior sweeps measured absolute answer NLL. But the generation eval (exp04)
showed that NLL improvement does NOT track Exact Match: priming can lower NLL
without improving correctness. That is the signature of *entropy reduction*
(the prefix makes the model globally more confident) rather than *useful
representation* (the prefix surfaces the correct answer).

Absolute NLL cannot distinguish these two. A CONTRASTIVE MARGIN can:

    margin = mean_over_distractors(NLL(distractor)) - NLL(correct)

If priming only reduces entropy, it lowers NLL on correct AND distractors
equally -> margin unchanged -> correctly reports "no real gain".
If priming genuinely surfaces the answer, it lowers NLL on the correct answer
MORE than on distractors -> margin grows -> real discrimination signal.

This sweep also resolves the TF-IDF *repetition* confound. TF-IDF keywords are
words copied from the document, so "keyword priming" might just be in-document
repetition rather than salience. We add a matched control:
  - random_docwords: random DISTINCT words from the same filtered pool TF-IDF
    draws from, chosen at random instead of by salience.

Length is matched across content sources (L=16) to remove the length confound
that contaminated the keywords-vs-instructions comparison.

Conditions (8)
--------------
  bare                 no prefix (baseline)
  tfidf_16             top TF-IDF document words, length-matched to 16
  random_docwords_16   random distinct document words, length-matched to 16   <- salience control
  random_vocab_16      random vocabulary tokens, 16                            <- repetition control
  generic_instr_16     "Extract the key facts from this text." -> 16
  oracle_16            the query itself -> 16
  tfidf_4              top TF-IDF words -> 4    (length curve)
  tfidf_64             top TF-IDF words -> 64   (length curve)

Metrics per (sample, condition)
--------------------------------
  nll_correct          mean NLL over correct-answer tokens (reproduces old metric)
  nll_correct_first    NLL of the FIRST correct-answer token (length-invariant)
  margin_mean          mean(distractor mean-NLL) - nll_correct
  margin_first         mean(distractor first-NLL) - nll_correct_first
  rank                 rank of correct among {correct + K distractors} (1 = best)
  top1                 1 if correct has lowest NLL else 0

Primary outcome: delta_margin = margin(condition) - margin(bare), per sample.
Positive delta_margin = priming improves answer DISCRIMINATION, not just confidence.

Models (5): Qwen 1.5B, Qwen 7B IT, Mistral 7B, Gemma 12B, Ministral 8B
Datasets (4): SQuAD v2, HotpotQA, TriviaQA, GSM8K
N_EVAL=300 samples, K=7 distractors (8-way discrimination)

Usage:
    cd /home/jupyter/research/directed_kvcache_publication
    python3 experiments/04_discrimination/build_discrimination_sweep.py
    cd experiments/04_discrimination
    # smoke test first:
    SMOKE=1 papermill 04_discrimination_sweep.ipynb 04_smoke.ipynb --no-progress-bar
    # full run:
    papermill 04_discrimination_sweep.ipynb 04_discrimination_sweep_executed.ipynb --no-progress-bar
"""

import os
import ast
import nbformat as nbf

os.makedirs("experiments/04_discrimination", exist_ok=True)

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise SyntaxError(f"Cell syntax error line {e.lineno}: {e.msg}\n  {e.text}") from e
    nb.cells.append(nbf.v4.new_code_cell(source))


md(r"""# Contrastive Answer-Discrimination Sweep

Tests whether cache priming improves the model's ability to *discriminate* the
correct answer from plausible distractors (a confound-resistant measure of
representation usefulness), versus merely reducing output entropy.

Key controls: `random_docwords` (salience vs repetition) and length-matching at L=16.""")


# =====================================================================
# Cell 1: Setup
# =====================================================================
code(r"""import os
os.umask(0o000)
import sys
sys.path.insert(0, "../..")
sys.path.insert(0, "../../../directed_kvcache_v4")

import json, time, gc, shutil, math
import random as pyrandom
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from dotenv import load_dotenv, find_dotenv

from lib.rope import select_kv_cache, reposition_kv_cache
from lib.cache import deep_copy_cache, make_prefix
from lib.quantization import norm_roundtrip_kv_cache
from lib.analysis import cohens_d
from lib.data import count_words
from model_adapters import (
    build_layer_inv_freqs, get_layer_types, get_model_info, get_sliding_cache_limit
)

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

def purge_hf_cache(model_name):
    slug = "models--" + model_name.replace("/", "--")
    p = os.path.join(HF_CACHE_DIR, slug)
    if os.path.isdir(p):
        gb = sum(os.path.getsize(os.path.join(dp, f))
                 for dp, _, fns in os.walk(p) for f in fns) / 1e9
        shutil.rmtree(p)
        print(f"  Purged cache for {model_name} ({gb:.1f} GB)")

SEED = 42
SMOKE = os.environ.get("SMOKE", "0") == "1"
N_EVAL = 5 if SMOKE else 300
K_DISTRACT = 3 if SMOKE else 7
L_MATCH = 16

MODELS = {
    'qwen25_1_5b': {'name': 'Qwen/Qwen2.5-1.5B-Instruct', 'loader': 'AutoModelForCausalLM'},
    'qwen25_7b':   {'name': 'Qwen/Qwen2.5-7B-Instruct',   'loader': 'AutoModelForCausalLM'},
    'mistral_7b':  {'name': 'mistralai/Mistral-7B-Instruct-v0.3', 'loader': 'AutoModelForCausalLM'},
    'gemma3_12b':  {'name': 'google/gemma-3-12b-it', 'loader': 'Gemma3ForConditionalGeneration'},
    'ministral_8b':{'name': 'mistralai/Ministral-8B-Instruct-2410', 'loader': 'AutoModelForCausalLM'},
}
if SMOKE:
    MODELS = {'qwen25_1_5b': MODELS['qwen25_1_5b']}

DATASETS = ['squad_v2', 'hotpotqa', 'triviaqa', 'gsm8k']
if SMOKE:
    DATASETS = ['squad_v2']

GENERIC_INSTRUCTION = "Extract the key facts from this text."

RESULTS_BASE = Path("../../results/exp05_discrimination")
RESULTS_BASE.mkdir(parents=True, exist_ok=True, mode=0o777)

# TF-IDF keywords precomputed in the ablation experiment (per dataset, per index)
PREFIXES_PATH = Path("../02_ablation/generated_prefixes.json")
assert PREFIXES_PATH.exists(), f"Run 02_ablation/generate_prefixes.py first: {PREFIXES_PATH}"
TFIDF_KEYWORDS = json.loads(PREFIXES_PATH.read_text())['tfidf_keywords']

print(f"SMOKE={SMOKE}  N_EVAL={N_EVAL}  K_DISTRACT={K_DISTRACT}  L_MATCH={L_MATCH}")
print(f"Models: {list(MODELS)}")
print(f"Datasets: {DATASETS}")
""")


# =====================================================================
# Cell 2: Dataset loading (IDENTICAL seeds/filters to ablation so TF-IDF indices align)
# =====================================================================
code(r"""print("Loading datasets (same seeds/filters as ablation sweep)...")
N_SAMPLES = 400  # load 400 to match generated_prefixes ordering, then use first N_EVAL
all_samples = {}

# --- SQuAD v2 ---
ds = load_dataset("rajpurkar/squad_v2", split="validation")
cand = []
for item in ds:
    passage = item.get('context', ''); query = item.get('question', '')
    answers = item.get('answers', {}).get('text', [])
    answer = answers[0] if answers else ''
    if passage and query and answer:
        wc = count_words(passage)
        if 30 <= wc <= 500:
            cand.append({'passage': passage, 'query': query, 'answer': answer, 'passage_words': wc})
pyrandom.seed(SEED + 200); pyrandom.shuffle(cand)
all_samples['squad_v2'] = cand[:N_SAMPLES]
del ds, cand; gc.collect()

# --- TriviaQA ---
ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation")
cand = []
for item in ds:
    ep = item.get('entity_pages', {}); wc_ctx = ep.get('wiki_context', [])
    if not wc_ctx or not wc_ctx[0]: continue
    passage = ' '.join(wc_ctx[0].split()[:500])
    query = item['question']; answer_val = item['answer']['value']
    aliases = item['answer'].get('aliases', [])
    pl = passage.lower()
    found = answer_val.lower() in pl or any(a.lower() in pl for a in aliases)
    if not found: continue
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer_val) >= 1:
        cand.append({'passage': passage, 'query': query, 'answer': answer_val,
                     'passage_words': wc, 'aliases': aliases})
pyrandom.seed(SEED + 300); pyrandom.shuffle(cand)
all_samples['triviaqa'] = cand[:N_SAMPLES]
del ds, cand; gc.collect()

# --- HotpotQA ---
ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
cand = []
for item in ds:
    ctx = item.get('context', {}); sf = item.get('supporting_facts', {})
    t2s = {t: s for t, s in zip(ctx.get('title', []), ctx.get('sentences', []))}
    parts = [t2s[t][sid] for t, sid in zip(sf.get('title', []), sf.get('sent_id', []))
             if t in t2s and sid < len(t2s[t])]
    if not parts: continue
    passage = ' '.join(parts); query = item['question']; answer = item['answer']
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        cand.append({'passage': passage, 'query': query, 'answer': answer, 'passage_words': wc})
pyrandom.seed(SEED + 400); pyrandom.shuffle(cand)
all_samples['hotpotqa'] = cand[:N_SAMPLES]
del ds, cand; gc.collect()

# --- GSM8K ---
ds = load_dataset("openai/gsm8k", "main", split="test")
cand = []
for item in ds:
    if '####' not in item['answer']: continue
    answer = item['answer'].split('####')[-1].strip()
    if not answer: continue
    passage = item['question']; wc = count_words(passage)
    if 10 <= wc <= 500:
        cand.append({'passage': passage, 'query': "What is the answer?", 'answer': answer, 'passage_words': wc})
pyrandom.seed(SEED + 600); pyrandom.shuffle(cand)
all_samples['gsm8k'] = cand[:N_SAMPLES]
del ds, cand; gc.collect()

for k in DATASETS:
    print(f"  {k}: {len(all_samples[k])} loaded (using first {N_EVAL})")
""")


# =====================================================================
# Cell 3: Distractor pools + random-docword selection
# =====================================================================
code(r"""# Build the distractor pool (all gold answers per dataset) and a normalizer.
import string, re as _re

def normalize_answer(s):
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = _re.sub(r"\b(a|an|the)\b", " ", s)
    return ' '.join(s.split())

DISTRACTOR_POOL = {ds: [s['answer'] for s in all_samples[ds]] for ds in DATASETS}

def _answer_type(a):
    # Type signature: (is_numeric, length_bucket). Used to pick type-matched
    # distractors so the discrimination metric measures document-grounded
    # answer selection, not trivial type plausibility (a date vs an entity).
    a = a.strip()
    is_num = bool(a) and (a[0].isdigit() or (a[0] == '-' and len(a) > 1 and a[1].isdigit()))
    ntok = len(a.split())
    lb = 0 if ntok <= 1 else (1 if ntok <= 3 else 2)
    return (is_num, lb)

TYPE_INDEX = {}
for _ds in DATASETS:
    ti = {}
    for _j, _a in enumerate(DISTRACTOR_POOL[_ds]):
        ti.setdefault(_answer_type(_a), []).append(_j)
    TYPE_INDEX[_ds] = ti

def pick_distractors(ds_key, idx, correct, aliases=None):
    # Pick K_DISTRACT type-matched, plausible-but-wrong answers from the same
    # dataset's gold answers. Type-matching (numeric vs alpha, length bucket)
    # makes distractors competitive so the margin reflects real discrimination.
    pool = DISTRACTOR_POOL[ds_key]
    bad = {normalize_answer(correct)}
    if aliases:
        bad |= {normalize_answer(a) for a in aliases}
    typ = _answer_type(correct)
    bucket = TYPE_INDEX[ds_key].get(typ, [])
    # Fall back to the full pool if the type bucket is too small to fill K.
    candidates = bucket if len(bucket) > K_DISTRACT * 3 else list(range(len(pool)))
    rng = pyrandom.Random(SEED + 7000 + idx)
    order = candidates[:]
    rng.shuffle(order)
    out = []
    for j in order:
        if j == idx:
            continue
        c = pool[j]
        nc = normalize_answer(c)
        if nc in bad or not nc:
            continue
        out.append(c)
        bad.add(nc)
        if len(out) >= K_DISTRACT:
            break
    return out

# Tokenizer-agnostic word filter, IDENTICAL to the TF-IDF generator's tokenize_simple,
# so random_docwords draws from the SAME pool TF-IDF selected from.
# chr(34) is the double-quote char (avoids quote-in-rawstring issues).
_STRIP = ".,;:!?" + chr(34) + "'()[]{}"
def tokenize_simple(text):
    return [w.strip(_STRIP) for w in text.lower().split() if len(w) > 2]

def random_docwords(passage, idx, k=10):
    # Pick k random DISTINCT words from the same filtered pool TF-IDF draws from.
    distinct = sorted(set(tokenize_simple(passage)))
    if not distinct:
        return passage[:50]
    rng = pyrandom.Random(SEED + 13000 + idx)
    rng.shuffle(distinct)
    return ' '.join(distinct[:k])

print("Distractor pools + random-docword selection ready.")
for ds in DATASETS:
    ex_d = pick_distractors(ds, 0, all_samples[ds][0]['answer'],
                            all_samples[ds][0].get('aliases'))
    print(f"  {ds}: correct={all_samples[ds][0]['answer'][:40]!r}  distractors={[d[:20] for d in ex_d]}")
""")


# =====================================================================
# Cell 4: Scoring functions
# =====================================================================
code(r"""_model = _tokenizer = _device = None
_layer_inv_freqs = _layer_types = _sliding_limit = _bos_id = _nl_ids = None

def _encode_phase_a(doc_ids, prefix_ids=None, apply_norm=True):
    input_ids = [_bos_id]
    if prefix_ids is not None:
        input_ids += list(prefix_ids) + _nl_ids
    input_ids += list(doc_ids)
    out = _model(input_ids=torch.tensor([input_ids], device=_device), use_cache=True)
    cache = out.past_key_values
    if prefix_ids is not None:
        doc_start = 1 + len(prefix_ids) + len(_nl_ids)
    else:
        doc_start = 1
    D = len(doc_ids)
    keep = [0] + list(range(doc_start, doc_start + D))
    if _sliding_limit is not None and len(keep) > _sliding_limit:
        raise ValueError(f"Cache overflow: {len(keep)} > {_sliding_limit}")
    cache = select_kv_cache(cache, keep, device=_device)
    if prefix_ids is not None:
        old_pos = torch.arange(doc_start, doc_start + D, device=_device)
        new_pos = torch.arange(1, 1 + D, device=_device)
        cache = reposition_kv_cache(cache, old_pos, new_pos,
                                    _layer_inv_freqs, _layer_types, bos_start=0)
    if apply_norm:
        cache = norm_roundtrip_kv_cache(cache)
    return cache, D

def _score_candidate(cache, D, query_ids, cand_ids):
    # Return (mean_nll, first_token_nll) for one candidate answer.
    pb = _nl_ids + list(query_ids) + _nl_ids + list(cand_ids)
    n = len(pb)
    pos = torch.arange(D + 1, D + 1 + n, device=_device).unsqueeze(0)
    cc = deep_copy_cache(cache)
    out = _model(input_ids=torch.tensor([pb], device=_device),
                 position_ids=pos, past_key_values=cc, use_cache=False)
    logits = out.logits[0]
    a0 = len(_nl_ids) + len(query_ids) + len(_nl_ids)
    al = logits[a0 - 1 : a0 - 1 + len(cand_ids)]
    tgt = torch.tensor(cand_ids, device=_device)
    per_tok = torch.nn.functional.cross_entropy(al, tgt, reduction='none')
    return per_tok.mean().item(), per_tok[0].item()

def score_condition(cache, D, query_ids, correct_ids, distractor_id_list):
    # Score correct + distractors against one cache; return metrics dict.
    c_mean, c_first = _score_candidate(cache, D, query_ids, correct_ids)
    d_means, d_firsts = [], []
    for did in distractor_id_list:
        dm, df = _score_candidate(cache, D, query_ids, did)
        d_means.append(dm); d_firsts.append(df)
    d_means = np.array(d_means); d_firsts = np.array(d_firsts)
    rank = 1 + int((d_means < c_mean).sum())
    return {
        'nll_correct': c_mean,
        'nll_correct_first': c_first,
        'margin_mean': float(d_means.mean() - c_mean),
        'margin_first': float(d_firsts.mean() - c_first),
        'rank': rank,
        'top1': int(rank == 1),
    }

print("Scoring functions defined.")
""")


# =====================================================================
# Cell 5: Main loop
# =====================================================================
code(r"""CONDITIONS = ['bare', 'tfidf_16', 'random_docwords_16', 'random_vocab_16',
              'generic_instr_16', 'oracle_16', 'tfidf_4', 'tfidf_64']

for model_key, spec in MODELS.items():
    print(f"\n{'#'*70}\n# {model_key} ({spec['name']})\n{'#'*70}")
    model_dir = RESULTS_BASE / model_key
    model_dir.mkdir(exist_ok=True, mode=0o777)
    scoring_key = f"discrim_{model_key}" + ("_smoke" if SMOKE else "")

    global _model, _tokenizer, _device, _layer_inv_freqs, _layer_types
    global _sliding_limit, _bos_id, _nl_ids

    _tokenizer = AutoTokenizer.from_pretrained(spec['name'], token=HF_TOKEN)
    loader = spec.get('loader', 'AutoModelForCausalLM')
    if loader == 'Gemma3ForConditionalGeneration':
        from transformers import Gemma3ForConditionalGeneration
        _model = Gemma3ForConditionalGeneration.from_pretrained(
            spec['name'], dtype=torch.bfloat16, token=HF_TOKEN, device_map='cuda:0').eval()
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            spec['name'], dtype=torch.bfloat16, token=HF_TOKEN, device_map='cuda:0').eval()

    _device = next(_model.parameters()).device
    _layer_inv_freqs = build_layer_inv_freqs(_model, device=_device)
    _layer_types = get_layer_types(_model)
    _sliding_limit = get_sliding_cache_limit(_model)
    _nl_ids = _tokenizer.encode("\n", add_special_tokens=False)
    _bos_id = _tokenizer.bos_token_id if _tokenizer.bos_token_id is not None else _tokenizer.pad_token_id

    info = get_model_info(_model)
    max_doc = (_sliding_limit - 1 - 64 - len(_nl_ids)) if _sliding_limit is not None else 765
    generic_ids = _tokenizer.encode(GENERIC_INSTRUCTION, add_special_tokens=False)
    print(f"  Loaded: {info['num_layers']} layers, head_dim={info['head_dim']}, max_doc={max_doc}")

    for ds_key in DATASETS:
        print(f"\n  --- {ds_key} ---")
        samples = all_samples[ds_key][:N_EVAL]
        ckpt = model_dir / f"checkpoint_{ds_key}.json"
        scored = []
        if ckpt.exists():
            prev = json.loads(ckpt.read_text())
            if prev.get('scoring_key') == scoring_key:
                scored = prev['samples']
                print(f"  Resumed {len(scored)}/{len(samples)}")

        for idx in range(len(scored), len(samples)):
            s = samples[idx]
            doc_ids = _tokenizer.encode(s['passage'], add_special_tokens=False)[:max_doc]
            query_ids = _tokenizer.encode(s['query'], add_special_tokens=False)
            correct_ids = _tokenizer.encode(s['answer'], add_special_tokens=False)
            if not correct_ids:
                continue
            D = len(doc_ids)

            distractors = pick_distractors(ds_key, idx, s['answer'], s.get('aliases'))
            distractor_ids = [_tokenizer.encode(d, add_special_tokens=False) for d in distractors]
            distractor_ids = [d for d in distractor_ids if d]  # drop empties

            # Build the prefixes for this sample (length-matched where applicable)
            rng = pyrandom.Random(SEED * 10007 + idx)
            tfidf_text = TFIDF_KEYWORDS[ds_key][idx]
            tfidf_ids = _tokenizer.encode(tfidf_text, add_special_tokens=False)
            rdw_text = random_docwords(s['passage'], idx, k=10)
            rdw_ids = _tokenizer.encode(rdw_text, add_special_tokens=False)
            rand_vocab = [rng.randint(100, _tokenizer.vocab_size - 1) for _ in range(L_MATCH)]

            prefixes = {
                'bare': None,
                'tfidf_16': make_prefix(tfidf_ids, L_MATCH) if tfidf_ids else None,
                'random_docwords_16': make_prefix(rdw_ids, L_MATCH) if rdw_ids else None,
                'random_vocab_16': rand_vocab,
                'generic_instr_16': make_prefix(generic_ids, L_MATCH),
                'oracle_16': make_prefix(query_ids, L_MATCH) if query_ids else None,
                'tfidf_4': make_prefix(tfidf_ids, 4) if tfidf_ids else None,
                'tfidf_64': make_prefix(tfidf_ids, 64) if tfidf_ids else None,
            }

            result = {'query': s['query'][:200], 'answer': s['answer'][:200],
                      'passage_words': s['passage_words'], 'n_distractors': len(distractor_ids)}

            with torch.no_grad():
                for cond in CONDITIONS:
                    pfx = prefixes[cond]
                    if cond != 'bare' and pfx is None:
                        continue
                    cache, D = _encode_phase_a(doc_ids, prefix_ids=pfx)
                    m = score_condition(cache, D, query_ids, correct_ids, distractor_ids)
                    for k, v in m.items():
                        result[f"{cond}__{k}"] = v
                    del cache
            scored.append(result)
            torch.cuda.empty_cache()

            if (idx + 1) % 20 == 0 or SMOKE:
                ckpt.write_text(json.dumps({'scoring_key': scoring_key, 'samples': scored}))
                bm = np.mean([x['bare__margin_mean'] for x in scored])
                km = np.mean([x['tfidf_16__margin_mean'] for x in scored if 'tfidf_16__margin_mean' in x])
                print(f"    [{idx+1}/{len(samples)}] bare margin={bm:+.3f}, tfidf16 margin={km:+.3f}")

        ckpt.write_text(json.dumps({'scoring_key': scoring_key, 'samples': scored}))
        print(f"  {ds_key}: {len(scored)} scored")

    del _model, _tokenizer; _model = _tokenizer = None
    gc.collect(); torch.cuda.empty_cache()
    purge_hf_cache(spec['name'])
    print("  Model unloaded.")

print(f"\n{'='*70}\nALL MODELS COMPLETE\n{'='*70}")
""")


# =====================================================================
# Cell 6: Analysis
# =====================================================================
code(r"""# Core analysis: does priming improve DISCRIMINATION (margin), not just confidence (nll)?
def cd(a):
    a = np.asarray(a, dtype=float)
    return a.mean() / (a.std(ddof=1) + 1e-12) if len(a) > 1 else 0.0

PRIMED = ['tfidf_16', 'random_docwords_16', 'random_vocab_16',
          'generic_instr_16', 'oracle_16', 'tfidf_4', 'tfidf_64']

print("CONTRASTIVE DISCRIMINATION RESULTS")
print("  d(nll)    = effect on absolute answer NLL   (old metric; entropy-sensitive)")
print("  d(margin) = effect on contrastive margin    (new metric; entropy-invariant)")
print("  If d(nll) >> d(margin), the effect is largely entropy reduction, not discrimination.\n")

for model_key in MODELS:
    md = RESULTS_BASE / model_key
    print(f"\n{'='*78}\n{model_key}\n{'='*78}")
    print(f"{'condition':<20s} | {'d(nll)':>7s} {'d(margin)':>9s} {'d(mgn1st)':>9s} | {'top1Δ':>6s}")
    print("-" * 62)
    # pool across datasets
    per_cond = {c: {'dnll': [], 'dmgn': [], 'dmgn1': [], 'dtop1': []} for c in PRIMED}
    for ds_key in DATASETS:
        ck = md / f"checkpoint_{ds_key}.json"
        if not ck.exists(): continue
        samples = json.loads(ck.read_text())['samples']
        for c in PRIMED:
            for s in samples:
                if f"{c}__nll_correct" not in s: continue
                # paired: bare - condition (positive = priming helps)
                per_cond[c]['dnll'].append(s['bare__nll_correct'] - s[f"{c}__nll_correct"])
                per_cond[c]['dmgn'].append(s[f"{c}__margin_mean"] - s['bare__margin_mean'])
                per_cond[c]['dmgn1'].append(s[f"{c}__margin_first"] - s['bare__margin_first'])
                per_cond[c]['dtop1'].append(s[f"{c}__top1"] - s['bare__top1'])
    for c in PRIMED:
        d = per_cond[c]
        if not d['dnll']: continue
        print(f"{c:<20s} | {cd(d['dnll']):>+7.2f} {cd(d['dmgn']):>+9.2f} {cd(d['dmgn1']):>+9.2f} | "
              f"{np.mean(d['dtop1']):>+6.1%}")

print("\n\nKEY CONTRASTS (pooled across models & datasets):")
print("  tfidf vs random_docwords  -> salience vs in-document repetition")
print("  random_docwords vs random_vocab -> document relevance vs generic token presence")
allc = {c: [] for c in PRIMED}
for model_key in MODELS:
    md = RESULTS_BASE / model_key
    for ds_key in DATASETS:
        ck = md / f"checkpoint_{ds_key}.json"
        if not ck.exists(): continue
        for s in json.loads(ck.read_text())['samples']:
            for c in PRIMED:
                if f"{c}__margin_mean" in s:
                    allc[c].append(s[f"{c}__margin_mean"] - s['bare__margin_mean'])
print()
for c in PRIMED:
    if allc[c]:
        print(f"  {c:<20s} d(margin) = {cd(allc[c]):+.3f}  (n={len(allc[c])})")
""")


out_path = "experiments/04_discrimination/04_discrimination_sweep.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"Notebook written to {out_path}")
print(f"Cells: {len(nb.cells)} ({sum(1 for c in nb.cells if c.cell_type=='markdown')} md, "
      f"{sum(1 for c in nb.cells if c.cell_type=='code')} code)")
