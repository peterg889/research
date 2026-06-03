#!/usr/bin/env python3
"""Confirmatory sweep: is the discrimination gain the INSTRUCTION FORM or one sentence?

The discrimination sweep (exp05) found that the ONLY prefix robustly improving
the contrastive margin was a single generic instruction,
"Extract the key facts from this text" (d(margin)=+0.27 pooled, positive for all
5 models). This sweep tests whether that is a property of *task instructions in
general* or an artifact of that specific sentence.

We score 8 instruction phrasings that vary systematically:
  imperative, "facts":      "Extract the key facts from this text."
  imperative, "information": "Identify the most important information in this passage."
  imperative, "summarize":   "Summarize the following text."
  imperative, "retrieval":   "Index the following information for retrieval."
  imperative, "attention":   "Pay close attention to the details in this text."
  imperative, "comprehend":  "Read and comprehend this text carefully."
  interrogative:             "What are the key facts in this text?"
  declarative (non-command): "This text contains important information."

Plus two anchors (identical seeds/distractors to exp05, so the scale is directly
comparable):
  random_docwords_16  repetition anchor (exp05 margin ~= 0)
  random_vocab_16     harmful anchor    (exp05 margin < 0)

All prefixes length-matched to L=16. Same metric (contrastive margin vs K=7
type-matched distractors), same 5 models x 4 datasets x 300 samples.

Verdict logic:
  - If MOST instructions show positive margin -> the effect is the instruction
    FORM (robust, generalizable finding).
  - If only "extract key facts" is positive -> fragile, sentence-specific.
  - Imperative vs interrogative vs declarative contrast localizes the mechanism.

Usage:
    cd /home/jupyter/research/directed_kvcache_publication
    python3 experiments/05_instruction_confirmatory/build_instruction_confirmatory.py
    cd experiments/05_instruction_confirmatory
    SMOKE=1 papermill 05_instruction_confirmatory.ipynb 05_smoke.ipynb --no-progress-bar
    papermill 05_instruction_confirmatory.ipynb 05_instruction_confirmatory_executed.ipynb --no-progress-bar
"""

import os
import ast
import nbformat as nbf

os.makedirs("experiments/05_instruction_confirmatory", exist_ok=True)

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}


def md(s):
    nb.cells.append(nbf.v4.new_markdown_cell(s))


def code(s):
    try:
        ast.parse(s)
    except SyntaxError as e:
        raise SyntaxError(f"Cell syntax error line {e.lineno}: {e.msg}\n  {e.text}") from e
    nb.cells.append(nbf.v4.new_code_cell(s))


md(r"""# Instruction-Form Confirmatory Sweep

Tests whether the contrastive-discrimination gain from a generic instruction is a
property of task instructions in general (robust) or of one specific sentence
(fragile). 8 phrasings + 2 anchors, scored with the same contrastive-margin
harness as exp05.""")


# ---- Cell 1: setup ----
code(r"""import os
os.umask(0o000)
import sys
sys.path.insert(0, "../..")
sys.path.insert(0, "../../../directed_kvcache_v4")

import json, time, gc, shutil, string
import re as _re
import random as pyrandom
from pathlib import Path

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
from model_adapters import (build_layer_inv_freqs, get_layer_types,
                            get_model_info, get_sliding_cache_limit)

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

def purge_hf_cache(name):
    slug = "models--" + name.replace("/", "--")
    p = os.path.join(HF_CACHE_DIR, slug)
    if os.path.isdir(p):
        gb = sum(os.path.getsize(os.path.join(dp, f))
                 for dp, _, fns in os.walk(p) for f in fns) / 1e9
        shutil.rmtree(p); print(f"  Purged {name} ({gb:.1f} GB)")

SEED = 42
SMOKE = os.environ.get("SMOKE", "0") == "1"
N_EVAL = 5 if SMOKE else 300
K_DISTRACT = 3 if SMOKE else 7
L_MATCH = 16

# Instruction phrasings under test (length-matched to L_MATCH).
# Keys become condition names: instr_<key>.
INSTRUCTIONS = {
    'extract':     "Extract the key facts from this text.",
    'identify':    "Identify the most important information in this passage.",
    'summarize':   "Summarize the following text.",
    'index':       "Index the following information for retrieval.",
    'attend':      "Pay close attention to the details in this text.",
    'comprehend':  "Read and comprehend this text carefully.",
    'question':    "What are the key facts in this text?",
    'declarative': "This text contains important information.",
}
# Mood tags for analysis
MOOD = {'extract':'imperative','identify':'imperative','summarize':'imperative',
        'index':'imperative','attend':'imperative','comprehend':'imperative',
        'question':'interrogative','declarative':'declarative'}

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

RESULTS_BASE = Path("../../results/exp06_instruction_confirmatory")
RESULTS_BASE.mkdir(parents=True, exist_ok=True, mode=0o777)

print(f"SMOKE={SMOKE} N_EVAL={N_EVAL} K={K_DISTRACT} L={L_MATCH}")
print(f"Instructions: {list(INSTRUCTIONS)}")
print(f"Models: {list(MODELS)}  Datasets: {DATASETS}")
""")


# ---- Cell 2: datasets (identical seeds/filters to exp05) ----
code(r"""print("Loading datasets (identical seeds/filters to exp05)...")
N_SAMPLES = 400
all_samples = {}

ds = load_dataset("rajpurkar/squad_v2", split="validation")
cand = []
for item in ds:
    passage = item.get('context',''); query = item.get('question','')
    answers = item.get('answers',{}).get('text',[]); answer = answers[0] if answers else ''
    if passage and query and answer:
        wc = count_words(passage)
        if 30 <= wc <= 500:
            cand.append({'passage':passage,'query':query,'answer':answer,'passage_words':wc})
pyrandom.seed(SEED+200); pyrandom.shuffle(cand); all_samples['squad_v2']=cand[:N_SAMPLES]
del ds,cand; gc.collect()

ds = load_dataset("mandarjoshi/trivia_qa","rc.wikipedia",split="validation")
cand = []
for item in ds:
    ep = item.get('entity_pages',{}); wc_ctx = ep.get('wiki_context',[])
    if not wc_ctx or not wc_ctx[0]: continue
    passage = ' '.join(wc_ctx[0].split()[:500])
    query = item['question']; answer_val = item['answer']['value']
    aliases = item['answer'].get('aliases',[]); pl = passage.lower()
    if not (answer_val.lower() in pl or any(a.lower() in pl for a in aliases)): continue
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer_val) >= 1:
        cand.append({'passage':passage,'query':query,'answer':answer_val,'passage_words':wc,'aliases':aliases})
pyrandom.seed(SEED+300); pyrandom.shuffle(cand); all_samples['triviaqa']=cand[:N_SAMPLES]
del ds,cand; gc.collect()

ds = load_dataset("hotpotqa/hotpot_qa","distractor",split="validation")
cand = []
for item in ds:
    ctx = item.get('context',{}); sf = item.get('supporting_facts',{})
    t2s = {t:s for t,s in zip(ctx.get('title',[]), ctx.get('sentences',[]))}
    parts = [t2s[t][sid] for t,sid in zip(sf.get('title',[]),sf.get('sent_id',[])) if t in t2s and sid < len(t2s[t])]
    if not parts: continue
    passage = ' '.join(parts); query = item['question']; answer = item['answer']
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        cand.append({'passage':passage,'query':query,'answer':answer,'passage_words':wc})
pyrandom.seed(SEED+400); pyrandom.shuffle(cand); all_samples['hotpotqa']=cand[:N_SAMPLES]
del ds,cand; gc.collect()

ds = load_dataset("openai/gsm8k","main",split="test")
cand = []
for item in ds:
    if '####' not in item['answer']: continue
    answer = item['answer'].split('####')[-1].strip()
    if not answer: continue
    passage = item['question']; wc = count_words(passage)
    if 10 <= wc <= 500:
        cand.append({'passage':passage,'query':"What is the answer?",'answer':answer,'passage_words':wc})
pyrandom.seed(SEED+600); pyrandom.shuffle(cand); all_samples['gsm8k']=cand[:N_SAMPLES]
del ds,cand; gc.collect()

for k in DATASETS: print(f"  {k}: {len(all_samples[k])} (using first {N_EVAL})")
""")


# ---- Cell 3: distractors (identical to exp05) ----
code(r"""def normalize_answer(s):
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = _re.sub(r"\b(a|an|the)\b", " ", s)
    return ' '.join(s.split())

DISTRACTOR_POOL = {ds: [s['answer'] for s in all_samples[ds]] for ds in DATASETS}

def _answer_type(a):
    a = a.strip()
    is_num = bool(a) and (a[0].isdigit() or (a[0]=='-' and len(a)>1 and a[1].isdigit()))
    ntok = len(a.split()); lb = 0 if ntok<=1 else (1 if ntok<=3 else 2)
    return (is_num, lb)

TYPE_INDEX = {}
for _ds in DATASETS:
    ti = {}
    for _j,_a in enumerate(DISTRACTOR_POOL[_ds]): ti.setdefault(_answer_type(_a),[]).append(_j)
    TYPE_INDEX[_ds] = ti

def pick_distractors(ds_key, idx, correct, aliases=None):
    pool = DISTRACTOR_POOL[ds_key]
    bad = {normalize_answer(correct)}
    if aliases: bad |= {normalize_answer(a) for a in aliases}
    bucket = TYPE_INDEX[ds_key].get(_answer_type(correct), [])
    candidates = bucket if len(bucket) > K_DISTRACT*3 else list(range(len(pool)))
    rng = pyrandom.Random(SEED+7000+idx); order = candidates[:]; rng.shuffle(order)
    out = []
    for j in order:
        if j == idx: continue
        c = pool[j]; nc = normalize_answer(c)
        if nc in bad or not nc: continue
        out.append(c); bad.add(nc)
        if len(out) >= K_DISTRACT: break
    return out

_STRIP = ".,;:!?" + chr(34) + "'()[]{}"
def tokenize_simple(text):
    return [w.strip(_STRIP) for w in text.lower().split() if len(w) > 2]

def random_docwords(passage, idx, k=10):
    distinct = sorted(set(tokenize_simple(passage)))
    if not distinct: return passage[:50]
    rng = pyrandom.Random(SEED+13000+idx); rng.shuffle(distinct)
    return ' '.join(distinct[:k])

print("Distractor pools ready.")
""")


# ---- Cell 4: scoring (identical to exp05) ----
code(r"""_model=_tokenizer=_device=None
_layer_inv_freqs=_layer_types=_sliding_limit=_bos_id=_nl_ids=None

def _encode_phase_a(doc_ids, prefix_ids=None, apply_norm=True):
    input_ids = [_bos_id]
    if prefix_ids is not None: input_ids += list(prefix_ids) + _nl_ids
    input_ids += list(doc_ids)
    out = _model(input_ids=torch.tensor([input_ids], device=_device), use_cache=True)
    cache = out.past_key_values
    doc_start = (1 + len(prefix_ids) + len(_nl_ids)) if prefix_ids is not None else 1
    D = len(doc_ids); keep = [0] + list(range(doc_start, doc_start+D))
    if _sliding_limit is not None and len(keep) > _sliding_limit:
        raise ValueError(f"overflow {len(keep)}>{_sliding_limit}")
    cache = select_kv_cache(cache, keep, device=_device)
    if prefix_ids is not None:
        cache = reposition_kv_cache(cache, torch.arange(doc_start,doc_start+D,device=_device),
                                    torch.arange(1,1+D,device=_device),
                                    _layer_inv_freqs,_layer_types,bos_start=0)
    if apply_norm: cache = norm_roundtrip_kv_cache(cache)
    return cache, D

def _score_candidate(cache, D, query_ids, cand_ids):
    pb = _nl_ids + list(query_ids) + _nl_ids + list(cand_ids); n = len(pb)
    pos = torch.arange(D+1, D+1+n, device=_device).unsqueeze(0)
    out = _model(input_ids=torch.tensor([pb],device=_device), position_ids=pos,
                 past_key_values=deep_copy_cache(cache), use_cache=False)
    a0 = len(_nl_ids)+len(query_ids)+len(_nl_ids)
    al = out.logits[0][a0-1:a0-1+len(cand_ids)]
    per = torch.nn.functional.cross_entropy(al, torch.tensor(cand_ids,device=_device), reduction='none')
    return per.mean().item(), per[0].item()

def score_condition(cache, D, query_ids, correct_ids, distractor_id_list):
    cm, cf = _score_candidate(cache, D, query_ids, correct_ids)
    dms, dfs = [], []
    for did in distractor_id_list:
        m,f = _score_candidate(cache, D, query_ids, did); dms.append(m); dfs.append(f)
    dms = np.array(dms); dfs = np.array(dfs)
    return {'nll_correct':cm,'nll_correct_first':cf,
            'margin_mean':float(dms.mean()-cm),'margin_first':float(dfs.mean()-cf),
            'rank':1+int((dms<cm).sum()),'top1':int((dms<cm).sum()==0)}
print("Scoring funcs ready.")
""")


# ---- Cell 5: main loop ----
code(r"""INSTR_CONDS = [f"instr_{k}" for k in INSTRUCTIONS]
CONDITIONS = INSTR_CONDS + ['random_docwords_16', 'random_vocab_16']

for model_key, spec in MODELS.items():
    print(f"\n{'#'*70}\n# {model_key} ({spec['name']})\n{'#'*70}")
    model_dir = RESULTS_BASE / model_key; model_dir.mkdir(exist_ok=True, mode=0o777)
    scoring_key = f"instrconf_{model_key}" + ("_smoke" if SMOKE else "")

    global _model,_tokenizer,_device,_layer_inv_freqs,_layer_types,_sliding_limit,_bos_id,_nl_ids
    _tokenizer = AutoTokenizer.from_pretrained(spec['name'], token=HF_TOKEN)
    if spec['loader'] == 'Gemma3ForConditionalGeneration':
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
    max_doc = (_sliding_limit-1-64-len(_nl_ids)) if _sliding_limit is not None else 765

    instr_ids = {k: _tokenizer.encode(v, add_special_tokens=False) for k,v in INSTRUCTIONS.items()}
    print(f"  Loaded: {info['num_layers']} layers, max_doc={max_doc}")

    for ds_key in DATASETS:
        print(f"\n  --- {ds_key} ---")
        samples = all_samples[ds_key][:N_EVAL]
        ckpt = model_dir / f"checkpoint_{ds_key}.json"; scored = []
        if ckpt.exists():
            prev = json.loads(ckpt.read_text())
            if prev.get('scoring_key') == scoring_key:
                scored = prev['samples']; print(f"  Resumed {len(scored)}/{len(samples)}")

        for idx in range(len(scored), len(samples)):
            s = samples[idx]
            doc_ids = _tokenizer.encode(s['passage'], add_special_tokens=False)[:max_doc]
            query_ids = _tokenizer.encode(s['query'], add_special_tokens=False)
            correct_ids = _tokenizer.encode(s['answer'], add_special_tokens=False)
            if not correct_ids: continue
            D = len(doc_ids)
            distractors = pick_distractors(ds_key, idx, s['answer'], s.get('aliases'))
            distractor_ids = [d for d in (_tokenizer.encode(x, add_special_tokens=False) for x in distractors) if d]

            rng = pyrandom.Random(SEED*10007 + idx)
            rdw_ids = _tokenizer.encode(random_docwords(s['passage'], idx, 10), add_special_tokens=False)
            rand_vocab = [rng.randint(100, _tokenizer.vocab_size-1) for _ in range(L_MATCH)]

            prefixes = {f"instr_{k}": make_prefix(instr_ids[k], L_MATCH) for k in INSTRUCTIONS}
            prefixes['random_docwords_16'] = make_prefix(rdw_ids, L_MATCH) if rdw_ids else None
            prefixes['random_vocab_16'] = rand_vocab

            result = {'query': s['query'][:200], 'answer': s['answer'][:200],
                      'passage_words': s['passage_words'], 'n_distractors': len(distractor_ids)}
            # bare reference
            with torch.no_grad():
                cache,D = _encode_phase_a(doc_ids, prefix_ids=None)
                for k,v in score_condition(cache,D,query_ids,correct_ids,distractor_ids).items():
                    result[f"bare__{k}"] = v
                del cache
                for cond in CONDITIONS:
                    pfx = prefixes[cond]
                    if pfx is None: continue
                    cache,D = _encode_phase_a(doc_ids, prefix_ids=pfx)
                    for k,v in score_condition(cache,D,query_ids,correct_ids,distractor_ids).items():
                        result[f"{cond}__{k}"] = v
                    del cache
            scored.append(result); torch.cuda.empty_cache()

            if (idx+1) % 20 == 0 or SMOKE:
                ckpt.write_text(json.dumps({'scoring_key':scoring_key,'samples':scored}))
                ex = np.mean([x['instr_extract__margin_mean']-x['bare__margin_mean'] for x in scored])
                sm = np.mean([x['instr_summarize__margin_mean']-x['bare__margin_mean'] for x in scored])
                print(f"    [{idx+1}/{len(samples)}] d_margin extract={ex:+.3f} summarize={sm:+.3f}")

        ckpt.write_text(json.dumps({'scoring_key':scoring_key,'samples':scored}))
        print(f"  {ds_key}: {len(scored)} scored")

    del _model,_tokenizer; _model=_tokenizer=None
    gc.collect(); torch.cuda.empty_cache(); purge_hf_cache(spec['name'])
    print("  unloaded.")

print(f"\n{'='*70}\nALL MODELS COMPLETE\n{'='*70}")
""")


# ---- Cell 6: analysis ----
code(r"""def cd(a):
    a=np.asarray(a,float); return a.mean()/(a.std(ddof=1)+1e-12) if len(a)>1 else 0.0

CONDS = [f"instr_{k}" for k in INSTRUCTIONS] + ['random_docwords_16','random_vocab_16']
print("CONFIRMATORY: d(margin) by instruction phrasing (paired vs bare, pooled across models)")
print("If most instructions are positive -> the effect is the instruction FORM.\n")
print(f"{'condition':<22s} {'mood':<13s} {'d(margin)':>9s} {'d(nll)':>8s}  n_models_pos")
print("-"*68)
pooled = {c: {'m':[], 'n':[]} for c in CONDS}
per_model_pos = {c: 0 for c in CONDS}
for model_key in MODELS:
    md_ = RESULTS_BASE / model_key
    for c in CONDS:
        mm = []
        for ds_key in DATASETS:
            ck = md_ / f"checkpoint_{ds_key}.json"
            if not ck.exists(): continue
            for s in json.loads(ck.read_text())['samples']:
                if f"{c}__margin_mean" in s:
                    pooled[c]['m'].append(s[f"{c}__margin_mean"]-s['bare__margin_mean'])
                    pooled[c]['n'].append(s['bare__nll_correct']-s[f"{c}__nll_correct"])
                    mm.append(s[f"{c}__margin_mean"]-s['bare__margin_mean'])
        if len(mm) > 1 and cd(mm) > 0: per_model_pos[c] += 1
for c in CONDS:
    if not pooled[c]['m']: continue
    mood = MOOD.get(c.replace('instr_',''), 'anchor') if c.startswith('instr_') else 'anchor'
    print(f"{c:<22s} {mood:<13s} {cd(pooled[c]['m']):>+9.3f} {cd(pooled[c]['n']):>+8.3f}  {per_model_pos[c]}/{len(MODELS)}")

# Mood-level summary
print("\nBy mood (mean d(margin) across instruction conditions):")
from collections import defaultdict
mood_vals = defaultdict(list)
for c in CONDS:
    if c.startswith('instr_') and pooled[c]['m']:
        mood_vals[MOOD[c.replace('instr_','')]].append(cd(pooled[c]['m']))
for mood, vals in mood_vals.items():
    print(f"  {mood:<14s}: mean d(margin) = {np.mean(vals):+.3f}  (n_phrasings={len(vals)})")
""")


out = "experiments/05_instruction_confirmatory/05_instruction_confirmatory.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"Notebook written to {out}")
print(f"Cells: {len(nb.cells)}")
