#!/usr/bin/env python3
# Build Exp 09 notebook: Cross-Model Generalization Sweep.
#
# All v4 experiments used T5Gemma 2 4B-4B. Does the enrichment effect
# generalize to other encoder-decoder models? This experiment runs the
# core 4-condition test on multiple models:
#   1. google/flan-t5-base (250M)
#   2. google/flan-t5-large (780M)
#   3. google/flan-t5-xl (3B)
#   4. facebook/bart-large (400M)
#
# NLL-only (no attention probes) so each model runs in minutes.
# Same 500 MS MARCO samples, same random prefixes, same conditions.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 09: Cross-Model Generalization Sweep

## Motivation

All v4 experiments (01-08) used T5Gemma 2 4B-4B. The enrichment effect
(prepending a prefix to the encoder improves answer NLL) could be a T5Gemma
quirk — especially since T5Gemma has unusual merged self+cross attention
in the decoder.

This experiment tests whether the core effect generalizes to:
- **Flan-T5** (base/large/xl) — standard T5 architecture, instruction-tuned
- **BART-large** — different pre-training (denoising AE vs span corruption)

## Design

**4 conditions per model:**

| # | Condition | Encoder input | Cross-attn mask | Decoder input |
|---|-----------|--------------|-----------------|---------------|
| 1 | bare | [document] | all | [BOS, query, answer] |
| 2 | oracle_trunc | [query + doc] | doc only | [BOS, query, answer] |
| 3 | random_trunc | [random + doc] | doc only | [BOS, query, answer] |
| 4 | bare_nq | [document] | all | [BOS, answer] |

**Key comparisons (per model):**
- **(2) vs (1)**: Does encoder enrichment help? (THE generalization test)
- **(3) vs (1)**: Is it structural or content-dependent?
- **(1) vs (4)**: How much does decoder query help?

**Same 500 MS MARCO samples across all models.** N=500, SEED=42.
NLL only — no attention probes (SDPA attention for speed).""")


# ===== Cell 2: Setup =====
code(r"""# Cell 2: Setup
import os
os.umask(0o000)

import sys, json, time, gc, re
import random as pyrandom
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, "../../..")
from lib.analysis import cohens_d

SEED = 42
N_SAMPLES = 500

RESULTS_DIR = Path("../../../results/exp09")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = [
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "facebook/bart-large",
]

print(f"Exp 09: Cross-Model Generalization Sweep")
print(f"N: {N_SAMPLES}, Models: {len(MODELS)}")
print(f"DEVICE: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
""")


# ===== Cell 3: Data loading =====
code(r"""# Cell 3: Load MS MARCO data + generate random prefixes
from lib.data import count_words
from datasets import load_dataset

print("Loading MS MARCO v1.1 validation...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

all_candidates = []
for item in ds:
    if len(all_candidates) >= 3 * N_SAMPLES:
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
            all_candidates.append({
                'passage': pt, 'query': query, 'answer': answer,
                'word_count': wc,
            })
            break

print(f"Total candidates: {len(all_candidates)}")
np.random.seed(SEED)
indices = np.random.permutation(len(all_candidates))
samples = [all_candidates[i] for i in indices[:N_SAMPLES]]
del ds, all_candidates
gc.collect()

# Generate random prefix TEXT for each sample (shared across models)
# 8 random common English words per sample
WORD_POOL = [
    "computer", "mountain", "hospital", "children", "building", "national",
    "business", "research", "students", "american", "possible", "economic",
    "personal", "together", "products", "services", "actually", "remember",
    "practice", "training", "industry", "complete", "critical", "function",
    "language", "standard", "material", "original", "physical", "security",
    "interest", "problems", "consider", "response", "pressure", "politics",
    "movement", "evidence", "southern", "northern", "exchange", "decision",
    "position", "increase", "describe", "military", "required", "approach",
    "strategy", "customer", "resource", "employee", "audience", "location",
    "property", "cultural", "activity", "strength", "analysis", "powerful",
    "election", "argument", "campaign", "maintain", "question", "behavior",
    "majority", "solution", "software", "consumer", "creative", "reaction",
    "european", "delivery", "organize", "involved", "relative", "learning",
    "positive", "numerous", "familiar", "engineer", "platform", "indicate",
    "previous", "pleasure", "opposite", "magazine", "document", "religion",
    "scenario", "workshop", "minority", "guidance", "estimate", "recently",
    "surprise", "champion", "pleasant", "grateful", "moderate", "boundary",
]

for i, s in enumerate(samples):
    rng = np.random.RandomState(SEED + i + 20000)
    words = rng.choice(WORD_POOL, size=8, replace=False)
    s['random_prefix'] = " ".join(words)

print(f"Loaded {len(samples)} samples")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Mean query words: {np.mean([count_words(s['query']) for s in samples]):.0f}")
print(f"Mean answer words: {np.mean([count_words(s['answer']) for s in samples]):.0f}")
print(f"Example random prefix: '{samples[0]['random_prefix']}'")
""")


# ===== Cell 4: Scoring function =====
code(r"""# Cell 4: Model-agnostic scoring function

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def count_prefix_tokens(tokenizer, prefix_text, document_text):
    full_text = prefix_text + "\n" + document_text
    full_ids = tokenizer(full_text, add_special_tokens=True, truncation=True,
                         max_length=2048).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids
    return len(full_ids) - len(doc_ids)


def score_sample(model, tokenizer, sample, device):
    # Score one sample under all 4 conditions.
    # Returns dict with NLL per condition.
    passage = sample['passage']
    query = sample['query']
    answer = sample['answer']
    random_prefix = sample['random_prefix']

    bos_id = model.config.decoder_start_token_id
    if bos_id is None:
        bos_id = tokenizer.pad_token_id or 0

    query_ids = tokenizer(query, add_special_tokens=False, truncation=True,
                          max_length=512).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False, truncation=True,
                           max_length=256).input_ids
    if len(answer_ids) == 0:
        return None

    result = {}

    # === Encoder pass 1: bare document ===
    enc_ids_bare = tokenizer(passage, return_tensors="pt",
                             add_special_tokens=True, truncation=True,
                             max_length=2048).input_ids.to(device)
    enc_mask_bare = torch.ones(1, enc_ids_bare.shape[1], device=device, dtype=torch.long)
    with torch.no_grad():
        enc_out_bare = model.get_encoder()(
            input_ids=enc_ids_bare, attention_mask=enc_mask_bare
        )

    # Condition 1: bare (decoder has query)
    dec_q = torch.tensor([[bos_id] + query_ids + answer_ids],
                          dtype=torch.long, device=device)
    n_q = len(query_ids)
    with torch.no_grad():
        out = model(encoder_outputs=enc_out_bare, attention_mask=enc_mask_bare,
                    decoder_input_ids=dec_q)
    logits = out.logits[0, n_q:n_q + len(answer_ids), :]
    targets = torch.tensor(answer_ids, dtype=torch.long, device=device)
    nll = -F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
    result['nll_bare'] = nll

    # Condition 4: bare_nq (no query in decoder)
    dec_nq = torch.tensor([[bos_id] + answer_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(encoder_outputs=enc_out_bare, attention_mask=enc_mask_bare,
                    decoder_input_ids=dec_nq)
    logits = out.logits[0, 0:len(answer_ids), :]
    nll = -F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
    result['nll_bare_nq'] = nll

    del enc_out_bare

    # === Encoder pass 2: oracle (query + document) ===
    oracle_text = query + "\n" + passage
    enc_ids_oracle = tokenizer(oracle_text, return_tensors="pt",
                               add_special_tokens=True, truncation=True,
                               max_length=2048).input_ids.to(device)
    enc_mask_oracle = torch.ones(1, enc_ids_oracle.shape[1], device=device, dtype=torch.long)
    with torch.no_grad():
        enc_out_oracle = model.get_encoder()(
            input_ids=enc_ids_oracle, attention_mask=enc_mask_oracle
        )

    # Build cross-attention mask (hide prefix)
    n_pfx = count_prefix_tokens(tokenizer, query, passage)
    cross_mask_oracle = torch.ones(1, enc_ids_oracle.shape[1], device=device, dtype=torch.long)
    cross_mask_oracle[:, :n_pfx] = 0

    # Condition 2: oracle_trunc
    with torch.no_grad():
        out = model(encoder_outputs=enc_out_oracle, attention_mask=cross_mask_oracle,
                    decoder_input_ids=dec_q)
    logits = out.logits[0, n_q:n_q + len(answer_ids), :]
    nll = -F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
    result['nll_oracle_trunc'] = nll

    del enc_out_oracle

    # === Encoder pass 3: random prefix ===
    random_text = random_prefix + "\n" + passage
    enc_ids_random = tokenizer(random_text, return_tensors="pt",
                               add_special_tokens=True, truncation=True,
                               max_length=2048).input_ids.to(device)
    enc_mask_random = torch.ones(1, enc_ids_random.shape[1], device=device, dtype=torch.long)
    with torch.no_grad():
        enc_out_random = model.get_encoder()(
            input_ids=enc_ids_random, attention_mask=enc_mask_random
        )

    n_pfx_rand = count_prefix_tokens(tokenizer, random_prefix, passage)
    cross_mask_random = torch.ones(1, enc_ids_random.shape[1], device=device, dtype=torch.long)
    cross_mask_random[:, :n_pfx_rand] = 0

    # Condition 3: random_trunc
    with torch.no_grad():
        out = model(encoder_outputs=enc_out_random, attention_mask=cross_mask_random,
                    decoder_input_ids=dec_q)
    logits = out.logits[0, n_q:n_q + len(answer_ids), :]
    nll = -F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
    result['nll_random_trunc'] = nll

    del enc_out_random, out
    return result


print("Scoring function defined.")
""")


# ===== Cell 5: Model sweep =====
code(r"""# Cell 5: Run sweep across all models
print("=" * 70)
print("MODEL SWEEP")
print("=" * 70)

COND_NAMES = ['bare', 'oracle_trunc', 'random_trunc', 'bare_nq']

all_model_results = {}

for model_idx, model_name in enumerate(MODELS):
    slug = model_name.replace("/", "_")
    ckpt_path = RESULTS_DIR / f"{slug}_checkpoint.json"

    # Check if already completed
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        if len(ckpt.get('results', [])) == N_SAMPLES:
            print(f"\n{'='*70}")
            print(f"[{model_idx+1}/{len(MODELS)}] {model_name} — LOADED FROM CHECKPOINT")
            print(f"{'='*70}")
            all_model_results[model_name] = ckpt['results']
            continue

    print(f"\n{'='*70}")
    print(f"[{model_idx+1}/{len(MODELS)}] {model_name}")
    print(f"{'='*70}")

    # Load model
    t0 = time.time()
    print(f"  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    gpu_mem = torch.cuda.memory_allocated() / 1e9
    print(f"  Loaded: {n_params:.0f}M params, {gpu_mem:.1f} GB GPU")
    print(f"  BOS/decoder_start_token_id: {model.config.decoder_start_token_id}")

    # Resume from partial checkpoint
    model_results = []
    start_idx = 0
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        if len(ckpt.get('results', [])) > 0:
            saved_queries = [r['query'][:50] for r in ckpt['results']]
            current_queries = [s['query'][:50] for s in samples[:len(saved_queries)]]
            if saved_queries == current_queries:
                model_results = ckpt['results']
                start_idx = len(model_results)
                print(f"  Resuming from checkpoint: {start_idx}/{N_SAMPLES}")

    if start_idx == 0:
        print(f"  Starting fresh: {N_SAMPLES} samples x {len(COND_NAMES)} conditions")

    for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
                  desc=f"  {model_name.split('/')[-1]}"):
        s = samples[i]
        try:
            result = score_sample(model, tokenizer, s, DEVICE)
        except Exception as e:
            print(f"  ERROR at sample {i}: {e}")
            result = None

        if result is None:
            continue
        result['query'] = s['query'][:50]
        model_results.append(result)

        if (i + 1) % 50 == 0 or i == N_SAMPLES - 1:
            ckpt = {
                'model': model_name,
                'n_total': N_SAMPLES,
                'results': model_results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            ckpt_path.write_text(json.dumps(ckpt))

        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Done: {len(model_results)} samples in {elapsed/60:.1f} min")

    # Quick summary
    for cond in COND_NAMES:
        vals = [r[f'nll_{cond}'] for r in model_results]
        print(f"    {cond:<20} NLL={np.mean(vals):.4f}")

    all_model_results[model_name] = model_results

    # Unload model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  GPU freed: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print(f"\n{'='*70}")
print(f"ALL MODELS COMPLETE")
print(f"{'='*70}")
""")


# ===== Cell 6: Cross-model comparison =====
code(r"""# Cell 6: Cross-model comparison
print("=" * 70)
print("CROSS-MODEL COMPARISON")
print("=" * 70)

# Include T5Gemma reference from Exp 01
T5GEMMA_REF = {
    'nll_bare': 2.554,
    'nll_oracle_trunc': 2.406,
    'nll_bare_nq': 3.676,
    'd_oracle': 0.228,
    'd_random': 0.080,
    'structural_frac': 0.35,
}

print(f"\n--- NLL by model and condition ---")
print(f"\n  {'Model':<25} {'bare':>8} {'oracle':>8} {'random':>8} {'bare_nq':>8}")
print(f"  {'-'*62}")

# T5Gemma reference
print(f"  {'T5Gemma-2-4B (Exp 01)':<25} {T5GEMMA_REF['nll_bare']:>8.3f} "
      f"{T5GEMMA_REF['nll_oracle_trunc']:>8.3f} {'—':>8} "
      f"{T5GEMMA_REF['nll_bare_nq']:>8.3f}")

model_summary = {}

for model_name in MODELS:
    if model_name not in all_model_results:
        continue
    res = all_model_results[model_name]
    short = model_name.split('/')[-1]

    nll = {}
    for cond in COND_NAMES:
        nll[cond] = np.array([r[f'nll_{cond}'] for r in res])

    print(f"  {short:<25} {nll['bare'].mean():>8.3f} {nll['oracle_trunc'].mean():>8.3f} "
          f"{nll['random_trunc'].mean():>8.3f} {nll['bare_nq'].mean():>8.3f}")

    model_summary[model_name] = nll

# Effect sizes
print(f"\n--- Effect sizes (Cohen's d, positive = condition helps) ---")
print(f"\n  {'Model':<25} {'d_oracle':>10} {'d_random':>10} {'struct%':>10} "
      f"{'d_dec_q':>10} {'oracle_sig':>10}")
print(f"  {'-'*80}")

# T5Gemma reference
print(f"  {'T5Gemma-2-4B (Exp 01)':<25} {T5GEMMA_REF['d_oracle']:>+10.3f} "
      f"{T5GEMMA_REF['d_random']:>+10.3f} {T5GEMMA_REF['structural_frac']*100:>9.0f}% "
      f"{'—':>10} {'***':>10}")

for model_name in MODELS:
    if model_name not in model_summary:
        continue
    nll = model_summary[model_name]
    short = model_name.split('/')[-1]

    # Oracle enrichment: bare - oracle (positive = oracle helps)
    oracle_diff = nll['bare'] - nll['oracle_trunc']
    d_oracle = cohens_d(oracle_diff)
    _, p_oracle = stats.ttest_1samp(oracle_diff, 0)
    sig_o = '***' if p_oracle < 0.001 else '**' if p_oracle < 0.01 else '*' if p_oracle < 0.05 else 'ns'

    # Random enrichment: bare - random (positive = random helps)
    random_diff = nll['bare'] - nll['random_trunc']
    d_random = cohens_d(random_diff)

    # Structural fraction
    struct_frac = d_random / d_oracle if d_oracle != 0 else float('nan')

    # Decoder query effect: bare_nq - bare (positive = query helps)
    dec_q_diff = nll['bare_nq'] - nll['bare']
    d_dec_q = cohens_d(dec_q_diff)

    print(f"  {short:<25} {d_oracle:>+10.3f} {d_random:>+10.3f} "
          f"{struct_frac*100:>9.0f}% {d_dec_q:>+10.3f} {sig_o:>10}")

# Detailed pairwise tests
print(f"\n--- Pairwise significance tests ---")
for model_name in MODELS:
    if model_name not in model_summary:
        continue
    nll = model_summary[model_name]
    short = model_name.split('/')[-1]

    print(f"\n  {short}:")
    for cond, label in [('oracle_trunc', 'oracle_trunc vs bare'),
                        ('random_trunc', 'random_trunc vs bare'),
                        ('bare_nq', 'bare vs bare_nq (query effect)')]:
        if cond == 'bare_nq':
            diff = nll['bare_nq'] - nll['bare']
            d = cohens_d(diff)
        else:
            diff = nll['bare'] - nll[cond]
            d = cohens_d(diff)
        _, p = stats.ttest_1samp(diff, 0)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        win = (diff > 0).mean() * 100
        print(f"    {label:<30} d={d:+.3f}  win={win:.0f}%  p={p:.2e} {sig}")
""")


# ===== Cell 7: Summary + save =====
code(r"""# Cell 7: Summary and save
print("=" * 70)
print("SUMMARY — Exp 09: Cross-Model Generalization")
print("=" * 70)

# Collect summary for all models
summary = {}

for model_name in MODELS:
    if model_name not in model_summary:
        continue
    nll = model_summary[model_name]
    short = model_name.split('/')[-1]
    res = all_model_results[model_name]

    oracle_diff = nll['bare'] - nll['oracle_trunc']
    random_diff = nll['bare'] - nll['random_trunc']
    dec_q_diff = nll['bare_nq'] - nll['bare']

    d_oracle = cohens_d(oracle_diff)
    d_random = cohens_d(random_diff)
    d_dec_q = cohens_d(dec_q_diff)
    _, p_oracle = stats.ttest_1samp(oracle_diff, 0)
    _, p_random = stats.ttest_1samp(random_diff, 0)
    struct_frac = d_random / d_oracle if d_oracle != 0 else float('nan')

    summary[model_name] = {
        'short_name': short,
        'n_samples': len(res),
        'nll_bare': float(nll['bare'].mean()),
        'nll_oracle_trunc': float(nll['oracle_trunc'].mean()),
        'nll_random_trunc': float(nll['random_trunc'].mean()),
        'nll_bare_nq': float(nll['bare_nq'].mean()),
        'd_oracle': float(d_oracle),
        'd_random': float(d_random),
        'd_dec_q': float(d_dec_q),
        'p_oracle': float(p_oracle),
        'p_random': float(p_random),
        'structural_fraction': float(struct_frac),
    }

# Count models where oracle is significant
n_sig = sum(1 for v in summary.values() if v['p_oracle'] < 0.05)
n_total = len(summary)

print(f"\nModels tested: {n_total}")
print(f"Models with significant oracle enrichment (p<0.05): {n_sig}/{n_total}")

print(f"\n  {'Model':<25} {'d_oracle':>10} {'d_random':>10} {'struct%':>10} {'Sig':>5}")
print(f"  {'-'*55}")

# T5Gemma reference
print(f"  {'T5Gemma-2-4B':<25} {'+0.228':>10} {'+0.080':>10} {'35%':>10} {'***':>5}")

for model_name in MODELS:
    if model_name not in summary:
        continue
    s = summary[model_name]
    sig = '***' if s['p_oracle'] < 0.001 else '**' if s['p_oracle'] < 0.01 else '*' if s['p_oracle'] < 0.05 else 'ns'
    print(f"  {s['short_name']:<25} {s['d_oracle']:>+10.3f} {s['d_random']:>+10.3f} "
          f"{s['structural_fraction']*100:>9.0f}% {sig:>5}")

# Verdict
if n_sig == n_total:
    print(f"\n  VERDICT: Enrichment effect GENERALIZES across all {n_total} models tested.")
elif n_sig > 0:
    print(f"\n  VERDICT: Enrichment effect generalizes to {n_sig}/{n_total} models.")
    print(f"  Not universal — may depend on architecture or training.")
else:
    print(f"\n  VERDICT: Enrichment effect does NOT generalize.")
    print(f"  The effect appears specific to T5Gemma.")

# Save
final_results = {
    'experiment': 'v4_exp09_cross_model_sweep',
    'dataset': 'ms_marco_v1.1',
    'n_samples': N_SAMPLES,
    'seed': SEED,
    'models_tested': MODELS,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'model_results': summary,
    't5gemma_reference': T5GEMMA_REF,
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")
print("Done!")
""")


# ===== Write notebook =====
out_path = "experiments/encoder_decoder/09/09_cross_model_sweep.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
