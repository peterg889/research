#!/usr/bin/env python3
# Build Exp 10 notebook: T5 Size Scaling.
#
# Exp 09 tested cross-architecture generalization (flan-T5 + BART) but those
# models had negative d_dec_q (query in decoder HURTS — against training
# distribution). This experiment uses standard (non-instruction-tuned) T5
# models to get a clean size-scaling curve within one architecture family.
#
# Models: t5-small (60M), t5-base (220M), t5-large (770M), t5-3b (3B)
# 6 conditions per model (core Exp 01 set), N=500, MS MARCO, SEED=42.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 10: T5 Size Scaling

## Motivation

Exp 09 showed enrichment generalizes across architectures (flan-T5 + BART), but
those models had **negative d_dec_q** — putting the query in the decoder *hurts*
because it violates their training distribution (instruction-tuned for
query-in-encoder / answer-in-decoder).

This experiment uses **standard (non-instruction-tuned) T5** models to get a clean
size-scaling curve within one architecture family, without the confound of
instruction tuning fighting the v4 decoder-query setup.

## Design

**4 models** (T5 size ladder):

| Model | Params | Encoder layers | Decoder layers | d_model |
|-------|--------|---------------|----------------|---------|
| t5-small | 60M | 6 | 6 | 512 |
| t5-base | 220M | 12 | 12 | 768 |
| t5-large | 770M | 24 | 24 | 1024 |
| t5-3b | 3B | 24 | 24 | 1024 |

**6 conditions per model:**

| # | Condition | Encoder input | Cross-attn mask | Decoder input |
|---|-----------|--------------|-----------------|---------------|
| 1 | bare | [document] | all | [BOS, query, answer] |
| 2 | oracle_trunc | [query + doc] | doc only | [BOS, query, answer] |
| 3 | surr_doc_trunc | [kw5 + doc] | doc only | [BOS, query, answer] |
| 4 | random_trunc | [random + doc] | doc only | [BOS, query, answer] |
| 5 | bare_nq | [document] | all | [BOS, answer] |
| 6 | oracle_trunc_nq | [query + doc] | doc only | [BOS, answer] |

**Key comparisons (per model):**
- **(2) vs (1)**: Oracle enrichment with query in decoder (THE test)
- **(3) vs (1)**: Surrogate (doc keywords) — practical value
- **(4) vs (1)**: Structural component (random prefix)
- **(6) vs (5)**: Oracle enrichment without query (v3 replication)
- **(1) vs (5)**: Decoder query effect (d_dec_q)
- **struct_frac**: d_random / d_oracle — how much is structural?

Same 500 MS MARCO samples across all models. N=500, SEED=42.
NLL only (no attention probes), SDPA for speed.""")


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
from collections import Counter
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, "../../..")
from lib.analysis import cohens_d

SEED = 42
N_SAMPLES = 500

RESULTS_DIR = Path("../../../results/exp10")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = [
    "google-t5/t5-small",
    "google-t5/t5-base",
    "google-t5/t5-large",
    "google-t5/t5-3b",
]

print(f"Exp 10: T5 Size Scaling")
print(f"N: {N_SAMPLES}, Models: {len(MODELS)}")
print(f"DEVICE: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
""")


# ===== Cell 3: Data loading =====
code(r"""# Cell 3: Load MS MARCO data + generate surrogates and random prefixes
from lib.data import count_words
from datasets import load_dataset

STOP_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
    'not', 'no', 'if', 'then', 'than', 'so', 'up', 'out', 'about',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he',
    'him', 'his', 'she', 'her', 'they', 'them', 'their', 'how', 'when',
    'where', 'why', 'much', 'many', 'some', 'any', 'all', 'each',
    'does', 'also', 'just', 'more', 'most', 'very', 'too', 'only',
}

def extract_keywords(text):
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]

def make_surrogate_from_doc(passage):
    content_words = extract_keywords(passage)
    if not content_words:
        return "information"
    counts = Counter(content_words)
    return " ".join(w for w, _ in counts.most_common(5))

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

# Generate surrogates and random prefix TEXT (shared across models)
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
    s['surr_doc'] = make_surrogate_from_doc(s['passage'])
    rng = np.random.RandomState(SEED + i + 20000)
    words = rng.choice(WORD_POOL, size=8, replace=False)
    s['random_prefix'] = " ".join(words)

print(f"Loaded {len(samples)} samples")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Mean query words: {np.mean([count_words(s['query']) for s in samples]):.0f}")
print(f"Mean answer words: {np.mean([count_words(s['answer']) for s in samples]):.0f}")
print(f"Example surr_doc: '{samples[0]['surr_doc']}'")
print(f"Example random prefix: '{samples[0]['random_prefix']}'")
""")


# ===== Cell 4: Scoring function =====
code(r"""# Cell 4: Model-agnostic scoring function (6 conditions)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def count_prefix_tokens(tokenizer, prefix_text, document_text):
    full_text = prefix_text + "\n" + document_text
    full_ids = tokenizer(full_text, add_special_tokens=True, truncation=True,
                         max_length=2048).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids
    return len(full_ids) - len(doc_ids)


def score_sample(model, tokenizer, sample, device):
    # Score one sample under all 6 conditions.
    passage = sample['passage']
    query = sample['query']
    answer = sample['answer']
    surr_doc = sample['surr_doc']
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

    targets = torch.tensor(answer_ids, dtype=torch.long, device=device)
    n_q = len(query_ids)
    n_a = len(answer_ids)
    result = {}

    # Decoder inputs
    dec_q = torch.tensor([[bos_id] + query_ids + answer_ids],
                          dtype=torch.long, device=device)
    dec_nq = torch.tensor([[bos_id] + answer_ids],
                           dtype=torch.long, device=device)

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
    with torch.no_grad():
        out = model(encoder_outputs=enc_out_bare, attention_mask=enc_mask_bare,
                    decoder_input_ids=dec_q)
    logits = out.logits[0, n_q:n_q + n_a, :]
    nll = -F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
    result['nll_bare'] = nll

    # Condition 5: bare_nq (no query in decoder)
    with torch.no_grad():
        out = model(encoder_outputs=enc_out_bare, attention_mask=enc_mask_bare,
                    decoder_input_ids=dec_nq)
    logits = out.logits[0, :n_a, :]
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

    n_pfx_oracle = count_prefix_tokens(tokenizer, query, passage)
    cross_mask_oracle = torch.ones(1, enc_ids_oracle.shape[1], device=device, dtype=torch.long)
    cross_mask_oracle[:, :n_pfx_oracle] = 0

    # Condition 2: oracle_trunc (decoder has query)
    with torch.no_grad():
        out = model(encoder_outputs=enc_out_oracle, attention_mask=cross_mask_oracle,
                    decoder_input_ids=dec_q)
    logits = out.logits[0, n_q:n_q + n_a, :]
    nll = -F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
    result['nll_oracle_trunc'] = nll

    # Condition 6: oracle_trunc_nq (no query in decoder)
    with torch.no_grad():
        out = model(encoder_outputs=enc_out_oracle, attention_mask=cross_mask_oracle,
                    decoder_input_ids=dec_nq)
    logits = out.logits[0, :n_a, :]
    nll = -F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
    result['nll_oracle_trunc_nq'] = nll

    del enc_out_oracle

    # === Encoder pass 3: surr_doc (top-5 keywords + document) ===
    surr_text = surr_doc + "\n" + passage
    enc_ids_surr = tokenizer(surr_text, return_tensors="pt",
                              add_special_tokens=True, truncation=True,
                              max_length=2048).input_ids.to(device)
    enc_mask_surr = torch.ones(1, enc_ids_surr.shape[1], device=device, dtype=torch.long)
    with torch.no_grad():
        enc_out_surr = model.get_encoder()(
            input_ids=enc_ids_surr, attention_mask=enc_mask_surr
        )

    n_pfx_surr = count_prefix_tokens(tokenizer, surr_doc, passage)
    cross_mask_surr = torch.ones(1, enc_ids_surr.shape[1], device=device, dtype=torch.long)
    cross_mask_surr[:, :n_pfx_surr] = 0

    # Condition 3: surr_doc_trunc (decoder has query)
    with torch.no_grad():
        out = model(encoder_outputs=enc_out_surr, attention_mask=cross_mask_surr,
                    decoder_input_ids=dec_q)
    logits = out.logits[0, n_q:n_q + n_a, :]
    nll = -F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
    result['nll_surr_doc_trunc'] = nll

    del enc_out_surr

    # === Encoder pass 4: random prefix ===
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

    # Condition 4: random_trunc (decoder has query)
    with torch.no_grad():
        out = model(encoder_outputs=enc_out_random, attention_mask=cross_mask_random,
                    decoder_input_ids=dec_q)
    logits = out.logits[0, n_q:n_q + n_a, :]
    nll = -F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()
    result['nll_random_trunc'] = nll

    del enc_out_random, out
    return result


print("Scoring function defined (6 conditions per sample).")
""")


# ===== Cell 5: Model sweep =====
code(r"""# Cell 5: Run sweep across all models
print("=" * 70)
print("MODEL SWEEP")
print("=" * 70)

COND_NAMES = ['bare', 'oracle_trunc', 'surr_doc_trunc', 'random_trunc',
              'bare_nq', 'oracle_trunc_nq']

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
code(r"""# Cell 6: Cross-model comparison table
print("=" * 70)
print("CROSS-MODEL COMPARISON")
print("=" * 70)

# Reference values from v4 Exp 01 (T5Gemma) and Exp 09 (flan-T5)
REFS = {
    'T5Gemma-2-4B': {
        'd_oracle': 0.228, 'd_random': 0.080, 'd_surr_doc': 0.148,
        'struct_frac': 0.35, 'd_dec_q': 0.309,
    },
    'flan-t5-base': {
        'd_oracle': 0.251, 'd_random': 0.107, 'struct_frac': 0.43,
        'd_dec_q': -0.438,
    },
    'flan-t5-large': {
        'd_oracle': 0.320, 'd_random': 0.247, 'struct_frac': 0.77,
        'd_dec_q': -0.416,
    },
    'flan-t5-xl': {
        'd_oracle': 0.430, 'd_random': 0.030, 'struct_frac': 0.07,
        'd_dec_q': -0.341,
    },
}

# NLL table
print(f"\n--- Mean NLL by model and condition ---")
print(f"\n  {'Model':<20} {'bare':>8} {'oracle':>8} {'surr_doc':>8} {'random':>8} "
      f"{'bare_nq':>8} {'orc_nq':>8}")
print(f"  {'-'*75}")

model_summary = {}

for model_name in MODELS:
    if model_name not in all_model_results:
        continue
    res = all_model_results[model_name]
    short = model_name.split('/')[-1]

    nll = {}
    for cond in COND_NAMES:
        nll[cond] = np.array([r[f'nll_{cond}'] for r in res])

    print(f"  {short:<20} {nll['bare'].mean():>8.3f} {nll['oracle_trunc'].mean():>8.3f} "
          f"{nll['surr_doc_trunc'].mean():>8.3f} {nll['random_trunc'].mean():>8.3f} "
          f"{nll['bare_nq'].mean():>8.3f} {nll['oracle_trunc_nq'].mean():>8.3f}")

    model_summary[model_name] = nll

# Effect sizes
print(f"\n--- Effect sizes (Cohen's d, positive = condition helps) ---")
print(f"\n  {'Model':<20} {'d_oracle':>9} {'d_surr':>9} {'d_random':>9} "
      f"{'struct%':>8} {'d_dec_q':>9} {'d_nq_orc':>9}")
print(f"  {'-'*75}")

# T5Gemma reference
ref = REFS['T5Gemma-2-4B']
print(f"  {'T5Gemma-2-4B*':<20} {ref['d_oracle']:>+9.3f} {ref['d_surr_doc']:>+9.3f} "
      f"{ref['d_random']:>+9.3f} {ref['struct_frac']*100:>7.0f}% {ref['d_dec_q']:>+9.3f} "
      f"{'—':>9}")

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

    # Surrogate enrichment
    surr_diff = nll['bare'] - nll['surr_doc_trunc']
    d_surr = cohens_d(surr_diff)

    # Random enrichment
    random_diff = nll['bare'] - nll['random_trunc']
    d_random = cohens_d(random_diff)

    # Structural fraction
    struct_frac = d_random / d_oracle if d_oracle != 0 else float('nan')

    # Decoder query effect: bare_nq - bare (positive = query helps)
    dec_q_diff = nll['bare_nq'] - nll['bare']
    d_dec_q = cohens_d(dec_q_diff)

    # v3 replication: oracle_trunc_nq vs bare_nq
    nq_diff = nll['bare_nq'] - nll['oracle_trunc_nq']
    d_nq_orc = cohens_d(nq_diff)

    print(f"  {short:<20} {d_oracle:>+9.3f}{sig_o:>3} {d_surr:>+9.3f} "
          f"{d_random:>+9.3f} {struct_frac*100:>7.0f}% {d_dec_q:>+9.3f} "
          f"{d_nq_orc:>+9.3f}")

# Flan-T5 references for comparison
print(f"\n  --- Flan-T5 references (Exp 09) ---")
for name in ['flan-t5-base', 'flan-t5-large', 'flan-t5-xl']:
    ref = REFS[name]
    sf = ref.get('struct_frac', 0)
    print(f"  {name+'*':<20} {ref['d_oracle']:>+9.3f}    {'—':>9} "
          f"{ref['d_random']:>+9.3f} {sf*100:>7.0f}% {ref['d_dec_q']:>+9.3f} {'—':>9}")

print(f"\n  * Reference values from previous experiments")
""")


# ===== Cell 7: Detailed pairwise tests =====
code(r"""# Cell 7: Detailed pairwise significance tests
print("=" * 70)
print("PAIRWISE SIGNIFICANCE TESTS")
print("=" * 70)

for model_name in MODELS:
    if model_name not in model_summary:
        continue
    nll = model_summary[model_name]
    short = model_name.split('/')[-1]

    print(f"\n  {short} ({len(all_model_results[model_name])} samples):")
    print(f"  {'Comparison':<35} {'d':>8} {'win%':>7} {'p':>12} {'sig':>5}")
    print(f"  {'-'*70}")

    comparisons = [
        ('oracle_trunc vs bare', nll['bare'] - nll['oracle_trunc']),
        ('surr_doc_trunc vs bare', nll['bare'] - nll['surr_doc_trunc']),
        ('random_trunc vs bare', nll['bare'] - nll['random_trunc']),
        ('oracle_trunc_nq vs bare_nq', nll['bare_nq'] - nll['oracle_trunc_nq']),
        ('bare vs bare_nq (query eff.)', nll['bare_nq'] - nll['bare']),
        ('surr_doc vs random (content)', nll['random_trunc'] - nll['surr_doc_trunc']),
    ]

    for label, diff in comparisons:
        d = cohens_d(diff)
        _, p = stats.ttest_1samp(diff, 0)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        win = (diff > 0).mean() * 100
        print(f"  {label:<35} {d:>+8.3f} {win:>6.1f}% {p:>12.2e} {sig:>5}")
""")


# ===== Cell 8: Size scaling analysis =====
code(r"""# Cell 8: Size scaling trends
print("=" * 70)
print("SIZE SCALING ANALYSIS")
print("=" * 70)

# Collect per-model metrics for scaling analysis
model_sizes = []
d_oracles = []
d_surrs = []
d_randoms = []
d_dec_qs = []
d_nq_orcs = []
struct_fracs = []
model_labels = []

for model_name in MODELS:
    if model_name not in model_summary:
        continue
    nll = model_summary[model_name]
    short = model_name.split('/')[-1]
    res = all_model_results[model_name]

    # Approximate param count from model name
    size_map = {'t5-small': 60, 't5-base': 220, 't5-large': 770, 't5-3b': 3000}
    n_params = size_map.get(short, 0)

    d_oracle = cohens_d(nll['bare'] - nll['oracle_trunc'])
    d_surr = cohens_d(nll['bare'] - nll['surr_doc_trunc'])
    d_random = cohens_d(nll['bare'] - nll['random_trunc'])
    d_dec_q = cohens_d(nll['bare_nq'] - nll['bare'])
    d_nq_orc = cohens_d(nll['bare_nq'] - nll['oracle_trunc_nq'])
    struct_frac = d_random / d_oracle if d_oracle != 0 else float('nan')

    model_sizes.append(n_params)
    d_oracles.append(d_oracle)
    d_surrs.append(d_surr)
    d_randoms.append(d_random)
    d_dec_qs.append(d_dec_q)
    d_nq_orcs.append(d_nq_orc)
    struct_fracs.append(struct_frac)
    model_labels.append(short)

print(f"\n--- Scaling trends ---")
print(f"\n  {'Model':<12} {'Params':>8} {'d_oracle':>10} {'d_surr':>10} {'d_random':>10} "
      f"{'struct%':>8} {'d_dec_q':>10} {'v4/v3%':>8}")
print(f"  {'-'*80}")

for i, short in enumerate(model_labels):
    v4_v3_ratio = d_oracles[i] / d_nq_orcs[i] * 100 if d_nq_orcs[i] > 0 else float('nan')
    print(f"  {short:<12} {model_sizes[i]:>7}M {d_oracles[i]:>+10.3f} {d_surrs[i]:>+10.3f} "
          f"{d_randoms[i]:>+10.3f} {struct_fracs[i]*100:>7.0f}% {d_dec_qs[i]:>+10.3f} "
          f"{v4_v3_ratio:>7.0f}%")

# Rank correlations with model size
if len(model_sizes) >= 4:
    log_sizes = np.log10(model_sizes)
    for metric_name, metric_vals in [
        ('d_oracle', d_oracles), ('d_random', d_randoms),
        ('struct_frac', struct_fracs), ('d_dec_q', d_dec_qs),
    ]:
        rho, p = stats.spearmanr(log_sizes, metric_vals)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"\n  Spearman rho(log_size, {metric_name}): {rho:+.3f} (p={p:.3f}) {sig}")

# Compare standard T5 vs flan-T5 d_dec_q
print(f"\n--- Standard T5 vs Flan-T5: decoder query effect ---")
print(f"  Standard T5 d_dec_q: {[f'{d:+.3f}' for d in d_dec_qs]}")
print(f"  Flan-T5 d_dec_q:     [-0.438, -0.416, -0.341] (all NEGATIVE)")
print(f"  Key question: do standard T5 models also have negative d_dec_q,")
print(f"  or is that specific to instruction tuning?")
""")


# ===== Cell 9: Summary + save =====
code(r"""# Cell 9: Summary and save
print("=" * 70)
print("SUMMARY — Exp 10: T5 Size Scaling")
print("=" * 70)

summary = {}

for model_name in MODELS:
    if model_name not in model_summary:
        continue
    nll = model_summary[model_name]
    short = model_name.split('/')[-1]
    res = all_model_results[model_name]

    oracle_diff = nll['bare'] - nll['oracle_trunc']
    surr_diff = nll['bare'] - nll['surr_doc_trunc']
    random_diff = nll['bare'] - nll['random_trunc']
    dec_q_diff = nll['bare_nq'] - nll['bare']
    nq_orc_diff = nll['bare_nq'] - nll['oracle_trunc_nq']

    d_oracle = cohens_d(oracle_diff)
    d_surr = cohens_d(surr_diff)
    d_random = cohens_d(random_diff)
    d_dec_q = cohens_d(dec_q_diff)
    d_nq_orc = cohens_d(nq_orc_diff)
    _, p_oracle = stats.ttest_1samp(oracle_diff, 0)
    _, p_surr = stats.ttest_1samp(surr_diff, 0)
    _, p_random = stats.ttest_1samp(random_diff, 0)
    struct_frac = d_random / d_oracle if d_oracle != 0 else float('nan')

    summary[model_name] = {
        'short_name': short,
        'n_samples': len(res),
        'nll_bare': float(nll['bare'].mean()),
        'nll_oracle_trunc': float(nll['oracle_trunc'].mean()),
        'nll_surr_doc_trunc': float(nll['surr_doc_trunc'].mean()),
        'nll_random_trunc': float(nll['random_trunc'].mean()),
        'nll_bare_nq': float(nll['bare_nq'].mean()),
        'nll_oracle_trunc_nq': float(nll['oracle_trunc_nq'].mean()),
        'd_oracle': float(d_oracle),
        'd_surr_doc': float(d_surr),
        'd_random': float(d_random),
        'd_dec_q': float(d_dec_q),
        'd_nq_oracle': float(d_nq_orc),
        'p_oracle': float(p_oracle),
        'p_surr_doc': float(p_surr),
        'p_random': float(p_random),
        'structural_fraction': float(struct_frac),
    }

# Final table
n_sig = sum(1 for v in summary.values() if v['p_oracle'] < 0.05)
n_total = len(summary)

print(f"\nModels tested: {n_total}")
print(f"Models with significant oracle enrichment (p<0.05): {n_sig}/{n_total}")

print(f"\n  {'Model':<12} {'Params':>8} {'d_oracle':>10} {'d_surr':>10} {'d_random':>10} "
      f"{'struct%':>8} {'d_dec_q':>10} {'Sig':>5}")
print(f"  {'-'*70}")

size_map = {'t5-small': '60M', 't5-base': '220M', 't5-large': '770M', 't5-3b': '3B'}
for model_name in MODELS:
    if model_name not in summary:
        continue
    s = summary[model_name]
    sig = '***' if s['p_oracle'] < 0.001 else '**' if s['p_oracle'] < 0.01 else '*' if s['p_oracle'] < 0.05 else 'ns'
    params = size_map.get(s['short_name'], '?')
    print(f"  {s['short_name']:<12} {params:>8} {s['d_oracle']:>+10.3f} {s['d_surr_doc']:>+10.3f} "
          f"{s['d_random']:>+10.3f} {s['structural_fraction']*100:>7.0f}% "
          f"{s['d_dec_q']:>+10.3f} {sig:>5}")

# Verdict
print(f"\n  VERDICT:")
if n_sig == n_total:
    print(f"  Enrichment effect is significant for ALL {n_total} standard T5 sizes.")
elif n_sig > 0:
    print(f"  Enrichment effect significant for {n_sig}/{n_total} models.")
else:
    print(f"  Enrichment NOT significant for standard T5 models.")

# d_dec_q comparison
pos_dec_q = sum(1 for v in summary.values() if v['d_dec_q'] > 0)
print(f"  Positive d_dec_q: {pos_dec_q}/{n_total} (vs 0/3 for flan-T5)")
if pos_dec_q > 0:
    print(f"  Standard T5 CAN benefit from decoder query (unlike flan-T5).")
else:
    print(f"  Standard T5 also has negative d_dec_q (same as flan-T5).")
    print(f"  The v4 decoder-query setup may not match T5's training distribution.")

# Save
final_results = {
    'experiment': 'v4_exp10_t5_size_scaling',
    'dataset': 'ms_marco_v1.1',
    'n_samples': N_SAMPLES,
    'seed': SEED,
    'models_tested': MODELS,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'model_results': summary,
    'references': {
        'T5Gemma_exp01': REFS['T5Gemma-2-4B'],
        'flan_t5_exp09': {k: REFS[k] for k in ['flan-t5-base', 'flan-t5-large', 'flan-t5-xl']},
    },
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")
print("Done!")
""")


# ===== Write notebook =====
out_path = "experiments/encoder_decoder/10/10_t5_size_scaling.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
