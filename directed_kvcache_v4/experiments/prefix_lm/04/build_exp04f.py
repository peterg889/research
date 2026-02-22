#!/usr/bin/env python3
# Build Prefix LM Exp 04f notebook: Attention Probing.
#
# Does oracle actually change doc-token attention differently than random?
# Extract attention weights during Phase A to compare how document tokens
# attend to oracle vs random vs no prime.
#
# 3 conditions (bare, random, oracle), N=500, attention extraction on last 10 layers.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 0: Markdown =====
md(r"""# Prefix LM Exp 04f: Attention Probing

## Motivation

Random tokens work as well as oracle under truncation (structural fraction ~105%).
But WHY? Does the oracle query actually change doc-token attention patterns, and
the effect just doesn't survive truncation? Or do oracle and random shift attention
identically?

## Design

During Phase A (`[BOS, prime, doc]` with causal attention), extract attention weights
and compute:

1. **Doc-to-prime attention**: What fraction of each doc token's attention goes to
   prime positions vs BOS vs other doc tokens?
2. **Attention entropy**: Is attention more focused (lower entropy) for oracle vs random?
3. **Layer-by-layer**: Which layers show the biggest prime-type differences?
4. **Correlation with NLL**: Do samples where oracle gets more attention also get more NLL benefit?

## Conditions (3)

| # | Condition | Prime | What it tests |
|---|-----------|-------|---------------|
| 1 | `bare` | (none) | No prime baseline |
| 2 | `random` | 8 random words | Structural attention pattern |
| 3 | `oracle` | real query | Semantic attention pattern |

N=500 samples. Attention extracted from every 4th layer (11 layers total).""")


# ===== Cell 1: Setup =====
code(r"""# Cell 1: Setup
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

MODEL_NAME = "google/gemma-3-12b-it"

RESULTS_DIR = Path("../../../results/prefix_lm_exp04f")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONDITIONS = ["bare", "random", "oracle"]

# Probe every 4th layer + last layer
# Gemma 3 12B has 48 layers
PROBE_LAYERS = list(range(0, 48, 4)) + [47]
PROBE_LAYERS = sorted(set(PROBE_LAYERS))

print(f"Prefix LM Exp 04f: Attention Probing")
print(f"N: {N_SAMPLES}, Conditions: {len(CONDITIONS)}")
print(f"Probe layers: {PROBE_LAYERS} ({len(PROBE_LAYERS)} layers)")
print(f"DEVICE: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
""")


# ===== Cell 2: Load model =====
code(r"""# Cell 2: Load model + tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

print(f"transformers version: {transformers.__version__}")

print(f"Loading {MODEL_NAME}...")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    token=HF_TOKEN,
)
model.eval()

n_params = sum(p.numel() for p in model.parameters()) / 1e9
gpu_mem = torch.cuda.memory_allocated() / 1e9
print(f"Loaded: {n_params:.1f}B params, {gpu_mem:.1f} GB GPU, {time.time()-t0:.0f}s")

# Verify number of layers (Gemma 3 config uses text_config)
n_layers = model.config.text_config.num_hidden_layers
print(f"Model has {n_layers} layers")
# Update PROBE_LAYERS if needed
PROBE_LAYERS = [l for l in PROBE_LAYERS if l < n_layers]
if (n_layers - 1) not in PROBE_LAYERS:
    PROBE_LAYERS.append(n_layers - 1)
PROBE_LAYERS = sorted(set(PROBE_LAYERS))
print(f"Adjusted probe layers: {PROBE_LAYERS} ({len(PROBE_LAYERS)} layers)")
""")


# ===== Cell 3: Mask functions =====
code(r"""# Cell 3: Mask functions (same as Exp 04d/04e)

def make_phase_a_mask(n_s, n_d, dtype=torch.bfloat16):
    n_prefix = 1 + n_s + n_d
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((n_prefix, n_prefix), min_val, dtype=dtype),
                      diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def make_phase_b_mask(n_s, n_d, n_q, n_a, dtype=torch.bfloat16):
    n_prefix = 1 + n_s + n_d
    n_cont = n_q + n_a
    min_val = torch.finfo(dtype).min
    mask = torch.full((n_cont, n_prefix + n_cont), min_val, dtype=dtype)
    mask[:, :n_prefix] = 0.0
    if n_s > 0:
        mask[:, 1:1 + n_s] = min_val
    mask[:, n_prefix:] = torch.triu(
        torch.full((n_cont, n_cont), min_val, dtype=dtype), diagonal=1
    )
    return mask.unsqueeze(0).unsqueeze(0)


def make_mask_dict(mask_4d):
    return {"full_attention": mask_4d, "sliding_attention": mask_4d}


print("Mask functions defined.")
""")


# ===== Cell 4: Load data =====
code(r"""# Cell 4: Load MS MARCO data (same pipeline)
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

for i, s in enumerate(samples):
    rng = np.random.RandomState(SEED + i + 20000)
    words = rng.choice(WORD_POOL, size=8, replace=False)
    s['random_prefix'] = " ".join(words)

    q_words = set(re.sub(r'[^\w\s]', '', s['query'].lower()).split()) - STOP_WORDS
    d_words = set(re.sub(r'[^\w\s]', '', s['passage'].lower()).split()) - STOP_WORDS
    union = q_words | d_words
    s['query_doc_overlap'] = len(q_words & d_words) / len(union) if len(union) > 0 else 0.0

print(f"Loaded {len(samples)} samples")
""")


# ===== Cell 5: Scoring function with attention extraction =====
code(r"""# Cell 5: score_sample_with_attention()
#
# Phase A: Forward with output_attentions=True, extract attention stats
# Phase B: Forward with cached KVs (truncated), compute NLL
#
# For each probed layer, compute (averaged over heads):
#   - frac_bos: fraction of doc-token attention going to BOS
#   - frac_prime: fraction going to prime positions
#   - frac_doc: fraction going to other doc positions
#   - entropy: attention entropy for doc tokens

def score_sample_with_attention(model, tokenizer, sample, device, probe_layers):
    passage = sample['passage']
    query = sample['query']
    answer = sample['answer']
    random_prefix = sample['random_prefix']

    bos_id = tokenizer.bos_token_id

    doc_ids = tokenizer(passage, add_special_tokens=False, truncation=True,
                        max_length=1024).input_ids
    query_ids = tokenizer(query, add_special_tokens=False, truncation=True,
                          max_length=512).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False, truncation=True,
                           max_length=256).input_ids

    if len(answer_ids) == 0:
        return None

    oracle_ids = tokenizer(query, add_special_tokens=False, truncation=True,
                           max_length=256).input_ids
    random_ids = tokenizer(random_prefix, add_special_tokens=False).input_ids

    prefix_map = {
        "bare": [],
        "random": random_ids,
        "oracle": oracle_ids,
    }

    n_q = len(query_ids)
    n_a = len(answer_ids)
    n_d = len(doc_ids)

    targets = torch.tensor(answer_ids, dtype=torch.long, device=device)
    result = {'n_doc': n_d, 'n_query': n_q}

    for cond_name in CONDITIONS:
        surr_ids = prefix_map[cond_name]
        n_s = len(surr_ids)
        n_prefix = 1 + n_s + n_d
        doc_start = 1 + n_s  # first doc token position

        # === Phase A: with attention extraction ===
        prefix_tokens = [bos_id] + surr_ids + doc_ids
        prefix_input = torch.tensor([prefix_tokens], dtype=torch.long, device=device)

        phase_a_mask = make_phase_a_mask(n_s, n_d)
        phase_a_dict = make_mask_dict(phase_a_mask.to(device))
        phase_a_pos = torch.arange(n_prefix, device=device).unsqueeze(0)

        with torch.no_grad():
            out_a = model(input_ids=prefix_input, attention_mask=phase_a_dict,
                          position_ids=phase_a_pos, use_cache=True,
                          output_attentions=True)
        past_kv = out_a.past_key_values
        attentions = out_a.attentions  # tuple of (1, n_heads, n_prefix, n_prefix)

        # Extract attention statistics for doc tokens
        for layer_idx in probe_layers:
            if layer_idx >= len(attentions):
                continue
            attn = attentions[layer_idx][0]  # (n_heads, n_prefix, n_prefix)

            # Doc tokens: positions doc_start .. n_prefix-1
            if n_d == 0:
                continue
            doc_attn = attn[:, doc_start:, :]  # (n_heads, n_d, n_prefix)

            # Softmax already applied by model, so rows sum to 1
            # Fraction of attention to each region
            frac_bos = doc_attn[:, :, 0].mean().item()

            if n_s > 0:
                frac_prime = doc_attn[:, :, 1:doc_start].sum(dim=-1).mean().item()
            else:
                frac_prime = 0.0

            frac_doc = doc_attn[:, :, doc_start:].sum(dim=-1).mean().item()

            # Attention entropy (over full context, averaged over heads and doc tokens)
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            ent = -(doc_attn * (doc_attn + eps).log()).sum(dim=-1).mean().item()

            result[f'{cond_name}_L{layer_idx}_frac_bos'] = frac_bos
            result[f'{cond_name}_L{layer_idx}_frac_prime'] = frac_prime
            result[f'{cond_name}_L{layer_idx}_frac_doc'] = frac_doc
            result[f'{cond_name}_L{layer_idx}_entropy'] = ent

        del attentions

        # === Phase B: NLL (truncated) ===
        cont_tokens = query_ids + answer_ids
        n_cont = len(cont_tokens)
        cont_input = torch.tensor([cont_tokens], dtype=torch.long, device=device)

        phase_b_mask = make_phase_b_mask(n_s, n_d, n_q, n_a)
        phase_b_dict = make_mask_dict(phase_b_mask.to(device))
        phase_b_pos = torch.arange(n_prefix, n_prefix + n_cont,
                                    device=device).unsqueeze(0)

        with torch.no_grad():
            out_b = model(input_ids=cont_input, attention_mask=phase_b_dict,
                          position_ids=phase_b_pos, past_key_values=past_kv)

        answer_logits = out_b.logits[0, n_q - 1 : n_q + n_a - 1, :]
        log_probs = F.log_softmax(answer_logits, dim=-1)
        token_nlls = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        result[f'nll_{cond_name}'] = token_nlls.mean().item()

        del out_a, out_b, past_kv, prefix_input, cont_input
        del phase_a_mask, phase_b_mask, phase_a_dict, phase_b_dict
        del answer_logits, log_probs, token_nlls

    return result


print(f"Scoring function defined (with attention probing).")
print(f"Probing {len(PROBE_LAYERS)} layers per condition.")
""")


# ===== Cell 6: Main loop =====
code(r"""# Cell 6: Main scoring loop
from lib.data import count_words as _cw

print("=" * 70)
print("MAIN SCORING LOOP (with attention probing)")
print("=" * 70)

CKPT_PATH = RESULTS_DIR / "checkpoint.json"

all_results = []
start_idx = 0
if CKPT_PATH.exists():
    ckpt = json.loads(CKPT_PATH.read_text())
    if len(ckpt.get('results', [])) > 0:
        saved_queries = [r['query'][:50] for r in ckpt['results']]
        current_queries = [s['query'][:50] for s in samples[:len(saved_queries)]]
        if saved_queries == current_queries:
            all_results = ckpt['results']
            start_idx = len(all_results)
            print(f"Resuming from checkpoint: {start_idx}/{N_SAMPLES}")

if start_idx == 0:
    print(f"Starting fresh: {N_SAMPLES} samples x {len(CONDITIONS)} conditions")
    print(f"Attention probing: {len(PROBE_LAYERS)} layers")

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
              desc="Scoring+Attn"):
    s = samples[i]
    try:
        result = score_sample_with_attention(model, tokenizer, s, DEVICE, PROBE_LAYERS)
    except Exception as e:
        print(f"ERROR at sample {i}: {e}")
        import traceback; traceback.print_exc()
        result = None

    if result is None:
        continue

    result['query'] = s['query'][:50]
    result['query_doc_overlap'] = s['query_doc_overlap']
    result['answer_wc'] = _cw(s['answer'])
    result['doc_wc'] = s['word_count']
    all_results.append(result)

    if (i + 1) % 25 == 0 or i == N_SAMPLES - 1:
        ckpt = {
            'model': MODEL_NAME,
            'n_total': N_SAMPLES,
            'n_conditions': len(CONDITIONS),
            'probe_layers': PROBE_LAYERS,
            'results': all_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        CKPT_PATH.write_text(json.dumps(ckpt))

    if (i + 1) % 50 == 0:
        gc.collect()
        torch.cuda.empty_cache()

elapsed = time.time() - t0
print(f"\nDone: {len(all_results)} samples in {elapsed/60:.1f} min")

print(f"\nNLL summary:")
for cn in CONDITIONS:
    vals = [r[f'nll_{cn}'] for r in all_results]
    print(f"  {cn:<10} NLL={np.mean(vals):.4f}")
""")


# ===== Cell 7: Attention analysis =====
code(r"""# Cell 7: Attention analysis
print("=" * 70)
print("RESULTS: ATTENTION PROBING")
print("=" * 70)

N = len(all_results)

# --- A. NLL replication ---
nll = {}
for cn in CONDITIONS:
    nll[cn] = np.array([r[f'nll_{cn}'] for r in all_results])

print(f"\n--- A. NLL Replication ({N} samples) ---\n")
d_oracle = cohens_d(nll['bare'] - nll['oracle'])
d_random = cohens_d(nll['bare'] - nll['random'])
d_sem = cohens_d(nll['random'] - nll['oracle'])
_, p_sem = stats.ttest_1samp(nll['random'] - nll['oracle'], 0)
print(f"  d_oracle vs bare: {d_oracle:+.3f}")
print(f"  d_random vs bare: {d_random:+.3f}")
print(f"  d_oracle vs random (semantic): {d_sem:+.3f} (p={p_sem:.2e})")

# --- B. Layer-by-layer attention fractions ---
print(f"\n--- B. Doc-Token Attention Fractions by Layer ---\n")
print(f"  (fraction of doc-token attention going to BOS / prime / other-doc)")
print(f"  bare has no prime positions, so frac_prime=0 by definition.\n")

print(f"  {'Layer':>6} | {'--- bare ---':^26} | {'--- random ---':^26} | {'--- oracle ---':^26}")
print(f"  {'':>6} | {'BOS':>8} {'doc':>8} {'ent':>8} | {'BOS':>8} {'prime':>8} {'ent':>8} | {'BOS':>8} {'prime':>8} {'ent':>8}")
print(f"  {'-'*90}")

layer_data = {}
for layer_idx in PROBE_LAYERS:
    layer_data[layer_idx] = {}
    for cn in CONDITIONS:
        bos_key = f'{cn}_L{layer_idx}_frac_bos'
        prime_key = f'{cn}_L{layer_idx}_frac_prime'
        doc_key = f'{cn}_L{layer_idx}_frac_doc'
        ent_key = f'{cn}_L{layer_idx}_entropy'

        # Check if keys exist (some layers might be missing)
        if bos_key not in all_results[0]:
            continue

        bos_vals = np.array([r[bos_key] for r in all_results])
        prime_vals = np.array([r[prime_key] for r in all_results])
        doc_vals = np.array([r[doc_key] for r in all_results])
        ent_vals = np.array([r[ent_key] for r in all_results])

        layer_data[layer_idx][cn] = {
            'bos': bos_vals,
            'prime': prime_vals,
            'doc': doc_vals,
            'entropy': ent_vals,
        }

    if not layer_data[layer_idx]:
        continue

    bare = layer_data[layer_idx].get('bare', {})
    rand = layer_data[layer_idx].get('random', {})
    orac = layer_data[layer_idx].get('oracle', {})

    if bare and rand and orac:
        print(f"  L{layer_idx:>4} | "
              f"{bare['bos'].mean():>8.4f} {bare['doc'].mean():>8.4f} {bare['entropy'].mean():>8.3f} | "
              f"{rand['bos'].mean():>8.4f} {rand['prime'].mean():>8.4f} {rand['entropy'].mean():>8.3f} | "
              f"{orac['bos'].mean():>8.4f} {orac['prime'].mean():>8.4f} {orac['entropy'].mean():>8.3f}")

# --- C. Oracle vs Random attention comparison ---
print(f"\n--- C. Oracle vs Random: Does Prime Type Change Attention? ---\n")
print(f"  For each layer: paired t-test on frac_prime (oracle vs random)")
print(f"  Positive d means oracle prime gets MORE doc-token attention than random prime.\n")

print(f"  {'Layer':>6} {'d(frac_prime)':>14} {'p':>12} {'sig':>5} {'d(entropy)':>12} {'p':>12} {'sig':>5}")
print(f"  {'-'*75}")

for layer_idx in PROBE_LAYERS:
    if layer_idx not in layer_data:
        continue
    rand = layer_data[layer_idx].get('random', {})
    orac = layer_data[layer_idx].get('oracle', {})
    if not rand or not orac:
        continue

    # Compare frac_prime: oracle vs random
    diff_prime = orac['prime'] - rand['prime']
    d_prime = cohens_d(diff_prime)
    _, p_prime = stats.ttest_1samp(diff_prime, 0)
    sig_prime = '***' if p_prime < 0.001 else '**' if p_prime < 0.01 else '*' if p_prime < 0.05 else 'ns'

    # Compare entropy
    diff_ent = orac['entropy'] - rand['entropy']
    d_ent = cohens_d(diff_ent)
    _, p_ent = stats.ttest_1samp(diff_ent, 0)
    sig_ent = '***' if p_ent < 0.001 else '**' if p_ent < 0.01 else '*' if p_ent < 0.05 else 'ns'

    print(f"  L{layer_idx:>4} {d_prime:>+14.3f} {p_prime:>12.2e} {sig_prime:>5} "
          f"{d_ent:>+12.3f} {p_ent:>12.2e} {sig_ent:>5}")

# --- D. BOS attention: bare vs primed ---
print(f"\n--- D. BOS Attention Sink: Bare vs Primed ---\n")
print(f"  Does adding a prime reduce BOS absorption? (bare has no prime -> more BOS)")
print(f"  Positive d = bare has MORE BOS attention.\n")

print(f"  {'Layer':>6} {'bare_bos':>10} {'rand_bos':>10} {'orac_bos':>10} {'d(bare-rand)':>14} {'p':>12} {'sig':>5}")
print(f"  {'-'*75}")

for layer_idx in PROBE_LAYERS:
    if layer_idx not in layer_data:
        continue
    bare = layer_data[layer_idx].get('bare', {})
    rand = layer_data[layer_idx].get('random', {})
    orac = layer_data[layer_idx].get('oracle', {})
    if not bare or not rand:
        continue

    diff_bos = bare['bos'] - rand['bos']
    d_bos = cohens_d(diff_bos)
    _, p_bos = stats.ttest_1samp(diff_bos, 0)
    sig_bos = '***' if p_bos < 0.001 else '**' if p_bos < 0.01 else '*' if p_bos < 0.05 else 'ns'

    print(f"  L{layer_idx:>4} {bare['bos'].mean():>10.4f} {rand['bos'].mean():>10.4f} "
          f"{orac['bos'].mean():>10.4f} {d_bos:>+14.3f} {p_bos:>12.2e} {sig_bos:>5}")

# --- E. Correlation: attention to prime x NLL benefit ---
print(f"\n--- E. Attention-to-Prime x NLL Benefit Correlation ---\n")
print(f"  Does more doc-to-prime attention predict better NLL?")
print(f"  Use last probed layer for this analysis.\n")

last_layer = PROBE_LAYERS[-1]
for cn in ['random', 'oracle']:
    if cn not in layer_data.get(last_layer, {}):
        continue
    frac_prime = layer_data[last_layer][cn]['prime']
    nll_benefit = nll['bare'] - nll[cn]  # positive = condition helps

    r, p = stats.pearsonr(frac_prime, nll_benefit)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {cn}: frac_prime x nll_benefit -> r={r:+.3f}, p={p:.2e} {sig}")

# Also correlate oracle-random attention difference with oracle-random NLL difference
if 'random' in layer_data.get(last_layer, {}) and 'oracle' in layer_data.get(last_layer, {}):
    attn_diff = layer_data[last_layer]['oracle']['prime'] - layer_data[last_layer]['random']['prime']
    nll_diff = nll['random'] - nll['oracle']  # positive = oracle better
    r, p = stats.pearsonr(attn_diff, nll_diff)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"\n  Attention diff (oracle-random) x NLL diff (oracle-random):")
    print(f"    r={r:+.3f}, p={p:.2e} {sig}")
    print(f"    (positive r = samples where oracle gets more attention also show more NLL benefit)")
""")


# ===== Cell 8: Save results =====
code(r"""# Cell 8: Save results + verdict
print("=" * 70)
print("SUMMARY -- Prefix LM Exp 04f: Attention Probing")
print("=" * 70)

d_oracle_v_bare = cohens_d(nll['bare'] - nll['oracle'])
d_random_v_bare = cohens_d(nll['bare'] - nll['random'])

# Summary: average attention fractions across late layers
late_layers = [l for l in PROBE_LAYERS if l >= PROBE_LAYERS[-1] - 12]
print(f"\n  Late layers used for summary: {late_layers}")

for cn in ['random', 'oracle']:
    bos_vals = []
    prime_vals = []
    for l in late_layers:
        if cn in layer_data.get(l, {}):
            bos_vals.append(layer_data[l][cn]['bos'].mean())
            prime_vals.append(layer_data[l][cn]['prime'].mean())
    if bos_vals:
        print(f"  {cn}: mean_bos={np.mean(bos_vals):.4f}, mean_prime={np.mean(prime_vals):.4f}")

print(f"\n  VERDICT:")
# Check if oracle attention pattern differs from random
n_sig_layers = 0
for layer_idx in PROBE_LAYERS:
    rand = layer_data.get(layer_idx, {}).get('random', {})
    orac = layer_data.get(layer_idx, {}).get('oracle', {})
    if not rand or not orac:
        continue
    diff = orac['prime'] - rand['prime']
    _, p = stats.ttest_1samp(diff, 0)
    if p < 0.05:
        n_sig_layers += 1

total_layers = len([l for l in PROBE_LAYERS if 'random' in layer_data.get(l, {})])
print(f"  Oracle vs random frac_prime differs in {n_sig_layers}/{total_layers} probed layers.")

if n_sig_layers > total_layers * 0.5:
    print(f"  Oracle DOES attract different attention than random.")
    print(f"  The semantic signal exists in attention but doesn't survive truncation.")
else:
    print(f"  Oracle attention pattern ~ random. No meaningful attention difference.")
    print(f"  Content literally doesn't affect how doc tokens attend to the prime.")

# Save
summary = {
    'n_samples': N,
    'model': MODEL_NAME,
    'd_oracle': float(d_oracle_v_bare),
    'd_random': float(d_random_v_bare),
    'n_sig_layers': n_sig_layers,
    'total_probed_layers': total_layers,
    'probe_layers': PROBE_LAYERS,
}

final_results = {
    'experiment': 'prefix_lm_exp04f',
    'dataset': 'ms_marco_v1.1',
    'model': MODEL_NAME,
    'n_samples': N,
    'seed': SEED,
    'conditions': CONDITIONS,
    'probe_layers': PROBE_LAYERS,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'summary': summary,
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")
print("Done!")
""")


# ===== Write notebook =====
out_path = "experiments/prefix_lm/04/04f_attention_probing.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
