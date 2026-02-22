#!/usr/bin/env python3
# Build Prefix LM Exp 02 notebook: Semantic Isolation via Truncation x Content Factorial.
#
# Exp 01 showed causal prefixes help Gemma 3 12B IT (d~+0.45-0.47), but
# structural fraction is 140% -- random tokens work as well as oracle.
# All conditions used truncate=True (Phase B cannot attend to surrogate).
#
# This experiment adds truncate=False ("full") conditions where Phase B tokens
# attend DIRECTLY to cached surrogate KVs. If semantic content matters, oracle
# should benefit MORE from this direct channel than wrong-query or random.
#
# 8 conditions: 3 content x 2 truncation + bare + answer_leak.
# N=500 MS MARCO samples, SEED=42.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 0: Markdown =====
md(r"""# Prefix LM Exp 02: Semantic Isolation via Truncation x Content Factorial

## Motivation

Exp 01 showed causal prefixes help Gemma 3 12B IT: d_oracle=+0.452, d_random=+0.475,
structural fraction 140%. Random tokens work AS WELL as oracle under truncation.

**Key unused lever**: Exp 01 only tested `truncate=True` (Phase B cannot attend to
surrogate positions). With `truncate=False`, query/answer tokens attend DIRECTLY to
cached surrogate KVs -- a **direct semantic channel** that doesn't exist under truncation.

If semantic content matters, oracle should benefit MORE from direct access than
wrong-query or random.

## Conditions (3 x 2 + 2 = 8)

| # | Condition | Content | Truncate | Channel |
|---|-----------|---------|----------|---------|
| 1 | bare | none | n/a | Baseline |
| 2 | oracle_trunc | correct query | yes | Indirect only |
| 3 | wrong_query_trunc | wrong query | yes | Indirect, wrong semantics |
| 4 | random_trunc | random words | yes | Indirect, no semantics |
| 5 | oracle_full | correct query | no | Direct + indirect |
| 6 | wrong_query_full | wrong query | no | Direct wrong + indirect |
| 7 | random_full | random words | no | Direct noise + indirect |
| 8 | answer_leak_trunc | first 5 answer tokens | yes | Positive control |

**Wrong query**: query from sample `(i+1) % N` -- matched style/length, wrong content.

## Key Analyses

- **A**: Structural replication (bare vs random_trunc, expect d~+0.475)
- **B**: Semantic under truncation (oracle_trunc vs wrong_query_trunc)
- **C**: Semantic under full access (oracle_full vs wrong_query_full)
- **D**: Truncation x content interaction (THE critical test)
- **E**: Truncation main effect per content type
- **F**: Positive control (answer_leak_trunc vs random_trunc)
- **G**: Per-sample heterogeneity (correlate with overlap, length)
- **H**: Length-controlled regression (oracle vs wrong_query may differ in length)

## Two-Pass Design

Same as Exp 01. All conditions use causal attention for Phase A.

- **Phase A (offline)**: Process `[BOS, surrogate, doc]` with causal mask, `use_cache=True`
- **Phase B (online)**: Process `[query, answer]` with cached KVs
  - `_trunc`: surrogate positions masked from continuation
  - `_full`: ALL cached positions accessible (direct semantic channel)""")


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

RESULTS_DIR = Path("../../../results/prefix_lm_exp02")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 8 conditions: (prefix_type, truncate)
CONDITIONS = [
    ("bare",         True),   # bare -- baseline
    ("oracle",       True),   # oracle_trunc
    ("wrong_query",  True),   # wrong_query_trunc
    ("random",       True),   # random_trunc
    ("oracle",       False),  # oracle_full
    ("wrong_query",  False),  # wrong_query_full
    ("random",       False),  # random_full
    ("answer_leak",  True),   # answer_leak_trunc -- positive control
]

def condition_name(prefix_type, truncate):
    if prefix_type == "bare":
        return "bare"
    suffix = "trunc" if truncate else "full"
    return f"{prefix_type}_{suffix}"

COND_NAMES = [condition_name(p, t) for p, t in CONDITIONS]

print(f"Prefix LM Exp 02: Semantic Isolation (Truncation x Content)")
print(f"N: {N_SAMPLES}, Conditions: {len(CONDITIONS)}")
print(f"DEVICE: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"\nConditions:")
for cn in COND_NAMES:
    print(f"  {cn}")
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
print(f"BOS token id: {tokenizer.bos_token_id}")
print(f"Model dtype: {model.dtype}")
print(f"Attn implementation: {model.config._attn_implementation}")
""")


# ===== Cell 3: Phase A/B attention masks + sanity check =====
code(r"""# Cell 3: Phase A/B attention masks + sanity check
#
# Reused from Exp 01. Two-pass design:
#   Phase A: Process [BOS, surrogate, doc] -> cache KV states
#   Phase B: Process [query, answer] using cached KVs -> NLL
#
# Phase B truncate parameter controls whether surrogate positions are masked:
#   truncate=True  -> surrogate blocked (indirect channel only)
#   truncate=False -> surrogate accessible (direct + indirect channel)

def make_phase_a_mask(n_s, n_d, mode="causal", dtype=torch.bfloat16):
    # Phase A mask for prefix [BOS, surrogate, doc].
    # Returns (1, 1, n_prefix, n_prefix).
    # Always causal in this experiment (mode parameter kept for API compatibility).
    n_prefix = 1 + n_s + n_d
    min_val = torch.finfo(dtype).min
    if mode == "prefix_lm":
        mask = torch.zeros((n_prefix, n_prefix), dtype=dtype)
    else:
        mask = torch.triu(torch.full((n_prefix, n_prefix), min_val, dtype=dtype),
                          diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def make_phase_b_mask(n_s, n_d, n_q, n_a, truncate=True, dtype=torch.bfloat16):
    # Phase B mask for continuation [query, answer] attending to cached prefix.
    # Returns (1, 1, n_cont, n_prefix + n_cont).
    # Left block: attend to cached BOS + doc; mask surrogate if truncate.
    # Right block: causal self-attention among continuation tokens.
    n_prefix = 1 + n_s + n_d
    n_cont = n_q + n_a
    min_val = torch.finfo(dtype).min

    mask = torch.full((n_cont, n_prefix + n_cont), min_val, dtype=dtype)

    # Attend to all cached prefix positions
    mask[:, :n_prefix] = 0.0

    # Truncation: mask surrogate positions (1..n_s) from continuation
    if truncate and n_s > 0:
        mask[:, 1:1 + n_s] = min_val

    # Causal self-attention among continuation tokens
    mask[:, n_prefix:] = torch.triu(
        torch.full((n_cont, n_cont), min_val, dtype=dtype), diagonal=1
    )

    return mask.unsqueeze(0).unsqueeze(0)


def make_mask_dict(mask_4d):
    # Wrap 4D mask in Gemma 3's dict format (bypasses internal mask creation).
    # Both full and sliding attention layers get the same mask (seq < 1024).
    return {"full_attention": mask_4d, "sliding_attention": mask_4d}


# --- Sanity check: custom causal mask matches default forward ---
print("Mask sanity check: custom causal mask vs default forward...")
test_text = "The quick brown fox jumps over the lazy dog."
test_ids = tokenizer(test_text, return_tensors="pt",
                     add_special_tokens=True).input_ids.to(DEVICE)
Lt = test_ids.shape[1]

with torch.no_grad():
    out_default = model(input_ids=test_ids)

# Build custom causal mask (treat entire sequence as bare prefix, no continuation)
causal_mask = make_phase_a_mask(0, Lt - 1, mode="causal")
causal_dict = make_mask_dict(causal_mask.to(DEVICE))
causal_pos = torch.arange(Lt, device=DEVICE).unsqueeze(0)

with torch.no_grad():
    out_custom = model(input_ids=test_ids, attention_mask=causal_dict,
                       position_ids=causal_pos)

max_diff = (out_default.logits - out_custom.logits).abs().max().item()
print(f"  Max logit diff: {max_diff:.6f}")
assert max_diff < 0.1, (
    f"FAIL: Custom causal mask doesn't match default (max_diff={max_diff:.4f}). "
    f"Dict-based mask API may not work with this model/version.")
print(f"  PASS: Dict-based mask API verified.")

# --- Sanity check: truncate=False gives different mask than truncate=True ---
test_mask_trunc = make_phase_b_mask(5, 10, 3, 5, truncate=True)
test_mask_full = make_phase_b_mask(5, 10, 3, 5, truncate=False)
n_diff = (test_mask_trunc != test_mask_full).sum().item()
print(f"  Trunc vs Full mask: {n_diff} positions differ (expect 5*8=40)")
assert n_diff == 40, f"FAIL: Expected 40 differing positions, got {n_diff}"
print(f"  PASS: Truncation mask correctly blocks 5 surrogate positions from 8 cont tokens.")

del out_default, out_custom
gc.collect(); torch.cuda.empty_cache()
""")


# ===== Cell 4: Data loading =====
code(r"""# Cell 4: Load MS MARCO data + generate surrogates, wrong queries, overlap
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

def extract_keywords(text, top_k=10):
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    content = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    if not content:
        return ["information"]
    counts = Counter(content)
    return [w for w, _ in counts.most_common(top_k)]

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

# Generate surrogates, wrong queries, and overlap stats
for i, s in enumerate(samples):
    # Wrong query: deterministic rotation -- matched style/length, wrong content
    s['wrong_query'] = samples[(i + 1) % len(samples)]['query']

    # Random prefix (same as Exp 01)
    rng = np.random.RandomState(SEED + i + 20000)
    words = rng.choice(WORD_POOL, size=8, replace=False)
    s['random_prefix'] = " ".join(words)

    # Query-document token overlap (Jaccard on content words)
    q_words = set(re.sub(r'[^\w\s]', '', s['query'].lower()).split()) - STOP_WORDS
    d_words = set(re.sub(r'[^\w\s]', '', s['passage'].lower()).split()) - STOP_WORDS
    union = q_words | d_words
    s['query_doc_overlap'] = len(q_words & d_words) / len(union) if len(union) > 0 else 0.0

print(f"Loaded {len(samples)} samples")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Mean query words: {np.mean([count_words(s['query']) for s in samples]):.0f}")
print(f"Mean answer words: {np.mean([count_words(s['answer']) for s in samples]):.0f}")
print(f"Mean query-doc overlap (Jaccard): {np.mean([s['query_doc_overlap'] for s in samples]):.3f}")
print(f"\nExample wrong_query: '{samples[0]['wrong_query'][:80]}...'")
print(f"Example random prefix: '{samples[0]['random_prefix']}'")
""")


# ===== Cell 5: Scoring function =====
code(r"""# Cell 5: score_sample() -- two-pass scoring
#
# Phase A (offline): Forward [BOS, surr, doc] with causal mask, use_cache=True
# Phase B (online):  Forward [query, answer] using cached KVs
#
# For _trunc conditions: Phase B masks surrogate positions (indirect channel only)
# For _full conditions:  Phase B attends to ALL cached positions (direct + indirect)

def score_sample(model, tokenizer, sample, device, conditions):
    # Score one MS MARCO sample under all conditions.
    # Returns dict mapping nll_{cname} -> mean NLL, plus prefix lengths.
    passage = sample['passage']
    query = sample['query']
    answer = sample['answer']
    wrong_query_text = sample['wrong_query']
    random_prefix = sample['random_prefix']

    bos_id = tokenizer.bos_token_id

    # Tokenize segments (no special tokens -- we add BOS manually)
    doc_ids = tokenizer(passage, add_special_tokens=False, truncation=True,
                        max_length=1024).input_ids
    query_ids = tokenizer(query, add_special_tokens=False, truncation=True,
                          max_length=512).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False, truncation=True,
                           max_length=256).input_ids

    if len(answer_ids) == 0:
        return None

    # Surrogate token IDs for each prefix type
    oracle_ids = tokenizer(query, add_special_tokens=False, truncation=True,
                           max_length=256).input_ids
    wrong_query_ids = tokenizer(wrong_query_text, add_special_tokens=False,
                                truncation=True, max_length=256).input_ids
    random_ids = tokenizer(random_prefix, add_special_tokens=False, truncation=True,
                           max_length=256).input_ids
    answer_leak_ids = answer_ids[:5]

    prefix_map = {
        "bare": [],
        "oracle": oracle_ids,
        "wrong_query": wrong_query_ids,
        "random": random_ids,
        "answer_leak": answer_leak_ids,
    }

    n_q = len(query_ids)
    n_a = len(answer_ids)
    n_d = len(doc_ids)

    targets = torch.tensor(answer_ids, dtype=torch.long, device=device)
    result = {}

    # Store prefix lengths for post-hoc length regression
    result['n_oracle'] = len(oracle_ids)
    result['n_wrong_query'] = len(wrong_query_ids)

    for prefix_type, truncate in conditions:
        cname = condition_name(prefix_type, truncate)

        surr_ids = prefix_map[prefix_type]
        n_s = len(surr_ids)
        n_prefix = 1 + n_s + n_d

        # === Phase A: Cache [BOS, surrogate, doc] with causal attention ===
        prefix_tokens = [bos_id] + surr_ids + doc_ids
        prefix_input = torch.tensor([prefix_tokens], dtype=torch.long, device=device)

        phase_a_mask = make_phase_a_mask(n_s, n_d, mode="causal")
        phase_a_dict = make_mask_dict(phase_a_mask.to(device))
        phase_a_pos = torch.arange(n_prefix, device=device).unsqueeze(0)

        with torch.no_grad():
            out_a = model(input_ids=prefix_input, attention_mask=phase_a_dict,
                          position_ids=phase_a_pos, use_cache=True)
        past_kv = out_a.past_key_values

        # === Phase B: Evaluate [query, answer] with cached KVs ===
        cont_tokens = query_ids + answer_ids
        n_cont = len(cont_tokens)
        cont_input = torch.tensor([cont_tokens], dtype=torch.long, device=device)

        phase_b_mask = make_phase_b_mask(n_s, n_d, n_q, n_a, truncate=truncate)
        phase_b_dict = make_mask_dict(phase_b_mask.to(device))
        phase_b_pos = torch.arange(n_prefix, n_prefix + n_cont,
                                    device=device).unsqueeze(0)

        with torch.no_grad():
            out_b = model(input_ids=cont_input, attention_mask=phase_b_dict,
                          position_ids=phase_b_pos, past_key_values=past_kv)

        # === Compute NLL on answer tokens ===
        # Position n_q-1 in Phase B predicts first answer token
        answer_logits = out_b.logits[0, n_q - 1 : n_q + n_a - 1, :]
        log_probs = F.log_softmax(answer_logits, dim=-1)
        token_nlls = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        mean_nll = token_nlls.mean().item()

        result[f'nll_{cname}'] = mean_nll

        del out_a, out_b, past_kv, prefix_input, cont_input
        del phase_a_mask, phase_b_mask, phase_a_dict, phase_b_dict
        del answer_logits, log_probs, token_nlls

    return result


print("Scoring function defined (two-pass, 8 conditions per sample).")
""")


# ===== Cell 6: Main loop =====
code(r"""# Cell 6: Main scoring loop
from lib.data import count_words as _cw

print("=" * 70)
print("MAIN SCORING LOOP")
print("=" * 70)

CKPT_PATH = RESULTS_DIR / "checkpoint.json"

# Resume from checkpoint
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

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
              desc="Scoring"):
    s = samples[i]
    try:
        result = score_sample(model, tokenizer, s, DEVICE, CONDITIONS)
    except Exception as e:
        print(f"ERROR at sample {i}: {e}")
        result = None

    if result is None:
        continue

    # Store metadata for post-hoc analysis
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
            'condition_names': COND_NAMES,
            'results': all_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        CKPT_PATH.write_text(json.dumps(ckpt))

    if (i + 1) % 100 == 0:
        gc.collect()
        torch.cuda.empty_cache()

elapsed = time.time() - t0
print(f"\nDone: {len(all_results)} samples in {elapsed/60:.1f} min")
print(f"\nQuick summary:")
for cn in COND_NAMES:
    vals = [r[f'nll_{cn}'] for r in all_results]
    print(f"  {cn:<25} NLL={np.mean(vals):.4f}")
""")


# ===== Cell 7: Analysis A-F =====
code(r"""# Cell 7: Effect sizes and significance tests (Analyses A-F)
print("=" * 70)
print("RESULTS: EFFECT SIZES AND SIGNIFICANCE")
print("=" * 70)

# Extract NLL arrays
nll = {}
for cn in COND_NAMES:
    nll[cn] = np.array([r[f'nll_{cn}'] for r in all_results])

N = len(all_results)

# --- Mean NLL table ---
print(f"\n--- Mean NLL ({N} samples) ---\n")
print(f"  {'Condition':<25} {'Mean NLL':>10} {'Std':>8}")
print(f"  {'-'*45}")
for cn in COND_NAMES:
    print(f"  {cn:<25} {nll[cn].mean():>10.4f} {nll[cn].std():>8.4f}")

# --- 2x3 factorial table: content x truncation ---
print(f"\n--- 2x3 Factorial: Mean NLL by (content x truncation) ---\n")
print(f"  {'Content':<15} {'Trunc':>10} {'Full':>10} {'Diff(T-F)':>12} {'d':>8}")
print(f"  {'-'*58}")
for content in ['oracle', 'wrong_query', 'random']:
    t_name = f"{content}_trunc"
    f_name = f"{content}_full"
    diff = nll[t_name] - nll[f_name]
    d = cohens_d(diff)
    print(f"  {content:<15} {nll[t_name].mean():>10.4f} {nll[f_name].mean():>10.4f} "
          f"{diff.mean():>+12.4f} {d:>+8.3f}")

print(f"\n  bare:           {nll['bare'].mean():>10.4f}")
print(f"  answer_leak_t:  {nll['answer_leak_trunc'].mean():>10.4f}")

# --- Key comparisons ---
print(f"\n--- Key Comparisons (positive d = first condition is better) ---\n")
print(f"  {'Comparison':<55} {'d':>8} {'win%':>7} {'p':>12} {'sig':>5}")
print(f"  {'-'*90}")

comparisons = [
    # A. Structural replication (expect d ~ +0.475)
    ("A. d_structural: bare vs random_trunc",
     nll['bare'] - nll['random_trunc']),

    # B. Semantic under truncation (indirect channel only)
    ("B. d_semantic_trunc: oracle_trunc vs wrong_query_trunc",
     nll['wrong_query_trunc'] - nll['oracle_trunc']),

    # C. Semantic under full access (direct + indirect channel)
    ("C. d_semantic_full: oracle_full vs wrong_query_full",
     nll['wrong_query_full'] - nll['oracle_full']),

    # D. Truncation x content interaction (THE critical test)
    ("D. interaction: semantic_full - semantic_trunc",
     (nll['wrong_query_full'] - nll['oracle_full']) -
     (nll['wrong_query_trunc'] - nll['oracle_trunc'])),

    # E. Truncation main effect per content type
    ("E1. d_trunc_oracle: full vs trunc (oracle)",
     nll['oracle_trunc'] - nll['oracle_full']),

    ("E2. d_trunc_wrong_query: full vs trunc (wrong_query)",
     nll['wrong_query_trunc'] - nll['wrong_query_full']),

    ("E3. d_trunc_random: full vs trunc (random)",
     nll['random_trunc'] - nll['random_full']),

    # F. Positive control
    ("F. d_answer_leak: answer_leak vs random_trunc",
     nll['random_trunc'] - nll['answer_leak_trunc']),

    # Extra: oracle vs random under each truncation mode
    ("   d_oracle_vs_random_trunc: oracle vs random (trunc)",
     nll['random_trunc'] - nll['oracle_trunc']),

    ("   d_oracle_vs_random_full: oracle vs random (full)",
     nll['random_full'] - nll['oracle_full']),
]

for label, diff in comparisons:
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    win = (diff > 0).mean() * 100
    print(f"  {label:<55} {d:>+8.3f} {win:>6.1f}% {p:>12.2e} {sig:>5}")

# --- Structural fraction under truncation ---
d_oracle_trunc = cohens_d(nll['bare'] - nll['oracle_trunc'])
d_random_trunc = cohens_d(nll['bare'] - nll['random_trunc'])
struct_frac_trunc = d_random_trunc / d_oracle_trunc if d_oracle_trunc != 0 else float('nan')

d_oracle_full = cohens_d(nll['bare'] - nll['oracle_full'])
d_random_full = cohens_d(nll['bare'] - nll['random_full'])
struct_frac_full = d_random_full / d_oracle_full if d_oracle_full != 0 else float('nan')

print(f"\n  Structural fraction (d_random / d_oracle):")
print(f"    Truncation: {struct_frac_trunc:.1%}  (d_oracle={d_oracle_trunc:+.3f}, d_random={d_random_trunc:+.3f})")
print(f"    Full:       {struct_frac_full:.1%}  (d_oracle={d_oracle_full:+.3f}, d_random={d_random_full:+.3f})")
""")


# ===== Cell 8: Post-hoc analysis =====
code(r"""# Cell 8: Post-hoc analysis (Analyses G-H)
print("=" * 70)
print("POST-HOC: HETEROGENEITY AND LENGTH CONTROL")
print("=" * 70)

# --- G. Per-sample heterogeneity ---
# Correlate per-sample semantic effects with sample characteristics

semantic_trunc = nll['wrong_query_trunc'] - nll['oracle_trunc']
semantic_full = nll['wrong_query_full'] - nll['oracle_full']
interaction = semantic_full - semantic_trunc

overlap = np.array([r['query_doc_overlap'] for r in all_results])
answer_wc = np.array([r['answer_wc'] for r in all_results])
doc_wc = np.array([r['doc_wc'] for r in all_results])

print(f"\n--- G. Per-Sample Heterogeneity (N={N}) ---\n")
print(f"  {'Effect':<25} {'x':<18} {'r':>8} {'p':>12} {'sig':>5}")
print(f"  {'-'*72}")

effects = [
    ("semantic_trunc", semantic_trunc),
    ("semantic_full", semantic_full),
    ("interaction", interaction),
]
covariates = [
    ("query_doc_overlap", overlap),
    ("answer_wc", answer_wc),
    ("doc_wc", doc_wc),
]

for eff_name, eff_vals in effects:
    for cov_name, cov_vals in covariates:
        r, p = stats.pearsonr(eff_vals, cov_vals)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {eff_name:<25} {cov_name:<18} {r:>+8.3f} {p:>12.2e} {sig:>5}")

# --- Subpopulation analysis ---
print(f"\n--- G2. Subpopulation Analysis ---\n")

# Split by query-doc overlap (median)
med_overlap = np.median(overlap)
hi_overlap = overlap >= med_overlap
lo_overlap = ~hi_overlap

print(f"  Query-doc overlap split (median={med_overlap:.3f}):")
print(f"  {'Group':<20} {'N':>5} {'d_sem_trunc':>12} {'d_sem_full':>12} {'d_interact':>12}")
print(f"  {'-'*65}")
for label, mask in [("High overlap", hi_overlap), ("Low overlap", lo_overlap)]:
    d_st = cohens_d(semantic_trunc[mask])
    d_sf = cohens_d(semantic_full[mask])
    d_int = cohens_d(interaction[mask])
    print(f"  {label:<20} {mask.sum():>5} {d_st:>+12.3f} {d_sf:>+12.3f} {d_int:>+12.3f}")

# Split by answer length (<=5 vs >5 words, matching Exp 06)
short_ans = answer_wc <= 5
long_ans = ~short_ans

print(f"\n  Answer length split (<=5w vs >5w):")
print(f"  {'Group':<20} {'N':>5} {'d_sem_trunc':>12} {'d_sem_full':>12} {'d_interact':>12}")
print(f"  {'-'*65}")
for label, mask in [("Short (<=5w)", short_ans), ("Long (>5w)", long_ans)]:
    d_st = cohens_d(semantic_trunc[mask])
    d_sf = cohens_d(semantic_full[mask])
    d_int = cohens_d(interaction[mask])
    print(f"  {label:<20} {mask.sum():>5} {d_st:>+12.3f} {d_sf:>+12.3f} {d_int:>+12.3f}")

# --- H. Length-controlled regression ---
print(f"\n--- H. Length-Controlled Regression ---\n")
print(f"  Oracle and wrong_query may differ in token length -> confound.")
print(f"  Regress (NLL_wq - NLL_oracle) on (n_wq - n_oracle) per sample.\n")

n_oracle_arr = np.array([r['n_oracle'] for r in all_results])
n_wq_arr = np.array([r['n_wrong_query'] for r in all_results])
delta_len = n_wq_arr - n_oracle_arr

print(f"  Prefix length stats:")
print(f"    n_oracle:      mean={n_oracle_arr.mean():.1f}, std={n_oracle_arr.std():.1f}")
print(f"    n_wrong_query: mean={n_wq_arr.mean():.1f}, std={n_wq_arr.std():.1f}")
print(f"    delta (wq-orc): mean={delta_len.mean():.1f}, std={delta_len.std():.1f}")

print(f"\n  {'Mode':<10} {'intercept':>10} {'slope':>10} {'R^2':>8} {'p_int':>12} {'p_slope':>12}")
print(f"  {'-'*65}")

for mode, delta_nll in [("trunc", semantic_trunc), ("full", semantic_full)]:
    slope, intercept, r_val, p_val, se = stats.linregress(delta_len, delta_nll)
    # p-value for intercept (length-controlled semantic effect)
    n = len(delta_nll)
    x_bar = delta_len.mean()
    ss_x = np.sum((delta_len - x_bar)**2)
    residuals = delta_nll - (intercept + slope * delta_len)
    mse = np.sum(residuals**2) / (n - 2)
    se_intercept = np.sqrt(mse * (1/n + x_bar**2 / ss_x))
    t_intercept = intercept / se_intercept if se_intercept > 0 else 0
    p_intercept = 2 * stats.t.sf(abs(t_intercept), df=n-2)

    print(f"  {mode:<10} {intercept:>+10.4f} {slope:>+10.5f} {r_val**2:>8.4f} "
          f"{p_intercept:>12.2e} {p_val:>12.2e}")

print(f"\n  Intercept = length-controlled semantic effect.")
print(f"  If intercept is significant: genuine semantic signal beyond length.")
print(f"  If slope is significant: length is a confound.")
""")


# ===== Cell 9: Save results =====
code(r"""# Cell 9: Save final results and verdict
print("=" * 70)
print("SUMMARY -- Prefix LM Exp 02")
print("=" * 70)

summary = {
    'n_samples': N,
    'model': MODEL_NAME,
}

# NLL means
for cn in COND_NAMES:
    summary[f'nll_{cn}'] = float(nll[cn].mean())

# Key effect sizes
key_effects = {
    'd_structural': nll['bare'] - nll['random_trunc'],
    'd_semantic_trunc': nll['wrong_query_trunc'] - nll['oracle_trunc'],
    'd_semantic_full': nll['wrong_query_full'] - nll['oracle_full'],
    'd_interaction': (nll['wrong_query_full'] - nll['oracle_full']) -
                     (nll['wrong_query_trunc'] - nll['oracle_trunc']),
    'd_trunc_oracle': nll['oracle_trunc'] - nll['oracle_full'],
    'd_trunc_wrong_query': nll['wrong_query_trunc'] - nll['wrong_query_full'],
    'd_trunc_random': nll['random_trunc'] - nll['random_full'],
    'd_answer_leak': nll['random_trunc'] - nll['answer_leak_trunc'],
}

for name, diff in key_effects.items():
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    summary[name] = float(d)
    summary[f'{name}_p'] = float(p)

summary['structural_fraction_trunc'] = float(struct_frac_trunc)
summary['structural_fraction_full'] = float(struct_frac_full)

# --- Verdict ---
d_sem_t = cohens_d(nll['wrong_query_trunc'] - nll['oracle_trunc'])
_, p_sem_t = stats.ttest_1samp(nll['wrong_query_trunc'] - nll['oracle_trunc'], 0)
d_sem_f = cohens_d(nll['wrong_query_full'] - nll['oracle_full'])
_, p_sem_f = stats.ttest_1samp(nll['wrong_query_full'] - nll['oracle_full'], 0)
d_inter = cohens_d((nll['wrong_query_full'] - nll['oracle_full']) -
                   (nll['wrong_query_trunc'] - nll['oracle_trunc']))
_, p_inter = stats.ttest_1samp(
    (nll['wrong_query_full'] - nll['oracle_full']) -
    (nll['wrong_query_trunc'] - nll['oracle_trunc']), 0)
d_leak = cohens_d(nll['random_trunc'] - nll['answer_leak_trunc'])
_, p_leak = stats.ttest_1samp(nll['random_trunc'] - nll['answer_leak_trunc'], 0)
d_struct = cohens_d(nll['bare'] - nll['random_trunc'])

print(f"\n  d_structural (bare vs random_trunc):        {d_struct:+.3f}")
print(f"  d_semantic_trunc (oracle vs wq, trunc):     {d_sem_t:+.3f} (p={p_sem_t:.2e})")
print(f"  d_semantic_full (oracle vs wq, full):        {d_sem_f:+.3f} (p={p_sem_f:.2e})")
print(f"  d_interaction (full amplifies semantic?):     {d_inter:+.3f} (p={p_inter:.2e})")
print(f"  d_answer_leak (positive control):            {d_leak:+.3f} (p={p_leak:.2e})")

print(f"\n  VERDICT:")
if p_leak >= 0.05 or d_leak <= 0:
    print(f"  WARNING: Positive control FAILED (d_answer_leak={d_leak:+.3f}, p={p_leak:.2e}).")
    print(f"  The indirect channel may not transmit content at all for this model.")
else:
    print(f"  Positive control PASSED (d_answer_leak={d_leak:+.3f}, ***).")

if p_inter < 0.05 and d_inter > 0:
    print(f"  INTERACTION SIGNIFICANT: Full access amplifies semantic benefit (d={d_inter:+.3f}).")
    print(f"  Direct channel carries semantic signal -- content-specific caching may help.")
elif p_sem_f < 0.05 and d_sem_f > 0:
    print(f"  Semantic signal under full access (d={d_sem_f:+.3f}), but interaction ns.")
    print(f"  Some semantic signal exists, but truncation doesn't modulate it.")
elif p_sem_t < 0.05 and d_sem_t > 0:
    print(f"  Semantic signal even under truncation (d={d_sem_t:+.3f}).")
    print(f"  Content matters through indirect channel alone.")
else:
    print(f"  NO semantic signal detected under either truncation mode.")
    print(f"  Effect is purely structural -- content doesn't matter for caching.")

# Structural dominance summary
if struct_frac_trunc > 0.8:
    print(f"  Structural fraction (trunc): {struct_frac_trunc:.0%} -- structural dominates.")
if struct_frac_full > 0.8:
    print(f"  Structural fraction (full): {struct_frac_full:.0%} -- even with direct access.")

# Save
final_results = {
    'experiment': 'prefix_lm_exp02',
    'dataset': 'ms_marco_v1.1',
    'model': MODEL_NAME,
    'n_samples': N,
    'seed': SEED,
    'conditions': COND_NAMES,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'summary': summary,
    'exp01_references': {
        'd_causal_oracle': 0.452,
        'd_causal_random': 0.475,
        'structural_fraction': 1.40,
    },
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")
print("Done!")
""")


# ===== Write notebook =====
out_path = "experiments/prefix_lm/02/02_semantic_isolation.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
