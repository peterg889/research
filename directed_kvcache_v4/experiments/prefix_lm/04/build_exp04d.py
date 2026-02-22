#!/usr/bin/env python3
# Build Prefix LM Exp 04d notebook: Answer Enrichment (No Copy Shortcut).
#
# Exp 04c showed answer_prime beats random (d=+0.261, ***) in single-pass layout
# [BOS, prime, doc, query, answer]. But this includes a COPY SHORTCUT: answer tokens
# can directly attend to the identical answer tokens in the prime.
#
# This experiment removes the copy shortcut using two-pass truncation:
#   Phase A: [BOS, prime, doc] with causal attention -> cache KVs
#   Phase B: [query, answer] -> surrogate positions MASKED from continuation
#
# The answer can ONLY benefit if the prime enriched document representations during
# Phase A. No direct attention from answer to prime.
#
# 7 conditions, N=500. Reuses cached model answers from Exp 04c.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 0: Markdown =====
md(r"""# Prefix LM Exp 04d: Answer Enrichment (No Copy Shortcut)

## Motivation

Exp 04c showed answer priming beats random (d=+0.261, ***) in single-pass layout
`[BOS, prime, doc, query, answer]`. But the answer tokens can **directly attend** to
the identical answer tokens in the prime -- a copy shortcut, not genuine enrichment.

Model-generated answers and wrong answers performed WORSE than random (d=-0.265 and
d=-0.222 respectively). But was this direct semantic interference, or did they actually
poison document representations?

## Design: Two-Pass Truncation

Removes the copy shortcut. Forces ALL benefit to flow through document enrichment:

- **Phase A (offline)**: Process `[BOS, prime, doc]` with causal attention, `use_cache=True`
- **Phase B (online)**: Process `[query, answer]` with cached KVs, **prime positions masked**

The answer can only benefit if the prime made doc representations better during Phase A.
No direct attention from answer to prime.

## Conditions (7)

| # | Condition | Prime in Phase A | What it tests |
|---|-----------|-----------------|---------------|
| 1 | `bare` | (none) | Baseline -- no enrichment |
| 2 | `random` | 8 random words | Structural control |
| 3 | `oracle` | real query | Standard oracle |
| 4 | `answer_prime` | actual answer | Does answer enrich doc? |
| 5 | `wrong_answer` | answer from (i+1)%N | Style-matched, wrong content |
| 6 | `answer_5tok` | first 5 answer tokens | Partial content |
| 7 | `model_answer` | LLM-generated answer | LLM surrogate |

## Key Predictions

- If **answer_prime > random**: genuine enrichment -- answer content helps doc reps
- If **answer_prime ~ random**: Exp 04c content effect was pure copy artifact
- If **model_answer ~ random**: Exp 04c interference was direct-attention, removed by truncation
- If **model_answer < random**: misleading content poisons doc reps during Phase A""")


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

RESULTS_DIR = Path("../../../results/prefix_lm_exp04d")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONDITIONS = [
    "bare",            # no prime -- baseline
    "random",          # 8 random words -- structural control
    "oracle",          # real query -- standard oracle
    "answer_prime",    # actual answer text
    "wrong_answer",    # answer from sample (i+1)%N
    "answer_5tok",     # first 5 answer tokens
    "model_answer",    # LLM-generated answer (from Exp 04c cache)
]

print(f"Prefix LM Exp 04d: Answer Enrichment (No Copy Shortcut)")
print(f"N: {N_SAMPLES}, Conditions: {len(CONDITIONS)}")
print(f"DEVICE: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"\nConditions: {CONDITIONS}")
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
""")


# ===== Cell 3: Mask functions + sanity check =====
code(r"""# Cell 3: Phase A/B attention masks + sanity check
#
# Two-pass design (same as Exp 01/02):
#   Phase A: Process [BOS, surrogate, doc] -> cache KV states (causal attention)
#   Phase B: Process [query, answer] using cached KVs -> NLL
#            Surrogate positions MASKED from continuation (truncate=True always)

def make_phase_a_mask(n_s, n_d, dtype=torch.bfloat16):
    # Phase A mask for [BOS, surrogate, doc] under causal attention.
    # Returns (1, 1, n_prefix, n_prefix).
    n_prefix = 1 + n_s + n_d
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((n_prefix, n_prefix), min_val, dtype=dtype),
                      diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def make_phase_b_mask(n_s, n_d, n_q, n_a, dtype=torch.bfloat16):
    # Phase B mask for [query, answer] attending to cached prefix.
    # Surrogate positions (1..n_s) are ALWAYS masked (truncation).
    # Returns (1, 1, n_cont, n_prefix + n_cont).
    n_prefix = 1 + n_s + n_d
    n_cont = n_q + n_a
    min_val = torch.finfo(dtype).min

    mask = torch.full((n_cont, n_prefix + n_cont), min_val, dtype=dtype)

    # Attend to all cached prefix positions
    mask[:, :n_prefix] = 0.0

    # Truncation: mask surrogate positions (1..n_s) from continuation
    if n_s > 0:
        mask[:, 1:1 + n_s] = min_val

    # Causal self-attention among continuation tokens
    mask[:, n_prefix:] = torch.triu(
        torch.full((n_cont, n_cont), min_val, dtype=dtype), diagonal=1
    )

    return mask.unsqueeze(0).unsqueeze(0)


def make_mask_dict(mask_4d):
    # Wrap 4D mask in Gemma 3's dict format.
    return {"full_attention": mask_4d, "sliding_attention": mask_4d}


# --- Sanity check: custom causal mask matches default forward ---
print("Mask sanity check: custom causal mask vs default forward...")
test_text = "The quick brown fox jumps over the lazy dog."
test_ids = tokenizer(test_text, return_tensors="pt",
                     add_special_tokens=True).input_ids.to(DEVICE)
Lt = test_ids.shape[1]

with torch.no_grad():
    out_default = model(input_ids=test_ids)

# Build custom causal mask (treat entire sequence as bare prefix)
causal_mask = make_phase_a_mask(0, Lt - 1)
causal_dict = make_mask_dict(causal_mask.to(DEVICE))
causal_pos = torch.arange(Lt, device=DEVICE).unsqueeze(0)

with torch.no_grad():
    out_custom = model(input_ids=test_ids, attention_mask=causal_dict,
                       position_ids=causal_pos)

max_diff = (out_default.logits - out_custom.logits).abs().max().item()
print(f"  Max logit diff: {max_diff:.6f}")
assert max_diff < 0.1, (
    f"FAIL: Custom causal mask doesn't match default (max_diff={max_diff:.4f}).")
print(f"  PASS: Dict-based mask API verified.")

del out_default, out_custom
gc.collect(); torch.cuda.empty_cache()
""")


# ===== Cell 4: Load data + model answers =====
code(r"""# Cell 4: Load MS MARCO data + cached model answers from Exp 04c
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

# Generate random prefixes, wrong answers, and overlap
for i, s in enumerate(samples):
    rng = np.random.RandomState(SEED + i + 20000)
    words = rng.choice(WORD_POOL, size=8, replace=False)
    s['random_prefix'] = " ".join(words)
    s['wrong_answer'] = samples[(i + 1) % len(samples)]['answer']

    q_words = set(re.sub(r'[^\w\s]', '', s['query'].lower()).split()) - STOP_WORDS
    d_words = set(re.sub(r'[^\w\s]', '', s['passage'].lower()).split()) - STOP_WORDS
    union = q_words | d_words
    s['query_doc_overlap'] = len(q_words & d_words) / len(union) if len(union) > 0 else 0.0

# Load cached model answers from Exp 04c
GEN_CACHE = Path("../../../results/prefix_lm_exp04c/generated_answers.json")
if GEN_CACHE.exists():
    gen_data = json.loads(GEN_CACHE.read_text())
    # Verify alignment: check first few queries match
    for i in range(min(5, len(gen_data))):
        cached_q = gen_data[i]['query'][:50]
        sample_q = samples[i]['query'][:50]
        assert cached_q == sample_q, (
            f"Sample mismatch at {i}: cached='{cached_q}' vs current='{sample_q}'. "
            f"Data pipelines differ -- cannot reuse cached model answers.")
    for i, s in enumerate(samples):
        s['model_answer'] = gen_data[i]['model_answer']
    print(f"Loaded {len(gen_data)} cached model answers from Exp 04c.")
else:
    # Fallback: generate model answers if cache not available
    print("WARNING: No cached model answers found. Generating from scratch...")
    print("This will add ~28 min. Run Exp 04c first to cache answers.\n")
    t0 = time.time()
    for i in tqdm(range(N_SAMPLES), desc="Generating"):
        query = samples[i]['query']
        prompt = f"Question: {query}\nAnswer:"
        input_ids = tokenizer(prompt, return_tensors="pt",
                              add_special_tokens=True).input_ids.to(DEVICE)
        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=30, do_sample=False, temperature=1.0)
        gen_ids = output[0][input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        for stop_char in ['\n', '. ', '? ', '! ']:
            idx = gen_text.find(stop_char)
            if idx >= 0:
                gen_text = gen_text[:idx + len(stop_char)].strip()
                break
        samples[i]['model_answer'] = gen_text
        if (i + 1) % 100 == 0:
            gc.collect(); torch.cuda.empty_cache()
    elapsed = time.time() - t0
    print(f"Generated {N_SAMPLES} answers in {elapsed/60:.1f} min")

# Compute model answer overlap
model_overlaps = []
for s in samples:
    m_words = set(re.sub(r'[^\w\s]', '', s['model_answer'].lower()).split()) - STOP_WORDS
    r_words = set(re.sub(r'[^\w\s]', '', s['answer'].lower()).split()) - STOP_WORDS
    union = m_words | r_words
    overlap = len(m_words & r_words) / len(union) if len(union) > 0 else 0.0
    model_overlaps.append(overlap)
    s['model_answer_overlap'] = overlap

print(f"\nLoaded {len(samples)} samples")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Mean query words: {np.mean([count_words(s['query']) for s in samples]):.0f}")
print(f"Mean answer words: {np.mean([count_words(s['answer']) for s in samples]):.0f}")
print(f"Mean model answer overlap: {np.mean(model_overlaps):.3f}")
""")


# ===== Cell 5: Scoring function =====
code(r"""# Cell 5: score_sample() -- two-pass, all truncate=True
#
# Phase A (offline): Forward [BOS, surr, doc] with causal mask, use_cache=True
# Phase B (online):  Forward [query, answer] using cached KVs
#                    Surrogate positions MASKED from continuation (always)
#
# This removes the copy shortcut: answer tokens cannot attend to prime tokens.

def score_sample(model, tokenizer, sample, device):
    passage = sample['passage']
    query = sample['query']
    answer = sample['answer']
    wrong_answer_text = sample['wrong_answer']
    random_prefix = sample['random_prefix']
    model_answer_text = sample['model_answer']

    bos_id = tokenizer.bos_token_id

    doc_ids = tokenizer(passage, add_special_tokens=False, truncation=True,
                        max_length=1024).input_ids
    query_ids = tokenizer(query, add_special_tokens=False, truncation=True,
                          max_length=512).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False, truncation=True,
                           max_length=256).input_ids

    if len(answer_ids) == 0:
        return None

    # Prime token IDs for each condition
    oracle_ids = tokenizer(query, add_special_tokens=False, truncation=True,
                           max_length=256).input_ids
    random_ids = tokenizer(random_prefix, add_special_tokens=False).input_ids
    full_answer_ids = tokenizer(answer, add_special_tokens=False, truncation=True,
                                max_length=64).input_ids
    wrong_answer_ids = tokenizer(wrong_answer_text, add_special_tokens=False,
                                 truncation=True, max_length=64).input_ids
    answer_5tok_ids = full_answer_ids[:5]
    model_answer_ids = tokenizer(model_answer_text, add_special_tokens=False,
                                 truncation=True, max_length=64).input_ids

    prefix_map = {
        "bare": [],
        "random": random_ids,
        "oracle": oracle_ids,
        "answer_prime": full_answer_ids,
        "wrong_answer": wrong_answer_ids,
        "answer_5tok": answer_5tok_ids,
        "model_answer": model_answer_ids,
    }

    n_q = len(query_ids)
    n_a = len(answer_ids)
    n_d = len(doc_ids)

    targets = torch.tensor(answer_ids, dtype=torch.long, device=device)
    result = {
        'n_doc': n_d,
        'n_query': n_q,
        'n_oracle': len(oracle_ids),
        'n_answer_prime': len(full_answer_ids),
        'n_wrong_answer': len(wrong_answer_ids),
        'n_model_answer': len(model_answer_ids),
    }

    for cond_name in CONDITIONS:
        surr_ids = prefix_map[cond_name]
        n_s = len(surr_ids)
        n_prefix = 1 + n_s + n_d

        # === Phase A: Cache [BOS, surrogate, doc] with causal attention ===
        prefix_tokens = [bos_id] + surr_ids + doc_ids
        prefix_input = torch.tensor([prefix_tokens], dtype=torch.long, device=device)

        phase_a_mask = make_phase_a_mask(n_s, n_d)
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

        phase_b_mask = make_phase_b_mask(n_s, n_d, n_q, n_a)
        phase_b_dict = make_mask_dict(phase_b_mask.to(device))
        phase_b_pos = torch.arange(n_prefix, n_prefix + n_cont,
                                    device=device).unsqueeze(0)

        with torch.no_grad():
            out_b = model(input_ids=cont_input, attention_mask=phase_b_dict,
                          position_ids=phase_b_pos, past_key_values=past_kv)

        # === Compute NLL on answer tokens ===
        answer_logits = out_b.logits[0, n_q - 1 : n_q + n_a - 1, :]
        log_probs = F.log_softmax(answer_logits, dim=-1)
        token_nlls = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        result[f'nll_{cond_name}'] = token_nlls.mean().item()

        del out_a, out_b, past_kv, prefix_input, cont_input
        del phase_a_mask, phase_b_mask, phase_a_dict, phase_b_dict
        del answer_logits, log_probs, token_nlls

    return result


print(f"Scoring function defined (two-pass, {len(CONDITIONS)} conditions per sample).")
print(f"All conditions use truncation -- prime masked from Phase B.")
""")


# ===== Cell 6: Main loop =====
code(r"""# Cell 6: Main scoring loop
from lib.data import count_words as _cw

print("=" * 70)
print("MAIN SCORING LOOP")
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

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
              desc="Scoring"):
    s = samples[i]
    try:
        result = score_sample(model, tokenizer, s, DEVICE)
    except Exception as e:
        print(f"ERROR at sample {i}: {e}")
        result = None

    if result is None:
        continue

    result['query'] = s['query'][:50]
    result['query_doc_overlap'] = s['query_doc_overlap']
    result['model_answer_overlap'] = s['model_answer_overlap']
    result['answer_wc'] = _cw(s['answer'])
    result['doc_wc'] = s['word_count']
    all_results.append(result)

    if (i + 1) % 25 == 0 or i == N_SAMPLES - 1:
        ckpt = {
            'model': MODEL_NAME,
            'n_total': N_SAMPLES,
            'n_conditions': len(CONDITIONS),
            'condition_names': CONDITIONS,
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
for cn in CONDITIONS:
    vals = [r[f'nll_{cn}'] for r in all_results]
    print(f"  {cn:<16} NLL={np.mean(vals):.4f}")
""")


# ===== Cell 7: Analysis =====
code(r"""# Cell 7: Analysis
print("=" * 70)
print("RESULTS: ANSWER ENRICHMENT (NO COPY SHORTCUT)")
print("=" * 70)

nll = {}
for cn in CONDITIONS:
    nll[cn] = np.array([r[f'nll_{cn}'] for r in all_results])

N = len(all_results)

# --- A. Full ranking ---
print(f"\n--- A. Full Ranking ({N} samples) ---\n")
print(f"  {'Condition':<16} {'Mean NLL':>10} {'d vs bare':>10} {'d vs random':>12} {'p vs bare':>12} {'sig':>5}")
print(f"  {'-'*72}")

ranked = sorted(CONDITIONS, key=lambda cn: nll[cn].mean())
for cn in ranked:
    if cn == "bare":
        d_base = 0.0
        d_rand = cohens_d(nll['random'] - nll[cn])
        p_base = 1.0
    else:
        diff_base = nll['bare'] - nll[cn]
        d_base = cohens_d(diff_base)
        _, p_base = stats.ttest_1samp(diff_base, 0)
        diff_rand = nll['random'] - nll[cn]
        d_rand = cohens_d(diff_rand)
    sig = '***' if p_base < 0.001 else '**' if p_base < 0.01 else '*' if p_base < 0.05 else 'ns'
    print(f"  {cn:<16} {nll[cn].mean():>10.4f} {d_base:>+10.3f} {d_rand:>+12.3f} {p_base:>12.2e} {sig:>5}")

# --- B. Key comparisons ---
print(f"\n--- B. Key Comparisons (positive d = first is better) ---\n")
print(f"  {'Comparison':<55} {'d':>8} {'win%':>7} {'p':>12} {'sig':>5}")
print(f"  {'-'*90}")

comparisons = [
    # Structural replication (should match Exp 01/02: d~+0.45-0.48)
    ("B1. random vs bare (structural replication)",
     nll['bare'] - nll['random']),

    # Oracle vs bare (standard enrichment)
    ("B2. oracle vs bare",
     nll['bare'] - nll['oracle']),

    # THE KEY TEST: answer_prime vs random (content beyond structural)
    ("B3. answer_prime vs random (CONTENT ENRICHMENT?)",
     nll['random'] - nll['answer_prime']),

    # answer_prime vs oracle (does answer content beat query content?)
    ("B4. answer_prime vs oracle",
     nll['oracle'] - nll['answer_prime']),

    # model_answer vs random (LLM surrogate enrichment vs structural)
    ("B5. model_answer vs random (LLM ENRICHMENT?)",
     nll['random'] - nll['model_answer']),

    # wrong_answer vs random (wrong content enrichment)
    ("B6. wrong_answer vs random",
     nll['random'] - nll['wrong_answer']),

    # answer_5tok vs random (partial content)
    ("B7. answer_5tok vs random",
     nll['random'] - nll['answer_5tok']),

    # model_answer vs wrong_answer
    ("B8. model_answer vs wrong_answer",
     nll['wrong_answer'] - nll['model_answer']),
]

for label, diff in comparisons:
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    win = (diff > 0).mean() * 100
    print(f"  {label:<55} {d:>+8.3f} {win:>6.1f}% {p:>12.2e} {sig:>5}")

# --- C. Comparison with Exp 04c (copy shortcut present) ---
print(f"\n--- C. Comparison with Exp 04c (single-pass, copy shortcut present) ---\n")

# Exp 04c reference values (from results)
exp04c_ref = {
    'd_answer_vs_bare': 0.851,
    'd_answer_vs_random': 0.261,
    'd_model_vs_bare': 0.250,
    'd_model_vs_random': -0.265,
    'd_wrong_vs_bare': 0.424,
    'd_wrong_vs_random': -0.222,
    'd_random_vs_bare': 0.456,
}

d_answer_vs_bare_04d = cohens_d(nll['bare'] - nll['answer_prime'])
d_answer_vs_random_04d = cohens_d(nll['random'] - nll['answer_prime'])
d_model_vs_bare_04d = cohens_d(nll['bare'] - nll['model_answer'])
d_model_vs_random_04d = cohens_d(nll['random'] - nll['model_answer'])
d_wrong_vs_bare_04d = cohens_d(nll['bare'] - nll['wrong_answer'])
d_wrong_vs_random_04d = cohens_d(nll['random'] - nll['wrong_answer'])
d_random_vs_bare_04d = cohens_d(nll['bare'] - nll['random'])

print(f"  {'Effect':<30} {'Exp 04c':>10} {'Exp 04d':>10} {'Delta':>10} {'Interpretation'}")
print(f"  {'-'*85}")
rows = [
    ("d_random vs bare",
     exp04c_ref['d_random_vs_bare'], d_random_vs_bare_04d, "structural"),
    ("d_answer vs bare",
     exp04c_ref['d_answer_vs_bare'], d_answer_vs_bare_04d, "answer enrichment + copy"),
    ("d_answer vs random",
     exp04c_ref['d_answer_vs_random'], d_answer_vs_random_04d, "COPY SHORTCUT EFFECT"),
    ("d_model vs bare",
     exp04c_ref['d_model_vs_bare'], d_model_vs_bare_04d, "LLM surrogate total"),
    ("d_model vs random",
     exp04c_ref['d_model_vs_random'], d_model_vs_random_04d, "LLM interference"),
    ("d_wrong vs bare",
     exp04c_ref['d_wrong_vs_bare'], d_wrong_vs_bare_04d, "wrong content total"),
    ("d_wrong vs random",
     exp04c_ref['d_wrong_vs_random'], d_wrong_vs_random_04d, "wrong interference"),
]
for label, v_04c, v_04d, interp in rows:
    delta = v_04d - v_04c
    print(f"  {label:<30} {v_04c:>+10.3f} {v_04d:>+10.3f} {delta:>+10.3f}  {interp}")

# --- D. Model answer quality analysis ---
print(f"\n--- D. Model Answer Quality vs Enrichment Benefit ---\n")

model_overlap = np.array([r['model_answer_overlap'] for r in all_results])
model_benefit = nll['bare'] - nll['model_answer']

r_val, p_val = stats.pearsonr(model_overlap, model_benefit)
sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
print(f"  Correlation: model_answer_overlap x enrichment_benefit")
print(f"    r={r_val:+.3f}, p={p_val:.2e} {sig}")

hi_overlap = model_overlap > np.median(model_overlap)
lo_overlap = ~hi_overlap
print(f"\n  Split by model answer quality (median overlap={np.median(model_overlap):.3f}):")
print(f"  {'Group':<25} {'N':>5} {'d_model':>10} {'d_random':>10} {'d_model_vs_rand':>16}")
print(f"  {'-'*70}")
for label, mask in [("High overlap (good gen)", hi_overlap),
                     ("Low overlap (bad gen)", lo_overlap)]:
    d_ma = cohens_d((nll['bare'] - nll['model_answer'])[mask])
    d_rn = cohens_d((nll['bare'] - nll['random'])[mask])
    d_mr = cohens_d((nll['random'] - nll['model_answer'])[mask])
    print(f"  {label:<25} {mask.sum():>5} {d_ma:>+10.3f} {d_rn:>+10.3f} {d_mr:>+16.3f}")

# --- E. Answer length subpopulation ---
print(f"\n--- E. Answer Length Subpopulation ---\n")
answer_wc = np.array([r['answer_wc'] for r in all_results])
short = answer_wc <= 5
long = ~short

print(f"  {'Group':<15} {'N':>5} {'d_answer':>10} {'d_model':>10} {'d_wrong':>10} {'d_random':>10} {'d_oracle':>10}")
print(f"  {'-'*75}")
for label, mask in [("Short (<=5w)", short), ("Long (>5w)", long)]:
    d_ans = cohens_d((nll['bare'] - nll['answer_prime'])[mask])
    d_mod = cohens_d((nll['bare'] - nll['model_answer'])[mask])
    d_wrg = cohens_d((nll['bare'] - nll['wrong_answer'])[mask])
    d_rnd = cohens_d((nll['bare'] - nll['random'])[mask])
    d_orc = cohens_d((nll['bare'] - nll['oracle'])[mask])
    print(f"  {label:<15} {mask.sum():>5} {d_ans:>+10.3f} {d_mod:>+10.3f} {d_wrg:>+10.3f} {d_rnd:>+10.3f} {d_orc:>+10.3f}")

# --- F. Structural fraction ---
print(f"\n--- F. Structural Fraction ---\n")
d_oracle = cohens_d(nll['bare'] - nll['oracle'])
d_random = cohens_d(nll['bare'] - nll['random'])
struct_frac = d_random / d_oracle if d_oracle != 0 else float('nan')
print(f"  d_oracle={d_oracle:+.3f}, d_random={d_random:+.3f}")
print(f"  Structural fraction: {struct_frac:.0%}")
print(f"  (Exp 01/02 reference: ~105-140%)")
""")


# ===== Cell 8: Save results + verdict =====
code(r"""# Cell 8: Save results and verdict
print("=" * 70)
print("SUMMARY -- Prefix LM Exp 04d: Answer Enrichment (No Copy Shortcut)")
print("=" * 70)

d_struct = cohens_d(nll['bare'] - nll['random'])
d_oracle_base = cohens_d(nll['bare'] - nll['oracle'])

d_ans_vs_rand = cohens_d(nll['random'] - nll['answer_prime'])
_, p_ans_vs_rand = stats.ttest_1samp(nll['random'] - nll['answer_prime'], 0)

d_model_vs_rand = cohens_d(nll['random'] - nll['model_answer'])
_, p_model_vs_rand = stats.ttest_1samp(nll['random'] - nll['model_answer'], 0)

d_wrong_vs_rand = cohens_d(nll['random'] - nll['wrong_answer'])
_, p_wrong_vs_rand = stats.ttest_1samp(nll['random'] - nll['wrong_answer'], 0)

d_5tok_vs_rand = cohens_d(nll['random'] - nll['answer_5tok'])
_, p_5tok_vs_rand = stats.ttest_1samp(nll['random'] - nll['answer_5tok'], 0)

print(f"\n  Structural: d_random vs bare = {d_struct:+.3f}")
print(f"  Oracle:     d_oracle vs bare = {d_oracle_base:+.3f}")
print(f"")
print(f"  answer_prime vs random: d={d_ans_vs_rand:+.3f} (p={p_ans_vs_rand:.2e})")
print(f"  model_answer vs random: d={d_model_vs_rand:+.3f} (p={p_model_vs_rand:.2e})")
print(f"  wrong_answer vs random: d={d_wrong_vs_rand:+.3f} (p={p_wrong_vs_rand:.2e})")
print(f"  answer_5tok  vs random: d={d_5tok_vs_rand:+.3f} (p={p_5tok_vs_rand:.2e})")

print(f"\n  VERDICT:")

# Compare with Exp 04c
d_ans_04c = 0.261  # answer_prime vs random in Exp 04c
d_model_04c = -0.265  # model_answer vs random in Exp 04c

if p_ans_vs_rand < 0.05 and d_ans_vs_rand > 0.1:
    copy_frac = 1.0 - d_ans_vs_rand / d_ans_04c if d_ans_04c != 0 else float('nan')
    print(f"  Answer prime STILL beats random (d={d_ans_vs_rand:+.3f}).")
    print(f"  Genuine enrichment -- not just copy artifact.")
    print(f"  Copy shortcut accounted for ~{copy_frac:.0%} of Exp 04c's d=+0.261.")
elif p_ans_vs_rand >= 0.05:
    print(f"  Answer prime ~ random (d={d_ans_vs_rand:+.3f}, ns).")
    print(f"  Exp 04c's d=+0.261 was ENTIRELY a copy shortcut.")
    print(f"  Answer content does NOT enrich document representations.")
else:
    print(f"  Answer prime slightly worse than random (d={d_ans_vs_rand:+.3f}).")
    print(f"  Answer content may create mild interference even through enrichment.")

if p_model_vs_rand >= 0.05:
    print(f"  Model answer ~ random (d={d_model_vs_rand:+.3f}, ns).")
    print(f"  Exp 04c interference (d={d_model_04c:+.3f}) was direct-attention artifact.")
elif d_model_vs_rand < -0.05 and p_model_vs_rand < 0.05:
    print(f"  Model answer STILL hurts vs random (d={d_model_vs_rand:+.3f}).")
    print(f"  Misleading content poisons doc representations during Phase A.")
else:
    print(f"  Model answer marginally different from random (d={d_model_vs_rand:+.3f}).")

# Save
summary = {'n_samples': N, 'model': MODEL_NAME}
for cn in CONDITIONS:
    summary[f'nll_{cn}'] = float(nll[cn].mean())
summary['d_structural'] = float(d_struct)
summary['d_oracle'] = float(d_oracle_base)
summary['d_answer_vs_random'] = float(d_ans_vs_rand)
summary['d_model_vs_random'] = float(d_model_vs_rand)
summary['d_wrong_vs_random'] = float(d_wrong_vs_rand)
summary['d_answer_vs_random_04c'] = float(d_ans_04c)
summary['d_model_vs_random_04c'] = float(d_model_04c)

final_results = {
    'experiment': 'prefix_lm_exp04d',
    'dataset': 'ms_marco_v1.1',
    'model': MODEL_NAME,
    'n_samples': N,
    'seed': SEED,
    'conditions': CONDITIONS,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'summary': summary,
    'exp04c_reference': {
        'd_answer_vs_random': 0.261,
        'd_model_vs_random': -0.265,
        'd_wrong_vs_random': -0.222,
        'd_random_vs_bare': 0.456,
    },
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")
print("Done!")
""")


# ===== Write notebook =====
out_path = "experiments/prefix_lm/04/04d_answer_enrichment.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
