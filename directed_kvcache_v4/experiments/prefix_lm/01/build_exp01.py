#!/usr/bin/env python3
# Build Prefix LM Exp 01 notebook: Prefix LM Enrichment with Gemma 3 12B IT.
#
# Does surrogate-enriched prefix LM attention transfer the enrichment effect
# to a decoder-only model? Using google/gemma-3-12b-it with custom 4D
# attention masks: bidirectional on [surrogate+doc], causal on [query+answer].
#
# Two-pass design mirrors production encoder-decoder scenario:
#   Phase A (offline): Process [BOS, surrogate, doc] -> cache KV states
#   Phase B (online):  Process [query, answer] using cached KVs -> NLL
#
# 10 conditions: 2 attention modes x 4 prefix types + 2 extra controls.
# N=500 MS MARCO samples, SEED=42.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 0: Markdown =====
md(r"""# Prefix LM Exp 01: Prefix LM Enrichment with Gemma 3 12B IT

## Motivation

Experiments 01-10 tested surrogate-enriched encoder caching in encoder-decoder models
(T5Gemma, flan-T5, standard T5, BART). Core finding: prepending a short surrogate
prefix with bidirectional attention improves answer NLL (d=+0.228 to +0.531).

**Next question**: Does this transfer to a **decoder-only** model if we retrofit it
with prefix LM attention (bidirectional on [surrogate+doc], causal on [query+answer])?

## Model

`google/gemma-3-12b-it` in bfloat16 (~24GB on 40GB GPU). Single model for Exp 01.

## Two-Pass Design

Mirrors the encoder-decoder experiments' production scenario:

- **Phase A (offline)**: Process `[BOS, surrogate, doc]` with custom attention mask
  and `use_cache=True` → cache KV states as `past_key_values`
- **Phase B (online)**: Process `[query, answer]` using cached KVs via
  `past_key_values` → compute NLL on answer tokens

Position IDs are sequential: Phase A uses `0..n_prefix-1`, Phase B continues from
`n_prefix..n_prefix+n_cont-1`. This ensures correct RoPE encoding of token distances.

For `_nq` conditions (no query), Phase B input is just `[answer]`. The first answer
token's logit comes from Phase A's last position; remaining answer logits from Phase B.

## Token Layout

```
Phase A: [<bos>] [surrogate_tokens] [doc_tokens]
Phase B: [query_tokens] [answer_tokens]
```

No chat template — raw concatenation for clean experimental control. Conditions
differ only in attention pattern and surrogate content. NLL computed only on answer tokens.

## Conditions (2 x 4 + 2 = 10)

**Attention modes** (2):
- `causal`: Standard lower-triangular throughout (model's native mode)
- `prefix_lm`: Bidirectional on [surrogate+doc], causal on [query+answer]

**Prefix conditions** (4):
- `bare`: No surrogate (just [bos | doc] cached, then [query | answer])
- `oracle_trunc`: Real query as surrogate, masked from Phase B continuation
- `random_trunc`: Random tokens as surrogate, masked from Phase B continuation
- `surr_doc_trunc`: Document keywords (10 keywords) as surrogate, masked from Phase B

**Extra controls** (2):
- `causal_bare_nq`: Causal, no surrogate, no query ([bos | doc] -> [answer]) — floor
- `prefix_lm_oracle_trunc_nq`: Prefix LM with oracle, no query — isolates bidirectional benefit

## Key Metrics

- `d_bidirectional`: prefix_lm_bare - causal_bare → pure bidirectional encoding benefit
- `d_oracle`: prefix_lm_oracle_trunc - prefix_lm_bare → surrogate enrichment under prefix LM
- `d_surr_doc`: prefix_lm_surr_doc_trunc - prefix_lm_bare → keyword surrogate enrichment
- `d_random`: prefix_lm_random_trunc - prefix_lm_bare → structural component
- `structural_fraction`: d_random / d_oracle
- `d_causal_oracle`: causal_oracle_trunc - causal_bare → enrichment under causal attention
  (Note: under causal, doc tokens CAN attend to surrogate causally, so this may be > 0.
  Unlike encoder-decoder where cross-attention mask directly blocks surrogate access.)
- `d_dec_q`: compare _nq vs with-query conditions

## 4D Attention Mask

Gemma 3's `forward()` accepts `attention_mask` as a dict:
```python
attention_mask = {
    "full_attention": mask_4d,       # for global attention layers
    "sliding_attention": mask_4d,    # for sliding window layers
}
```
Float dtype. `0.0` = attend, `torch.finfo(dtype).min` = mask.
Phase A mask: `(1, 1, n_prefix, n_prefix)`.
Phase B mask: `(1, 1, n_cont, n_prefix + n_cont)` — accounts for cached prefix + new tokens.""")


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

RESULTS_DIR = Path("../../../results/prefix_lm_exp01")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 10 conditions: (attention_mode, prefix_type, has_query)
CONDITIONS = [
    ("causal",    "bare",          True),   # causal_bare
    ("causal",    "oracle_trunc",  True),   # causal_oracle_trunc
    ("causal",    "random_trunc",  True),   # causal_random_trunc
    ("causal",    "surr_doc_trunc", True),  # causal_surr_doc_trunc
    ("prefix_lm", "bare",          True),   # prefix_lm_bare
    ("prefix_lm", "oracle_trunc",  True),   # prefix_lm_oracle_trunc
    ("prefix_lm", "random_trunc",  True),   # prefix_lm_random_trunc
    ("prefix_lm", "surr_doc_trunc", True),  # prefix_lm_surr_doc_trunc
    ("causal",    "bare",          False),  # causal_bare_nq
    ("prefix_lm", "oracle_trunc",  False),  # prefix_lm_oracle_trunc_nq
]

def condition_name(mode, prefix, has_query):
    name = f"{mode}_{prefix}" if prefix != "bare" else f"{mode}_bare"
    if not has_query:
        name += "_nq"
    return name

COND_NAMES = [condition_name(m, p, q) for m, p, q in CONDITIONS]

print(f"Prefix LM Exp 01: Enrichment with Gemma 3 12B IT")
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
# Two-pass design:
#   Phase A: Process [BOS, surrogate, doc] -> cache KV states
#   Phase B: Process [query, answer] using cached KVs -> NLL
#
# Each phase gets its own attention mask. Phase A mask controls how prefix
# tokens attend to each other. Phase B mask controls how continuation tokens
# attend to cached prefix (with truncation) and to each other (causally).

def make_phase_a_mask(n_s, n_d, mode="prefix_lm", dtype=torch.bfloat16):
    # Phase A mask for prefix [BOS, surrogate, doc].
    # Returns (1, 1, n_prefix, n_prefix).
    # prefix_lm: fully bidirectional within prefix.
    # causal: standard lower-triangular.
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
print(f"  (Run test_attention_masks.py --model for comprehensive tests)")

del out_default, out_custom
gc.collect(); torch.cuda.empty_cache()
""")


# ===== Cell 4: Data loading =====
code(r"""# Cell 4: Load MS MARCO data + generate surrogates and random prefixes
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

# Generate surrogates
for i, s in enumerate(samples):
    s['surr_doc'] = " ".join(extract_keywords(s['passage'], top_k=10))
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


# ===== Cell 5: Scoring function =====
code(r"""# Cell 5: score_sample() — two-pass scoring
#
# Phase A (offline): Forward [BOS, surr, doc] with use_cache=True -> past_key_values
# Phase B (online):  Forward [query, answer] using cached KVs -> compute NLL
#
# For _nq conditions: Phase B input is [answer] only. The first answer token's
# logit comes from Phase A's last position; remaining from Phase B.

def score_sample(model, tokenizer, sample, device, conditions):
    # Score one MS MARCO sample under all conditions using two-pass design.
    # Returns dict mapping condition_name -> mean NLL on answer tokens.
    passage = sample['passage']
    query = sample['query']
    answer = sample['answer']
    surr_doc = sample['surr_doc']
    random_prefix = sample['random_prefix']

    bos_id = tokenizer.bos_token_id

    # Tokenize segments (no special tokens — we add BOS manually)
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
    random_ids = tokenizer(random_prefix, add_special_tokens=False, truncation=True,
                           max_length=256).input_ids
    surr_doc_ids = tokenizer(surr_doc, add_special_tokens=False, truncation=True,
                             max_length=256).input_ids

    prefix_map = {
        "bare": [],
        "oracle_trunc": oracle_ids,
        "random_trunc": random_ids,
        "surr_doc_trunc": surr_doc_ids,
    }

    targets = torch.tensor(answer_ids, dtype=torch.long, device=device)
    result = {}

    for attn_mode, prefix_type, has_query in conditions:
        cond_name = condition_name(attn_mode, prefix_type, has_query)

        surr_ids = prefix_map[prefix_type]
        n_s = len(surr_ids)
        n_d = len(doc_ids)
        n_q = len(query_ids) if has_query else 0
        n_a = len(answer_ids)
        n_prefix = 1 + n_s + n_d

        # === Phase A: Cache generation ===
        prefix_tokens = [bos_id] + surr_ids + doc_ids
        prefix_input = torch.tensor([prefix_tokens], dtype=torch.long, device=device)

        phase_a_mask = make_phase_a_mask(n_s, n_d, mode=attn_mode)
        phase_a_dict = make_mask_dict(phase_a_mask.to(device))
        phase_a_pos = torch.arange(n_prefix, device=device).unsqueeze(0)

        with torch.no_grad():
            out_a = model(input_ids=prefix_input, attention_mask=phase_a_dict,
                          position_ids=phase_a_pos, use_cache=True)
        past_kv = out_a.past_key_values

        # === Phase B: Evaluation with cached KVs ===
        if has_query:
            cont_tokens = query_ids + answer_ids
        else:
            cont_tokens = list(answer_ids)
        n_cont = len(cont_tokens)

        cont_input = torch.tensor([cont_tokens], dtype=torch.long, device=device)

        truncate = (prefix_type != "bare")
        phase_b_mask = make_phase_b_mask(n_s, n_d, n_q, n_a, truncate=truncate)
        phase_b_dict = make_mask_dict(phase_b_mask.to(device))
        phase_b_pos = torch.arange(n_prefix, n_prefix + n_cont,
                                    device=device).unsqueeze(0)

        with torch.no_grad():
            out_b = model(input_ids=cont_input, attention_mask=phase_b_dict,
                          position_ids=phase_b_pos, past_key_values=past_kv)

        # === Compute NLL on answer tokens ===
        if has_query:
            # Position n_q-1 in Phase B predicts first answer token (a0)
            # Position n_q+n_a-2 predicts last answer token (a_{n_a-1})
            answer_logits = out_b.logits[0, n_q - 1 : n_q + n_a - 1, :]
        else:
            # First answer token predicted by Phase A's last position
            logit_first = out_a.logits[0, -1:, :]
            if n_a > 1:
                # Remaining answer tokens predicted by Phase B positions 0..n_a-2
                logit_rest = out_b.logits[0, :n_a - 1, :]
                answer_logits = torch.cat([logit_first, logit_rest], dim=0)
            else:
                answer_logits = logit_first

        log_probs = F.log_softmax(answer_logits, dim=-1)
        token_nlls = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        mean_nll = token_nlls.mean().item()

        result[f'nll_{cond_name}'] = mean_nll

        del out_a, out_b, past_kv, prefix_input, cont_input
        del phase_a_mask, phase_b_mask, phase_a_dict, phase_b_dict
        del answer_logits, log_probs, token_nlls

    return result


print("Scoring function defined (two-pass, 10 conditions per sample).")
""")


# ===== Cell 6: Main loop =====
code(r"""# Cell 6: Main scoring loop
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
    result['query'] = s['query'][:50]
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
    print(f"  {cn:<35} NLL={np.mean(vals):.4f}")
""")


# ===== Cell 7: Analysis =====
code(r"""# Cell 7: Effect sizes and significance tests
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
print(f"  {'Condition':<35} {'Mean NLL':>10} {'Std':>8}")
print(f"  {'-'*55}")
for cn in COND_NAMES:
    print(f"  {cn:<35} {nll[cn].mean():>10.4f} {nll[cn].std():>8.4f}")

# --- Key comparisons ---
print(f"\n--- Key Comparisons (positive d = condition helps) ---\n")
print(f"  {'Comparison':<50} {'d':>8} {'win%':>7} {'p':>12} {'sig':>5}")
print(f"  {'-'*85}")

comparisons = [
    # Pure bidirectional benefit (no surrogate)
    ("d_bidirectional: plm_bare vs c_bare",
     nll['causal_bare'] - nll['prefix_lm_bare']),

    # Oracle enrichment under prefix LM
    ("d_oracle: plm_oracle vs plm_bare",
     nll['prefix_lm_bare'] - nll['prefix_lm_oracle_trunc']),

    # Surrogate enrichment under prefix LM
    ("d_surr_doc: plm_surr_doc vs plm_bare",
     nll['prefix_lm_bare'] - nll['prefix_lm_surr_doc_trunc']),

    # Structural (random) under prefix LM
    ("d_random: plm_random vs plm_bare",
     nll['prefix_lm_bare'] - nll['prefix_lm_random_trunc']),

    # Oracle under causal (doc attends causally to surrogate)
    ("d_causal_oracle: c_oracle vs c_bare (causal pfx)",
     nll['causal_bare'] - nll['causal_oracle_trunc']),

    # Random under causal
    ("d_causal_random: c_random vs c_bare (causal pfx)",
     nll['causal_bare'] - nll['causal_random_trunc']),

    # Surr_doc under causal
    ("d_causal_surr: c_surr_doc vs c_bare (causal pfx)",
     nll['causal_bare'] - nll['causal_surr_doc_trunc']),

    # Decoder query effect (causal)
    ("d_dec_q: c_bare vs c_bare_nq (query helps?)",
     nll['causal_bare_nq'] - nll['causal_bare']),

    # Oracle bidirectional without query
    ("d_plm_oracle_nq: plm_orc_nq vs c_bare_nq",
     nll['causal_bare_nq'] - nll['prefix_lm_oracle_trunc_nq']),

    # Interaction: bidirectional with vs without surrogate
    ("d_bidir_with_oracle: plm_orc vs c_orc",
     nll['causal_oracle_trunc'] - nll['prefix_lm_oracle_trunc']),
]

for label, diff in comparisons:
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    win = (diff > 0).mean() * 100
    print(f"  {label:<50} {d:>+8.3f} {win:>6.1f}% {p:>12.2e} {sig:>5}")

# Structural fraction
d_oracle_val = cohens_d(nll['prefix_lm_bare'] - nll['prefix_lm_oracle_trunc'])
d_random_val = cohens_d(nll['prefix_lm_bare'] - nll['prefix_lm_random_trunc'])
struct_frac = d_random_val / d_oracle_val if d_oracle_val != 0 else float('nan')
print(f"\n  Structural fraction (d_random / d_oracle): {struct_frac:.1%}")

# --- 2x4 table: attention mode x prefix type ---
print(f"\n--- 2x4 Table: Mean NLL by (attention mode x prefix type) ---\n")
print(f"  {'Prefix':<20} {'Causal':>10} {'Prefix LM':>10} {'Diff (C-P)':>12} {'d':>8}")
print(f"  {'-'*65}")
for prefix in ['bare', 'oracle_trunc', 'random_trunc', 'surr_doc_trunc']:
    c_name = f"causal_{prefix}"
    p_name = f"prefix_lm_{prefix}"
    diff = nll[c_name] - nll[p_name]
    d = cohens_d(diff)
    print(f"  {prefix:<20} {nll[c_name].mean():>10.4f} {nll[p_name].mean():>10.4f} "
          f"{diff.mean():>+12.4f} {d:>+8.3f}")
""")


# ===== Cell 8: Save results =====
code(r"""# Cell 8: Save final results
print("=" * 70)
print("SUMMARY -- Prefix LM Exp 01")
print("=" * 70)

# Collect summary
summary = {
    'n_samples': N,
    'model': MODEL_NAME,
}

# NLL means
for cn in COND_NAMES:
    summary[f'nll_{cn}'] = float(nll[cn].mean())

# Key effect sizes
key_effects = {
    'd_bidirectional': nll['causal_bare'] - nll['prefix_lm_bare'],
    'd_oracle': nll['prefix_lm_bare'] - nll['prefix_lm_oracle_trunc'],
    'd_surr_doc': nll['prefix_lm_bare'] - nll['prefix_lm_surr_doc_trunc'],
    'd_random': nll['prefix_lm_bare'] - nll['prefix_lm_random_trunc'],
    'd_causal_oracle': nll['causal_bare'] - nll['causal_oracle_trunc'],
    'd_dec_q': nll['causal_bare_nq'] - nll['causal_bare'],
    'd_plm_oracle_nq': nll['causal_bare_nq'] - nll['prefix_lm_oracle_trunc_nq'],
}

for name, diff in key_effects.items():
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    summary[name] = float(d)
    summary[f'{name}_p'] = float(p)

summary['structural_fraction'] = float(struct_frac)

# Verdict (use significance, not arbitrary d threshold)
d_bidir = cohens_d(nll['causal_bare'] - nll['prefix_lm_bare'])
_, p_bidir = stats.ttest_1samp(nll['causal_bare'] - nll['prefix_lm_bare'], 0)
d_oracle = cohens_d(nll['prefix_lm_bare'] - nll['prefix_lm_oracle_trunc'])
_, p_oracle = stats.ttest_1samp(nll['prefix_lm_bare'] - nll['prefix_lm_oracle_trunc'], 0)
d_causal_orc = cohens_d(nll['causal_bare'] - nll['causal_oracle_trunc'])
_, p_causal_orc = stats.ttest_1samp(nll['causal_bare'] - nll['causal_oracle_trunc'], 0)

print(f"\n  d_bidirectional (prefix LM bare vs causal bare): {d_bidir:+.3f} (p={p_bidir:.2e})")
print(f"  d_oracle (prefix LM: oracle vs bare):             {d_oracle:+.3f} (p={p_oracle:.2e})")
print(f"  d_causal_oracle (causal: oracle vs bare):          {d_causal_orc:+.3f} (p={p_causal_orc:.2e})")
print(f"  structural_fraction:                               {struct_frac:.1%}")

print(f"\n  VERDICT:")
if p_bidir < 0.05 and d_bidir > 0:
    print(f"  Bidirectional attention HELPS decoder-only models (d={d_bidir:+.3f}, ***).")
elif p_bidir < 0.05 and d_bidir < 0:
    print(f"  Bidirectional attention HURTS decoder-only models (d={d_bidir:+.3f}, ***).")
    print(f"  Model was trained with causal attention -- bidirectional disrupts representations.")
else:
    print(f"  Bidirectional attention has no significant effect (d={d_bidir:+.3f}, ns).")

if p_oracle < 0.05 and d_oracle > 0:
    print(f"  Surrogate enrichment transfers under prefix LM (d={d_oracle:+.3f}, sig).")
else:
    print(f"  Surrogate enrichment does NOT transfer under prefix LM (d={d_oracle:+.3f}, ns).")

if p_causal_orc < 0.05 and d_causal_orc > 0:
    print(f"  Causal prefix WORKS (d={d_causal_orc:+.3f}, ***). Doc attends causally to surrogate.")
    print(f"  The enrichment effect transfers via CAUSAL attention, not bidirectionality.")
else:
    print(f"  Causal prefix has no effect (d={d_causal_orc:+.3f}).")

# Save
final_results = {
    'experiment': 'prefix_lm_exp01',
    'dataset': 'ms_marco_v1.1',
    'model': MODEL_NAME,
    'n_samples': N,
    'seed': SEED,
    'conditions': COND_NAMES,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'summary': summary,
    'enc_dec_references': {
        'T5Gemma_exp01': {
            'd_oracle': 0.228, 'd_random': 0.080, 'structural_frac': 0.35,
        },
    },
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")
print("Done!")
""")


# ===== Write notebook =====
out_path = "experiments/prefix_lm/01/01_prefix_lm_enrichment.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
