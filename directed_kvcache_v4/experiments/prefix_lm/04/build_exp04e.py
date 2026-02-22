#!/usr/bin/env python3
# Build Prefix LM Exp 04e notebook: Semantic Amplification & Relevance Hints.
#
# Exp 04d showed: under two-pass truncation (no copy shortcut), ALL content hurts
# vs random. Even the actual answer creates mild interference (d=-0.103). The
# structural mechanism dominates completely.
#
# WHY isn't the semantic effect stronger? Four hypotheses:
#   H1: Questions are too easy (MS MARCO short factoid) -- limited semantic headroom
#   H2: Prime is too short (~8 tokens) -- not enough to shift doc representations
#   H3: Model can't anticipate relevance -- doesn't know which doc aspects matter
#   H4: The mechanism is purely structural -- content literally doesn't matter
#
# This experiment tests H1-H3 by making the semantic signal LOUDER, LONGER, and
# MORE EXPLICIT. If nothing beats random, H4 is confirmed.
#
# 11 conditions, two-pass truncation, N=500.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 0: Markdown =====
md(r"""# Prefix LM Exp 04e: Semantic Amplification & Relevance Hints

## Motivation

Exp 04d showed under two-pass truncation (no copy shortcut), ALL content conditions
hurt vs random. The structural mechanism dominates completely.

**Why isn't the semantic effect stronger?** Four hypotheses:

| # | Hypothesis | Test |
|---|-----------|------|
| H1 | Questions too easy | Stratify by difficulty |
| H2 | Prime too short | Repeat query 3x, 5x |
| H3 | Model can't anticipate relevance | Give explicit relevance hints |
| H4 | Mechanism purely structural | If NOTHING beats random |

## Conditions (11)

All two-pass truncation: Phase A caches `[BOS, prime, doc]`, Phase B evaluates
`[query, answer]` with prime positions masked.

| # | Condition | Content | Hypothesis | Tokens (est) |
|---|-----------|---------|-----------|-------------|
| 1 | `bare` | (none) | baseline | 0 |
| 2 | `random` | 8 random words | structural control | ~9 |
| 3 | `oracle` | real query | standard semantic | ~9 |
| 4 | `oracle_3x` | query x 3 | H2: more signal | ~27 |
| 5 | `oracle_5x` | query x 5 | H2: max signal | ~45 |
| 6 | `random_long` | 40 random words | H2: length control | ~45 |
| 7 | `relevant_sent` | doc sentence with highest answer overlap | H3: relevance | ~15-30 |
| 8 | `irrelevant_sent` | doc sentence with lowest answer overlap | H3: control | ~15-30 |
| 9 | `answer_vocab` | answer words that appear in doc | H3: vocabulary bridge | ~5-12 |
| 10 | `pointer` | "the answer is about [overlap words]" | H3: instructed hint | ~10-18 |
| 11 | `oracle_plus_vocab` | query + answer-doc overlap words | H3: max semantic info | ~15-22 |

## Key Analyses

- **A**: Full ranking (does ANYTHING beat random?)
- **B**: Repetition scaling (oracle_1x vs 3x vs 5x) + length control (random_long)
- **C**: Relevance hints (relevant_sent, answer_vocab, pointer vs random)
- **D**: Maximum semantic info (oracle_plus_vocab vs oracle vs random)
- **E**: Difficulty stratification (split by query-doc overlap, answer length, query length)
- **F**: Structural fraction""")


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

RESULTS_DIR = Path("../../../results/prefix_lm_exp04e")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONDITIONS = [
    "bare",
    "random",
    "oracle",
    "oracle_3x",
    "oracle_5x",
    "random_long",
    "relevant_sent",
    "irrelevant_sent",
    "answer_vocab",
    "pointer",
    "oracle_plus_vocab",
]

print(f"Prefix LM Exp 04e: Semantic Amplification & Relevance Hints")
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


# --- Sanity check ---
print("Mask sanity check: custom causal mask vs default forward...")
test_text = "The quick brown fox jumps over the lazy dog."
test_ids = tokenizer(test_text, return_tensors="pt",
                     add_special_tokens=True).input_ids.to(DEVICE)
Lt = test_ids.shape[1]

with torch.no_grad():
    out_default = model(input_ids=test_ids)

causal_mask = make_phase_a_mask(0, Lt - 1)
causal_dict = make_mask_dict(causal_mask.to(DEVICE))
causal_pos = torch.arange(Lt, device=DEVICE).unsqueeze(0)

with torch.no_grad():
    out_custom = model(input_ids=test_ids, attention_mask=causal_dict,
                       position_ids=causal_pos)

max_diff = (out_default.logits - out_custom.logits).abs().max().item()
print(f"  Max logit diff: {max_diff:.6f}")
assert max_diff < 0.1, f"FAIL: max_diff={max_diff:.4f}"
print(f"  PASS: Dict-based mask API verified.")

del out_default, out_custom
gc.collect(); torch.cuda.empty_cache()
""")


# ===== Cell 4: Data loading + per-sample preparation =====
code(r"""# Cell 4: Load MS MARCO + prepare per-sample fields
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

def content_words(text):
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]

def jaccard(set_a, set_b):
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0

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

# --- Prepare per-sample fields ---
for i, s in enumerate(samples):
    # Random prefix (8 words, same seed as all prior experiments)
    rng = np.random.RandomState(SEED + i + 20000)
    words = rng.choice(WORD_POOL, size=8, replace=False)
    s['random_prefix'] = " ".join(words)

    # Random long prefix (40 words, different seed)
    rng_long = np.random.RandomState(SEED + i + 40000)
    words_long = rng_long.choice(WORD_POOL, size=40, replace=True)
    s['random_long'] = " ".join(words_long)

    # Query-doc overlap
    q_words = set(content_words(s['query']))
    d_words = set(content_words(s['passage']))
    s['query_doc_overlap'] = jaccard(q_words, d_words)

    # --- Sentence extraction for relevant_sent / irrelevant_sent ---
    sents = re.split(r'(?<=[.!?])\s+', s['passage'].strip())
    sents = [sent.strip() for sent in sents if len(sent.strip().split()) >= 3]
    if not sents:
        sents = [s['passage']]

    a_words = set(content_words(s['answer']))

    sent_overlaps = []
    for sent in sents:
        s_words = set(content_words(sent))
        sent_overlaps.append(jaccard(s_words, a_words))

    best_idx = max(range(len(sent_overlaps)), key=lambda j: sent_overlaps[j])
    worst_idx = min(range(len(sent_overlaps)), key=lambda j: sent_overlaps[j])

    s['relevant_sent'] = sents[best_idx]
    s['irrelevant_sent'] = sents[worst_idx]
    s['relevant_sent_overlap'] = sent_overlaps[best_idx]
    s['irrelevant_sent_overlap'] = sent_overlaps[worst_idx]
    s['n_sentences'] = len(sents)

    # --- Answer vocabulary bridge ---
    # Words appearing in BOTH the answer and the document
    overlap_words = sorted(a_words & d_words)
    if not overlap_words:
        # Fall back to top answer content words
        overlap_words = content_words(s['answer'])[:5]
    s['answer_vocab'] = " ".join(overlap_words[:10])
    s['n_answer_vocab_words'] = len(overlap_words[:10])

    # --- Pointer instruction ---
    kw = overlap_words[:5] if overlap_words else content_words(s['query'])[:3]
    s['pointer'] = "the answer is about " + " ".join(kw)

    # --- Oracle + vocabulary ---
    s['oracle_plus_vocab'] = s['query'] + " " + s['answer_vocab']

print(f"\nLoaded {len(samples)} samples")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Mean query words: {np.mean([count_words(s['query']) for s in samples]):.0f}")
print(f"Mean answer words: {np.mean([count_words(s['answer']) for s in samples]):.0f}")

print(f"\n--- Per-sample field stats ---")
print(f"  relevant_sent overlap:   mean={np.mean([s['relevant_sent_overlap'] for s in samples]):.3f}")
print(f"  irrelevant_sent overlap: mean={np.mean([s['irrelevant_sent_overlap'] for s in samples]):.3f}")
print(f"  n_sentences per doc:     mean={np.mean([s['n_sentences'] for s in samples]):.1f}")
print(f"  answer_vocab words:      mean={np.mean([s['n_answer_vocab_words'] for s in samples]):.1f}")

print(f"\n--- Examples ---")
for j in range(3):
    print(f"\n  Sample {j}:")
    print(f"    Q: {samples[j]['query'][:70]}")
    print(f"    A: {samples[j]['answer'][:70]}")
    print(f"    relevant_sent: {samples[j]['relevant_sent'][:70]}...")
    print(f"    irrelevant_sent: {samples[j]['irrelevant_sent'][:70]}...")
    print(f"    answer_vocab: {samples[j]['answer_vocab']}")
    print(f"    pointer: {samples[j]['pointer']}")
""")


# ===== Cell 5: Scoring function =====
code(r"""# Cell 5: score_sample() -- two-pass, all truncate=True, 11 conditions

def score_sample(model, tokenizer, sample, device):
    passage = sample['passage']
    query = sample['query']
    answer = sample['answer']

    bos_id = tokenizer.bos_token_id

    doc_ids = tokenizer(passage, add_special_tokens=False, truncation=True,
                        max_length=1024).input_ids
    query_ids = tokenizer(query, add_special_tokens=False, truncation=True,
                          max_length=512).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False, truncation=True,
                           max_length=256).input_ids

    if len(answer_ids) == 0:
        return None

    # Tokenize all prime variants
    oracle_ids = tokenizer(query, add_special_tokens=False, truncation=True,
                           max_length=256).input_ids
    random_ids = tokenizer(sample['random_prefix'], add_special_tokens=False).input_ids
    random_long_ids = tokenizer(sample['random_long'], add_special_tokens=False).input_ids
    relevant_sent_ids = tokenizer(sample['relevant_sent'], add_special_tokens=False,
                                  truncation=True, max_length=128).input_ids
    irrelevant_sent_ids = tokenizer(sample['irrelevant_sent'], add_special_tokens=False,
                                    truncation=True, max_length=128).input_ids
    answer_vocab_ids = tokenizer(sample['answer_vocab'], add_special_tokens=False).input_ids
    pointer_ids = tokenizer(sample['pointer'], add_special_tokens=False).input_ids
    oracle_plus_vocab_ids = tokenizer(sample['oracle_plus_vocab'], add_special_tokens=False,
                                      truncation=True, max_length=256).input_ids

    prefix_map = {
        "bare": [],
        "random": random_ids,
        "oracle": oracle_ids,
        "oracle_3x": oracle_ids * 3,
        "oracle_5x": oracle_ids * 5,
        "random_long": random_long_ids,
        "relevant_sent": relevant_sent_ids,
        "irrelevant_sent": irrelevant_sent_ids,
        "answer_vocab": answer_vocab_ids,
        "pointer": pointer_ids,
        "oracle_plus_vocab": oracle_plus_vocab_ids,
    }

    n_q = len(query_ids)
    n_a = len(answer_ids)
    n_d = len(doc_ids)

    targets = torch.tensor(answer_ids, dtype=torch.long, device=device)
    result = {
        'n_doc': n_d,
        'n_query': n_q,
    }

    # Record prime token counts for length analysis
    for cname in CONDITIONS:
        result[f'n_prime_{cname}'] = len(prefix_map[cname])

    for cond_name in CONDITIONS:
        surr_ids = prefix_map[cond_name]
        n_s = len(surr_ids)
        n_prefix = 1 + n_s + n_d

        # === Phase A: Cache [BOS, surrogate, doc] ===
        prefix_tokens = [bos_id] + surr_ids + doc_ids
        prefix_input = torch.tensor([prefix_tokens], dtype=torch.long, device=device)

        phase_a_mask = make_phase_a_mask(n_s, n_d)
        phase_a_dict = make_mask_dict(phase_a_mask.to(device))
        phase_a_pos = torch.arange(n_prefix, device=device).unsqueeze(0)

        with torch.no_grad():
            out_a = model(input_ids=prefix_input, attention_mask=phase_a_dict,
                          position_ids=phase_a_pos, use_cache=True)
        past_kv = out_a.past_key_values

        # === Phase B: Evaluate [query, answer] ===
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


print(f"Scoring function defined (two-pass, {len(CONDITIONS)} conditions per sample).")
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
    result['relevant_sent_overlap'] = s['relevant_sent_overlap']
    result['irrelevant_sent_overlap'] = s['irrelevant_sent_overlap']
    result['n_answer_vocab_words'] = s['n_answer_vocab_words']
    result['answer_wc'] = _cw(s['answer'])
    result['query_wc'] = _cw(s['query'])
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
    print(f"  {cn:<20} NLL={np.mean(vals):.4f}")
""")


# ===== Cell 7: Analysis =====
code(r"""# Cell 7: Analysis
print("=" * 70)
print("RESULTS: SEMANTIC AMPLIFICATION & RELEVANCE HINTS")
print("=" * 70)

nll = {}
for cn in CONDITIONS:
    nll[cn] = np.array([r[f'nll_{cn}'] for r in all_results])

N = len(all_results)

# --- A. Full ranking ---
print(f"\n--- A. Full Ranking ({N} samples) ---\n")
print(f"  {'Condition':<20} {'Mean NLL':>10} {'d vs bare':>10} {'d vs random':>12} {'p vs rand':>12} {'sig':>5}")
print(f"  {'-'*78}")

ranked = sorted(CONDITIONS, key=lambda cn: nll[cn].mean())
for cn in ranked:
    if cn == "bare":
        d_base = 0.0
        d_rand = cohens_d(nll['random'] - nll[cn])
        p_rand = 1.0
    else:
        diff_base = nll['bare'] - nll[cn]
        d_base = cohens_d(diff_base)
        diff_rand = nll['random'] - nll[cn]
        d_rand = cohens_d(diff_rand)
        _, p_rand = stats.ttest_1samp(diff_rand, 0)
    sig = '***' if p_rand < 0.001 else '**' if p_rand < 0.01 else '*' if p_rand < 0.05 else 'ns'
    print(f"  {cn:<20} {nll[cn].mean():>10.4f} {d_base:>+10.3f} {d_rand:>+12.3f} {p_rand:>12.2e} {sig:>5}")

# Does ANYTHING beat random?
any_beats_random = False
for cn in CONDITIONS:
    if cn in ("bare", "random"):
        continue
    diff = nll['random'] - nll[cn]
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    if d > 0 and p < 0.05:
        any_beats_random = True
        break

print(f"\n  ANY condition beats random? {'YES' if any_beats_random else 'NO'}")

# --- B. Hypothesis H2: Repetition scaling ---
print(f"\n--- B. H2: Repetition Scaling (does more query signal help?) ---\n")
print(f"  {'Comparison':<50} {'d':>8} {'win%':>7} {'p':>12} {'sig':>5}")
print(f"  {'-'*82}")

h2_tests = [
    ("oracle_3x vs oracle (3x repetition helps?)",
     nll['oracle'] - nll['oracle_3x']),
    ("oracle_5x vs oracle (5x repetition helps?)",
     nll['oracle'] - nll['oracle_5x']),
    ("oracle_5x vs oracle_3x (diminishing returns?)",
     nll['oracle_3x'] - nll['oracle_5x']),
    ("random_long vs random (length alone helps?)",
     nll['random'] - nll['random_long']),
    ("oracle_5x vs random_long (content at matched length?)",
     nll['random_long'] - nll['oracle_5x']),
    ("oracle_5x vs random (max repetition vs structural?)",
     nll['random'] - nll['oracle_5x']),
]

for label, diff in h2_tests:
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    win = (diff > 0).mean() * 100
    print(f"  {label:<50} {d:>+8.3f} {win:>6.1f}% {p:>12.2e} {sig:>5}")

# Mean token counts
print(f"\n  Prime token counts (mean):")
for cn in ['oracle', 'oracle_3x', 'oracle_5x', 'random', 'random_long']:
    vals = [r[f'n_prime_{cn}'] for r in all_results]
    print(f"    {cn:<20} {np.mean(vals):.1f} tokens")

# --- C. Hypothesis H3: Relevance hints ---
print(f"\n--- C. H3: Relevance Hints (does explicit relevance help?) ---\n")
print(f"  {'Comparison':<50} {'d':>8} {'win%':>7} {'p':>12} {'sig':>5}")
print(f"  {'-'*82}")

h3_tests = [
    ("relevant_sent vs random",
     nll['random'] - nll['relevant_sent']),
    ("irrelevant_sent vs random",
     nll['random'] - nll['irrelevant_sent']),
    ("relevant_sent vs irrelevant_sent (relevance?)",
     nll['irrelevant_sent'] - nll['relevant_sent']),
    ("answer_vocab vs random (vocabulary bridge?)",
     nll['random'] - nll['answer_vocab']),
    ("pointer vs random (instructed hint?)",
     nll['random'] - nll['pointer']),
    ("oracle_plus_vocab vs random (max semantic?)",
     nll['random'] - nll['oracle_plus_vocab']),
    ("oracle_plus_vocab vs oracle (vocab adds value?)",
     nll['oracle'] - nll['oracle_plus_vocab']),
]

for label, diff in h3_tests:
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    win = (diff > 0).mean() * 100
    print(f"  {label:<50} {d:>+8.3f} {win:>6.1f}% {p:>12.2e} {sig:>5}")

# --- D. Hypothesis H1: Question difficulty stratification ---
print(f"\n--- D. H1: Question Difficulty Stratification ---\n")

answer_wc = np.array([r['answer_wc'] for r in all_results])
query_wc = np.array([r['query_wc'] for r in all_results])
qd_overlap = np.array([r['query_doc_overlap'] for r in all_results])

# Semantic signal = oracle - random (positive means oracle beats random)
semantic = nll['random'] - nll['oracle']

print(f"  Correlation of semantic signal (d_oracle_vs_random) with difficulty proxies:")
print(f"  {'Proxy':<25} {'r':>8} {'p':>12} {'sig':>5}")
print(f"  {'-'*55}")

for proxy_name, proxy_vals in [("answer_wc", answer_wc),
                                ("query_wc", query_wc),
                                ("query_doc_overlap", qd_overlap)]:
    r, p = stats.pearsonr(semantic, proxy_vals)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {proxy_name:<25} {r:>+8.3f} {p:>12.2e} {sig:>5}")

# Split by answer length
short = answer_wc <= 5
long = answer_wc > 5

print(f"\n  Split by answer length:")
print(f"  {'Group':<15} {'N':>5} {'d_oracle':>10} {'d_random':>10} {'d_orc-rand':>12} {'p':>12} {'sig':>5}")
print(f"  {'-'*75}")
for label, mask in [("Short (<=5w)", short), ("Long (>5w)", long)]:
    d_orc = cohens_d((nll['bare'] - nll['oracle'])[mask])
    d_rnd = cohens_d((nll['bare'] - nll['random'])[mask])
    diff = (nll['random'] - nll['oracle'])[mask]
    d_sem = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {label:<15} {mask.sum():>5} {d_orc:>+10.3f} {d_rnd:>+10.3f} {d_sem:>+12.3f} {p:>12.2e} {sig:>5}")

# Split by query-doc overlap
med_ov = np.median(qd_overlap)
hi_ov = qd_overlap >= med_ov
lo_ov = ~hi_ov

print(f"\n  Split by query-doc overlap (median={med_ov:.3f}):")
print(f"  {'Group':<20} {'N':>5} {'d_oracle':>10} {'d_random':>10} {'d_orc-rand':>12} {'p':>12} {'sig':>5}")
print(f"  {'-'*80}")
for label, mask in [("High overlap", hi_ov), ("Low overlap", lo_ov)]:
    d_orc = cohens_d((nll['bare'] - nll['oracle'])[mask])
    d_rnd = cohens_d((nll['bare'] - nll['random'])[mask])
    diff = (nll['random'] - nll['oracle'])[mask]
    d_sem = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {label:<20} {mask.sum():>5} {d_orc:>+10.3f} {d_rnd:>+10.3f} {d_sem:>+12.3f} {p:>12.2e} {sig:>5}")

# Full condition breakdown by difficulty
print(f"\n  All conditions by difficulty stratum (d vs bare):")
print(f"  {'Condition':<20} {'Short d':>10} {'Long d':>10} {'HiOv d':>10} {'LoOv d':>10}")
print(f"  {'-'*65}")
for cn in ranked:
    ds = cohens_d((nll['bare'] - nll[cn])[short])
    dl = cohens_d((nll['bare'] - nll[cn])[long])
    dh = cohens_d((nll['bare'] - nll[cn])[hi_ov])
    dlo = cohens_d((nll['bare'] - nll[cn])[lo_ov])
    print(f"  {cn:<20} {ds:>+10.3f} {dl:>+10.3f} {dh:>+10.3f} {dlo:>+10.3f}")

# --- E. Length-controlled regression ---
print(f"\n--- E. Length-Controlled Analysis ---\n")
# For each content condition, compute d vs random AFTER controlling for length
# Simple approach: compute d for conditions with similar token counts

print(f"  Prime length stats (mean tokens):")
for cn in CONDITIONS:
    if cn == "bare":
        continue
    vals = [r[f'n_prime_{cn}'] for r in all_results]
    print(f"    {cn:<20} {np.mean(vals):>6.1f} +/- {np.std(vals):>5.1f}")

# Regression: NLL on n_prime_tokens for all non-bare conditions
# Pool all conditions, regress NLL on (n_tokens, condition_dummies)
from itertools import chain
print(f"\n  OLS: NLL ~ n_tokens + condition_dummies (pooled, content conds only)")
content_conds = [cn for cn in CONDITIONS if cn != "bare"]
ys, xs_len, xs_dummies = [], [], []
for cn in content_conds:
    for r in all_results:
        ys.append(r[f'nll_{cn}'])
        xs_len.append(r[f'n_prime_{cn}'])
y = np.array(ys)
x_len = np.array(xs_len)

# Just test: does n_tokens predict NLL?
slope, intercept, r_val, p_val, se = stats.linregress(x_len, y)
print(f"    n_tokens: slope={slope:+.5f}, r2={r_val**2:.4f}, p={p_val:.2e}")

# --- F. Structural fraction ---
print(f"\n--- F. Structural Fraction ---\n")
d_oracle = cohens_d(nll['bare'] - nll['oracle'])
d_random = cohens_d(nll['bare'] - nll['random'])
struct_frac = d_random / d_oracle if d_oracle != 0 else float('nan')
print(f"  d_oracle={d_oracle:+.3f}, d_random={d_random:+.3f}")
print(f"  Structural fraction: {struct_frac:.0%}")
""")


# ===== Cell 8: Save results + verdict =====
code(r"""# Cell 8: Save results + verdict
print("=" * 70)
print("SUMMARY -- Prefix LM Exp 04e")
print("=" * 70)

d_struct = cohens_d(nll['bare'] - nll['random'])
d_oracle_base = cohens_d(nll['bare'] - nll['oracle'])

# Key H2 tests
d_5x_vs_1x = cohens_d(nll['oracle'] - nll['oracle_5x'])
_, p_5x_vs_1x = stats.ttest_1samp(nll['oracle'] - nll['oracle_5x'], 0)

d_5x_vs_randlong = cohens_d(nll['random_long'] - nll['oracle_5x'])
_, p_5x_vs_randlong = stats.ttest_1samp(nll['random_long'] - nll['oracle_5x'], 0)

d_randlong_vs_rand = cohens_d(nll['random'] - nll['random_long'])
_, p_randlong_vs_rand = stats.ttest_1samp(nll['random'] - nll['random_long'], 0)

# Key H3 tests
d_relsent_vs_rand = cohens_d(nll['random'] - nll['relevant_sent'])
_, p_relsent_vs_rand = stats.ttest_1samp(nll['random'] - nll['relevant_sent'], 0)

d_vocab_vs_rand = cohens_d(nll['random'] - nll['answer_vocab'])
_, p_vocab_vs_rand = stats.ttest_1samp(nll['random'] - nll['answer_vocab'], 0)

d_pointer_vs_rand = cohens_d(nll['random'] - nll['pointer'])
_, p_pointer_vs_rand = stats.ttest_1samp(nll['random'] - nll['pointer'], 0)

d_opv_vs_rand = cohens_d(nll['random'] - nll['oracle_plus_vocab'])
_, p_opv_vs_rand = stats.ttest_1samp(nll['random'] - nll['oracle_plus_vocab'], 0)

print(f"\n  H2: Prime too short?")
print(f"    oracle_5x vs oracle:      d={d_5x_vs_1x:+.3f} (p={p_5x_vs_1x:.2e})")
print(f"    oracle_5x vs random_long: d={d_5x_vs_randlong:+.3f} (p={p_5x_vs_randlong:.2e})")
print(f"    random_long vs random:    d={d_randlong_vs_rand:+.3f} (p={p_randlong_vs_rand:.2e})")

print(f"\n  H3: Model can't anticipate relevance?")
print(f"    relevant_sent vs random:    d={d_relsent_vs_rand:+.3f} (p={p_relsent_vs_rand:.2e})")
print(f"    answer_vocab vs random:     d={d_vocab_vs_rand:+.3f} (p={p_vocab_vs_rand:.2e})")
print(f"    pointer vs random:          d={d_pointer_vs_rand:+.3f} (p={p_pointer_vs_rand:.2e})")
print(f"    oracle+vocab vs random:     d={d_opv_vs_rand:+.3f} (p={p_opv_vs_rand:.2e})")

print(f"\n  VERDICT:")

# H2
if p_5x_vs_1x < 0.05 and d_5x_vs_1x > 0:
    print(f"  H2 SUPPORTED: Repeating query 5x helps (d={d_5x_vs_1x:+.3f}). Prime was too short.")
    if p_5x_vs_randlong < 0.05 and d_5x_vs_randlong > 0:
        print(f"    AND content matters at matched length (d={d_5x_vs_randlong:+.3f}).")
    else:
        print(f"    BUT oracle_5x ~ random_long -- still just length/structural.")
else:
    print(f"  H2 REJECTED: Repeating query 5x doesn't help (d={d_5x_vs_1x:+.3f}). Not a length issue.")

# H3
any_h3_works = False
for name, d_val, p_val in [("relevant_sent", d_relsent_vs_rand, p_relsent_vs_rand),
                             ("answer_vocab", d_vocab_vs_rand, p_vocab_vs_rand),
                             ("pointer", d_pointer_vs_rand, p_pointer_vs_rand),
                             ("oracle+vocab", d_opv_vs_rand, p_opv_vs_rand)]:
    if p_val < 0.05 and d_val > 0:
        any_h3_works = True
        print(f"  H3 SUPPORTED: {name} beats random (d={d_val:+.3f}). Relevance helps!")

if not any_h3_works:
    print(f"  H3 REJECTED: No relevance hint beats random. Model doesn't use content.")

# H4
any_beats_random = False
for cn in CONDITIONS:
    if cn in ("bare", "random"):
        continue
    diff = nll['random'] - nll[cn]
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    if d > 0 and p < 0.05:
        any_beats_random = True
        break

if not any_beats_random:
    print(f"  H4 CONFIRMED: NOTHING beats random under truncation.")
    print(f"  The mechanism is purely structural. Content is irrelevant for enrichment.")
    print(f"  Structural fraction: {struct_frac:.0%}")

# Save
summary = {'n_samples': N, 'model': MODEL_NAME}
for cn in CONDITIONS:
    summary[f'nll_{cn}'] = float(nll[cn].mean())
summary['d_structural'] = float(d_struct)
summary['d_oracle'] = float(d_oracle_base)
summary['structural_fraction'] = float(struct_frac)

final_results = {
    'experiment': 'prefix_lm_exp04e',
    'dataset': 'ms_marco_v1.1',
    'model': MODEL_NAME,
    'n_samples': N,
    'seed': SEED,
    'conditions': CONDITIONS,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'summary': summary,
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")
print("Done!")
""")


# ===== Write notebook =====
out_path = "experiments/prefix_lm/04/04e_semantic_amplification.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
