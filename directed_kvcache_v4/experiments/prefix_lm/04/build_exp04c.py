#!/usr/bin/env python3
# Build Prefix LM Exp 04c notebook: Answer Priming Test.
#
# Exp 04b showed: primes before the doc help (structural d~+0.45), and semantic
# valence matters (positive > negative, d=+0.248). But no instruction beat random.
#
# New question: What if the prime contains the ANSWER? Either:
# - The actual answer (ceiling / positive control)
# - A model-generated answer from the query alone (realistic LLM surrogate)
# - A wrong answer from another sample (style-matched, wrong content)
#
# If answer priming dramatically improves NLL, the model uses answer content
# in the prime to shortcut answer generation. If it doesn't help beyond random,
# the structural mechanism dominates even for content-matched primes.
#
# 6 conditions, single forward pass, N=500.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 0: Markdown =====
md(r"""# Prefix LM Exp 04c: Answer Priming Test

## Motivation

Exp 04b showed primes before the doc help (structural d~+0.45) and semantic valence
matters (positive > negative, d=+0.248). But no positive instruction beat random tokens.

**Question**: What if you prime with the ANSWER itself? Or an LLM-generated surrogate?

## Conditions (6)

All use single forward pass, native causal attention.

| # | Condition | Prime content | What it tests |
|---|-----------|---------------|---------------|
| 1 | `doc_query` | (none) | Baseline |
| 2 | `random_prime` | 8 random words | Structural control |
| 3 | `answer_prime` | actual answer text | Ceiling: answer leak |
| 4 | `wrong_answer` | answer from sample (i+1)%N | Style-matched, wrong content |
| 5 | `answer_5tok` | first 5 tokens of actual answer | Partial answer leak |
| 6 | `model_answer` | model-generated answer (query only, no doc) | LLM surrogate |

Layout: `[BOS, prime, doc, query, answer]` for primed conditions.

## Key Questions

- **A**: Does answer priming beat random? (content-specific benefit beyond structural)
- **B**: How big is the ceiling? (answer_prime vs baseline)
- **C**: Does the model-generated answer help? (LLM surrogate viability)
- **D**: Does wrong_answer match random? (answer style vs content)
- **E**: Does partial leak (5 tokens) help proportionally?
- **F**: How does model answer quality correlate with priming benefit?""")


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

RESULTS_DIR = Path("../../../results/prefix_lm_exp04c")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONDITIONS = [
    "doc_query",       # baseline: [BOS, doc, query, answer]
    "random_prime",    # [BOS, random_8_words, doc, query, answer]
    "answer_prime",    # [BOS, actual_answer, doc, query, answer]
    "wrong_answer",    # [BOS, answer_from_other_sample, doc, query, answer]
    "answer_5tok",     # [BOS, first_5_answer_tokens, doc, query, answer]
    "model_answer",    # [BOS, model_generated_answer, doc, query, answer]
]

print(f"Prefix LM Exp 04c: Answer Priming Test")
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


# ===== Cell 3: Load data =====
code(r"""# Cell 3: Load MS MARCO data (same pipeline as Exp 01-04)
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

print(f"Loaded {len(samples)} samples")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Mean query words: {np.mean([count_words(s['query']) for s in samples]):.0f}")
print(f"Mean answer words: {np.mean([count_words(s['answer']) for s in samples]):.0f}")
""")


# ===== Cell 4: Generate model answers =====
code(r"""# Cell 4: Generate model answers from query alone (no document)
#
# For each sample, prompt the model with just the query and generate a short answer.
# This simulates an LLM surrogate: "What would the model guess without seeing the doc?"
# Generated answers are used as primes in the scoring phase.

GEN_CKPT_PATH = RESULTS_DIR / "generated_answers.json"

if GEN_CKPT_PATH.exists():
    print("Loading cached generated answers...")
    gen_data = json.loads(GEN_CKPT_PATH.read_text())
    for i, s in enumerate(samples):
        s['model_answer'] = gen_data[i]['model_answer']
    print(f"Loaded {len(gen_data)} cached generated answers.")
else:
    print("Generating model answers from query alone (no document)...")
    print("This runs generation for each sample -- ~5-8 minutes.\n")

    t0 = time.time()
    for i in tqdm(range(N_SAMPLES), desc="Generating"):
        query = samples[i]['query']

        # Simple prompt: just the question with a prompt for answer
        prompt = f"Question: {query}\nAnswer:"
        input_ids = tokenizer(prompt, return_tensors="pt",
                              add_special_tokens=True).input_ids.to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=30,
                do_sample=False,       # greedy for reproducibility
                temperature=1.0,
            )

        # Extract only the generated tokens (after the prompt)
        gen_ids = output[0][input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Truncate to first sentence or 30 tokens (whichever is shorter)
        # Stop at first period, newline, or end
        for stop_char in ['\n', '. ', '? ', '! ']:
            idx = gen_text.find(stop_char)
            if idx >= 0:
                gen_text = gen_text[:idx + len(stop_char)].strip()
                break

        samples[i]['model_answer'] = gen_text

        if (i + 1) % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\nGenerated {N_SAMPLES} answers in {elapsed/60:.1f} min")

    # Cache generated answers
    gen_data = [{'query': s['query'][:80], 'model_answer': s['model_answer'],
                 'real_answer': s['answer'][:80]} for s in samples]
    GEN_CKPT_PATH.write_text(json.dumps(gen_data, indent=2))
    print(f"Cached to {GEN_CKPT_PATH}")

# Show examples and stats
print(f"\n--- Generated Answer Examples ---\n")
for i in range(5):
    print(f"  Q: {samples[i]['query'][:70]}")
    print(f"  Model: {samples[i]['model_answer'][:70]}")
    print(f"  Real:  {samples[i]['answer'][:70]}")
    print()

# Token overlap between model answer and real answer
model_overlaps = []
for s in samples:
    m_words = set(re.sub(r'[^\w\s]', '', s['model_answer'].lower()).split()) - STOP_WORDS
    r_words = set(re.sub(r'[^\w\s]', '', s['answer'].lower()).split()) - STOP_WORDS
    union = m_words | r_words
    overlap = len(m_words & r_words) / len(union) if len(union) > 0 else 0.0
    model_overlaps.append(overlap)
    s['model_answer_overlap'] = overlap

print(f"Model answer stats:")
print(f"  Mean tokens: {np.mean([len(tokenizer(s['model_answer'], add_special_tokens=False).input_ids) for s in samples]):.1f}")
print(f"  Mean word overlap with real answer (Jaccard): {np.mean(model_overlaps):.3f}")
print(f"  Overlap > 0: {sum(1 for o in model_overlaps if o > 0)} / {N_SAMPLES}")
""")


# ===== Cell 5: Scoring function =====
code(r"""# Cell 5: score_sample() -- single forward pass, native causal attention

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

    # Prime token IDs
    random_ids = tokenizer(random_prefix, add_special_tokens=False).input_ids
    full_answer_ids = tokenizer(answer, add_special_tokens=False, truncation=True,
                                max_length=64).input_ids
    wrong_answer_ids = tokenizer(wrong_answer_text, add_special_tokens=False,
                                 truncation=True, max_length=64).input_ids
    answer_5tok_ids = full_answer_ids[:5]
    model_answer_ids = tokenizer(model_answer_text, add_special_tokens=False,
                                 truncation=True, max_length=64).input_ids

    n_a = len(answer_ids)
    targets = torch.tensor(answer_ids, dtype=torch.long, device=device)

    # Build sequences: [BOS, (prime), doc, query, answer]
    base = doc_ids + query_ids + answer_ids
    sequences = {
        "doc_query":     [bos_id] + base,
        "random_prime":  [bos_id] + random_ids + base,
        "answer_prime":  [bos_id] + full_answer_ids + base,
        "wrong_answer":  [bos_id] + wrong_answer_ids + base,
        "answer_5tok":   [bos_id] + answer_5tok_ids + base,
        "model_answer":  [bos_id] + model_answer_ids + base,
    }

    result = {
        'n_doc': len(doc_ids),
        'n_query': len(query_ids),
        'n_answer_prime': len(full_answer_ids),
        'n_wrong_answer': len(wrong_answer_ids),
        'n_model_answer': len(model_answer_ids),
    }

    for name, seq in sequences.items():
        input_tensor = torch.tensor([seq], dtype=torch.long, device=device)
        n_before = len(seq) - n_a

        with torch.no_grad():
            out = model(input_ids=input_tensor)

        answer_logits = out.logits[0, n_before - 1 : n_before + n_a - 1, :]
        log_probs = F.log_softmax(answer_logits, dim=-1)
        token_nlls = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        result[f'nll_{name}'] = token_nlls.mean().item()

        del out, input_tensor, answer_logits, log_probs, token_nlls

    return result


print(f"Scoring function defined ({len(CONDITIONS)} conditions per sample).")
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
print("RESULTS: ANSWER PRIMING TEST")
print("=" * 70)

nll = {}
for cn in CONDITIONS:
    nll[cn] = np.array([r[f'nll_{cn}'] for r in all_results])

N = len(all_results)

# --- A. Full ranking ---
print(f"\n--- A. Full Ranking ({N} samples) ---\n")
print(f"  {'Condition':<16} {'Mean NLL':>10} {'d vs baseline':>14} {'d vs random':>12} {'p':>12} {'sig':>5}")
print(f"  {'-'*72}")

ranked = sorted(CONDITIONS, key=lambda cn: nll[cn].mean())
for cn in ranked:
    if cn == "doc_query":
        d_base, d_rand = 0.0, cohens_d(nll['random_prime'] - nll[cn])
        p_base = 1.0
    else:
        diff_base = nll['doc_query'] - nll[cn]
        d_base = cohens_d(diff_base)
        _, p_base = stats.ttest_1samp(diff_base, 0)
        diff_rand = nll['random_prime'] - nll[cn]
        d_rand = cohens_d(diff_rand)
    sig = '***' if p_base < 0.001 else '**' if p_base < 0.01 else '*' if p_base < 0.05 else 'ns'
    print(f"  {cn:<16} {nll[cn].mean():>10.4f} {d_base:>+14.3f} {d_rand:>+12.3f} {p_base:>12.2e} {sig:>5}")

# --- B. Key comparisons ---
print(f"\n--- B. Key Comparisons (positive d = first is better) ---\n")
print(f"  {'Comparison':<50} {'d':>8} {'win%':>7} {'p':>12} {'sig':>5}")
print(f"  {'-'*82}")

comparisons = [
    # Ceiling: answer leak vs baseline
    ("B1. answer_prime vs baseline (CEILING)",
     nll['doc_query'] - nll['answer_prime']),

    # Answer leak vs random (content-specific beyond structural)
    ("B2. answer_prime vs random",
     nll['random_prime'] - nll['answer_prime']),

    # Model answer vs baseline
    ("B3. model_answer vs baseline",
     nll['doc_query'] - nll['model_answer']),

    # Model answer vs random (does LLM surrogate beat random?)
    ("B4. model_answer vs random",
     nll['random_prime'] - nll['model_answer']),

    # Wrong answer vs random (answer style vs content)
    ("B5. wrong_answer vs random",
     nll['random_prime'] - nll['wrong_answer']),

    # Partial leak vs random
    ("B6. answer_5tok vs random",
     nll['random_prime'] - nll['answer_5tok']),

    # Model answer vs wrong answer (quality of LLM surrogate)
    ("B7. model_answer vs wrong_answer",
     nll['wrong_answer'] - nll['model_answer']),

    # Answer prime vs model answer (ceiling - LLM surrogate)
    ("B8. answer_prime vs model_answer (gap)",
     nll['model_answer'] - nll['answer_prime']),
]

for label, diff in comparisons:
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    win = (diff > 0).mean() * 100
    print(f"  {label:<50} {d:>+8.3f} {win:>6.1f}% {p:>12.2e} {sig:>5}")

# --- C. Prime length stats ---
print(f"\n--- C. Prime Length Stats ---\n")
for field, label in [('n_answer_prime', 'answer_prime'),
                      ('n_wrong_answer', 'wrong_answer'),
                      ('n_model_answer', 'model_answer')]:
    vals = [r[field] for r in all_results]
    print(f"  {label:<16} mean={np.mean(vals):.1f}, std={np.std(vals):.1f}, "
          f"min={np.min(vals)}, max={np.max(vals)}")

# --- D. Model answer quality analysis ---
print(f"\n--- D. Model Answer Quality vs Priming Benefit ---\n")

model_overlap = np.array([r['model_answer_overlap'] for r in all_results])
model_benefit = nll['doc_query'] - nll['model_answer']  # positive = model_answer helps

r_val, p_val = stats.pearsonr(model_overlap, model_benefit)
sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
print(f"  Correlation: model_answer_overlap x priming_benefit")
print(f"    r={r_val:+.3f}, p={p_val:.2e} {sig}")

# Split by model answer quality
hi_overlap = model_overlap > np.median(model_overlap)
lo_overlap = ~hi_overlap
print(f"\n  Split by model answer quality (median overlap={np.median(model_overlap):.3f}):")
print(f"  {'Group':<25} {'N':>5} {'d_model_ans':>12} {'d_random':>10}")
print(f"  {'-'*55}")
for label, mask in [("High overlap (good gen)", hi_overlap),
                     ("Low overlap (bad gen)", lo_overlap)]:
    d_ma = cohens_d(model_benefit[mask])
    d_rn = cohens_d((nll['doc_query'] - nll['random_prime'])[mask])
    print(f"  {label:<25} {mask.sum():>5} {d_ma:>+12.3f} {d_rn:>+10.3f}")

# --- E. Answer length subpopulation ---
print(f"\n--- E. Answer Length Subpopulation ---\n")
answer_wc = np.array([r['answer_wc'] for r in all_results])
short = answer_wc <= 5
long = ~short

print(f"  {'Group':<15} {'N':>5} {'d_answer':>10} {'d_model':>10} {'d_wrong':>10} {'d_random':>10}")
print(f"  {'-'*65}")
for label, mask in [("Short (<=5w)", short), ("Long (>5w)", long)]:
    d_ans = cohens_d((nll['doc_query'] - nll['answer_prime'])[mask])
    d_mod = cohens_d((nll['doc_query'] - nll['model_answer'])[mask])
    d_wrg = cohens_d((nll['doc_query'] - nll['wrong_answer'])[mask])
    d_rnd = cohens_d((nll['doc_query'] - nll['random_prime'])[mask])
    print(f"  {label:<15} {mask.sum():>5} {d_ans:>+10.3f} {d_mod:>+10.3f} {d_wrg:>+10.3f} {d_rnd:>+10.3f}")
""")


# ===== Cell 8: Save results =====
code(r"""# Cell 8: Save results and verdict
print("=" * 70)
print("SUMMARY -- Prefix LM Exp 04c: Answer Priming Test")
print("=" * 70)

d_ceiling = cohens_d(nll['doc_query'] - nll['answer_prime'])
d_ans_vs_rand = cohens_d(nll['random_prime'] - nll['answer_prime'])
_, p_ans_vs_rand = stats.ttest_1samp(nll['random_prime'] - nll['answer_prime'], 0)
d_model = cohens_d(nll['doc_query'] - nll['model_answer'])
d_model_vs_rand = cohens_d(nll['random_prime'] - nll['model_answer'])
_, p_model_vs_rand = stats.ttest_1samp(nll['random_prime'] - nll['model_answer'], 0)
d_wrong = cohens_d(nll['doc_query'] - nll['wrong_answer'])
d_wrong_vs_rand = cohens_d(nll['random_prime'] - nll['wrong_answer'])
_, p_wrong_vs_rand = stats.ttest_1samp(nll['random_prime'] - nll['wrong_answer'], 0)

print(f"\n  d_ceiling (answer_prime vs baseline): {d_ceiling:+.3f}")
print(f"  d_answer_vs_random:                   {d_ans_vs_rand:+.3f} (p={p_ans_vs_rand:.2e})")
print(f"  d_model_answer (vs baseline):         {d_model:+.3f}")
print(f"  d_model_vs_random:                    {d_model_vs_rand:+.3f} (p={p_model_vs_rand:.2e})")
print(f"  d_wrong_answer_vs_random:             {d_wrong_vs_rand:+.3f} (p={p_wrong_vs_rand:.2e})")

print(f"\n  VERDICT:")

# Answer ceiling
if d_ans_vs_rand > 0.1 and p_ans_vs_rand < 0.05:
    print(f"  Answer prime beats random (d={d_ans_vs_rand:+.3f}, ***): content matters for answers!")
    print(f"  Total ceiling: d={d_ceiling:+.3f} vs baseline.")
elif p_ans_vs_rand >= 0.05:
    print(f"  Answer prime ~ random (d={d_ans_vs_rand:+.3f}, ns): even the actual answer")
    print(f"  doesn't help beyond structural. Content is irrelevant.")

# Model answer
if p_model_vs_rand < 0.05 and d_model_vs_rand > 0:
    print(f"  LLM surrogate beats random (d={d_model_vs_rand:+.3f}): generated answer adds value!")
    frac = d_model_vs_rand / d_ans_vs_rand if d_ans_vs_rand != 0 else float('nan')
    print(f"  LLM captures {frac:.0%} of the answer-prime ceiling.")
elif p_model_vs_rand >= 0.05:
    print(f"  LLM surrogate ~ random (d={d_model_vs_rand:+.3f}, ns): generation doesn't help.")

# Wrong answer
if p_wrong_vs_rand < 0.05 and d_wrong_vs_rand > 0:
    print(f"  Wrong answer > random (d={d_wrong_vs_rand:+.3f}): answer STYLE helps.")
elif p_wrong_vs_rand >= 0.05:
    print(f"  Wrong answer ~ random (d={d_wrong_vs_rand:+.3f}, ns): answer style doesn't matter.")

# Save
summary = {'n_samples': N, 'model': MODEL_NAME}
for cn in CONDITIONS:
    summary[f'nll_{cn}'] = float(nll[cn].mean())
summary['d_ceiling'] = float(d_ceiling)
summary['d_answer_vs_random'] = float(d_ans_vs_rand)
summary['d_model_vs_baseline'] = float(d_model)
summary['d_model_vs_random'] = float(d_model_vs_rand)

final_results = {
    'experiment': 'prefix_lm_exp04c',
    'dataset': 'ms_marco_v1.1',
    'model': MODEL_NAME,
    'n_samples': N,
    'seed': SEED,
    'conditions': CONDITIONS,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'summary': summary,
    'exp04b_reference': {
        'd_random_vs_baseline': 0.456,
        'd_positive_vs_baseline': 0.431,
    },
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")
print("Done!")
""")


# ===== Write notebook =====
out_path = "experiments/prefix_lm/04/04c_answer_priming.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
