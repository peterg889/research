#!/usr/bin/env python3
# Generate Exp 09 notebook: Model Generality (T5-XL).
#
# All results so far are on T5Gemma only. T5-XL (3B, standard T5 architecture) uses
# learned relative position bias (not RoPE), no sliding window, standard separated
# cross-attention. If the mechanism exists in T5-XL, it's a general encoder-decoder
# property, not a T5Gemma-specific artifact.
import json

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.split("\n")
                  if isinstance(source, str) else source})


def code(source):
    lines = source.split("\n") if isinstance(source, str) else source
    formatted = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type": "code", "metadata": {}, "source": formatted,
                  "outputs": [], "execution_count": None})


# ============================================================
# Cell 1: Markdown title
# ============================================================
md(r"""# Experiment 09: Model Generality -- T5-XL
## Does the structural mechanism generalize beyond T5Gemma?

### Motivation
All Exps 01-08 used T5Gemma 2 (4B-4B), a custom Gemma-based encoder-decoder with:
- Dual-frequency RoPE
- Sliding window + full attention layers
- Merged self+cross attention in decoder
- GQA (4 KV heads -> 8 Q heads)

T5-XL (3B) uses a completely different architecture:
- **Learned relative position bias** (not RoPE)
- **No sliding window** (full attention in every layer)
- **Standard separated cross-attention** (not merged)
- **Multi-head attention** (not GQA)

If the structural benefit exists in T5-XL, it means the mechanism is a general
property of **encoder-decoder architectures with bidirectional attention**, not an
artifact of T5Gemma's specific design.

### Architecture comparison
| Property | T5Gemma 2 4B-4B | T5-XL |
|----------|-----------------|-------|
| Parameters | ~8B (4B+4B) | 3B |
| Position encoding | Dual-frequency RoPE | Learned relative bias |
| Attention pattern | Sliding window + full | Full attention only |
| Cross-attention | Merged with self-attn | Separate layer |
| KV heads | 4 (GQA) | 32 (MHA) |
| Hidden size | 2560 | 2048 |
| Encoder layers | 34 | 24 |
| Decoder layers | 34 | 24 |
| Pretraining | Text + image | Text only |

### Conditions (8)
| # | Condition | Prefix | Purpose |
|---|-----------|--------|---------|
| 1 | bare | (none) | lower bound |
| 2 | oracle\_trunc | real query | upper bound |
| 3 | random\_trunc | ~20 random words | structural control |
| 4 | scrambled\_oracle\_trunc | query words shuffled | vocabulary control |
| 5 | surr\_template\_trunc | "What is [kw]?" | best heuristic |
| 6 | static\_fact\_trunc | "What are the key facts?" | content-agnostic |
| 7 | repeat\_the\_trunc | "the" x 10 | attention sink test |
| 8 | single\_word\_trunc | "X" | minimal prefix |

### N=500 (same samples as Exp 02 for direct comparison)
""")

# ============================================================
# Cell 2: Setup
# ============================================================
code("""# Cell 2: Setup
import os
os.umask(0o000)

import sys
import json
import time
import re
import gc
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, "../..")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("../../results/exp09")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

N_SAMPLES = 500
MODEL_NAME = "google/t5-xl-lm-adapt"
MODEL_FALLBACK = "google-t5/t5-xl"
N_BONFERRONI = 7  # 7 non-bare conditions

print(f"Exp 09: Model Generality -- T5-XL")
print(f"Model: {MODEL_NAME} (fallback: {MODEL_FALLBACK})")
print(f"N: {N_SAMPLES}")
print(f"Bonferroni comparisons: {N_BONFERRONI}")
print(f"CUDA: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")""")

# ============================================================
# Cell 3: Load T5-XL model
# ============================================================
code(r"""# Cell 3: Load T5-XL model
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Try LM-adapted version first (better for NLL scoring), fall back to standard
model_loaded = False
for name in [MODEL_NAME, MODEL_FALLBACK]:
    try:
        print(f"Trying {name}...")
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = T5ForConditionalGeneration.from_pretrained(
            name, device_map="auto", torch_dtype=torch.bfloat16,
        )
        model.eval()
        MODEL_NAME = name
        model_loaded = True
        print(f"Loaded {name}")
        break
    except Exception as e:
        print(f"  Failed: {e}")
        continue

if not model_loaded:
    raise RuntimeError("Could not load T5-XL model from any source")

DEVICE = next(model.parameters()).device
print(f"Model: {MODEL_NAME}")
print(f"dtype: {next(model.parameters()).dtype}")
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Architecture info
n_enc_layers = model.config.num_layers
n_dec_layers = model.config.num_decoder_layers
n_heads = model.config.num_heads
d_model = model.config.d_model
print(f"Encoder layers: {n_enc_layers}, Decoder layers: {n_dec_layers}")
print(f"Attention heads: {n_heads}, d_model: {d_model}")
print(f"Relative attention bias: {model.config.relative_attention_num_buckets} buckets, "
      f"{model.config.relative_attention_max_distance} max distance")""")

# ============================================================
# Cell 4: Scoring helpers (adapted for T5 API)
# ============================================================
code(r"""# Cell 4: Scoring helpers adapted for T5-XL

def score_nll(encoder_text, answer_text, prefix_token_count=0, truncate=False):
    # Score NLL of answer given encoder text, with optional prefix truncation.
    # Adapted for T5ForConditionalGeneration API.
    enc_ids = tokenizer(encoder_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=1024).input_ids.to(DEVICE)
    total_enc_len = enc_ids.shape[1]

    enc_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        encoder_outputs = model.encoder(
            input_ids=enc_ids, attention_mask=enc_mask
        )

    # Cross-attention mask for decoder
    if truncate and prefix_token_count > 0:
        cross_attn_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)
        cross_attn_mask[:, :prefix_token_count] = 0
    else:
        cross_attn_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)

    ans_ids = tokenizer(answer_text, return_tensors="pt",
                        add_special_tokens=False, truncation=True,
                        max_length=256).input_ids.to(DEVICE)

    if ans_ids.shape[1] == 0:
        return 0.0

    with torch.no_grad():
        outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=cross_attn_mask,
            labels=ans_ids,
        )

    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[0].gather(1, ans_ids[0].unsqueeze(1)).squeeze(1)
    mean_nll = -token_log_probs.mean().item()

    del encoder_outputs, outputs, logits, log_probs
    return mean_nll


def count_prefix_tokens(prefix_text, document_text):
    # Count prefix tokens in [prefix + newline + document].
    full_text = prefix_text + "\n" + document_text
    full_ids = tokenizer(full_text, add_special_tokens=True).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True).input_ids
    return len(full_ids) - len(doc_ids)


# === Surrogate generation ===
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

def make_surrogate_template(passage):
    content_words = extract_keywords(passage)
    if not content_words:
        return "What is this about?"
    counts = Counter(content_words)
    top_word = counts.most_common(1)[0][0]
    return f"What is {top_word}?"

STATIC_FACT = "What are the key facts I need to know?"

# Quick sanity check: verify tokenizer works
test_text = "What is information?"
test_ids = tokenizer(test_text, add_special_tokens=True).input_ids
print(f"Tokenizer test: '{test_text}' -> {len(test_ids)} tokens")
print(f"  Token IDs: {test_ids[:10]}")
print(f"  Vocab size: {tokenizer.vocab_size}")
print("Helpers defined.")""")

# ============================================================
# Cell 5: Load MS MARCO data (same 500 as Exp 02)
# ============================================================
code(r"""# Cell 5: Load MS MARCO data (same 500 samples as Exp 02)
from lib.data import count_words
from datasets import load_dataset

print("Loading MS MARCO v1.1 validation...")
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

samples = []
for item in ds:
    if len(samples) >= N_SAMPLES * 3:
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
            samples.append({
                'passage': pt, 'query': query, 'answer': answer,
                'word_count': wc,
            })
            break

np.random.seed(SEED)
np.random.shuffle(samples)
samples = samples[:N_SAMPLES]
del ds
gc.collect()

# Verify alignment with Exp 02
EXP02_CHECKPOINT = Path("../../results/exp02/checkpoint.json")
if EXP02_CHECKPOINT.exists():
    exp02_ckpt = json.loads(EXP02_CHECKPOINT.read_text())
    exp02_results = exp02_ckpt.get('results', [])
    matched = 0
    for i in range(min(20, len(exp02_results))):
        if samples[i]['query'] == exp02_results[i]['query']:
            matched += 1
    print(f"Exp 02 alignment check: {matched}/20 queries match")
else:
    print("Exp 02 checkpoint not found (alignment check skipped)")

print(f"Selected {len(samples)} samples")
print(f"Mean doc words: {np.mean([s['word_count'] for s in samples]):.0f}")""")

# ============================================================
# Cell 6: Generate conditions + verification examples
# ============================================================
code(r"""# Cell 6: Generate conditions
COND_NAMES = [
    'bare', 'oracle_trunc', 'random_trunc', 'scrambled_oracle_trunc',
    'surr_template_trunc', 'static_fact_trunc', 'repeat_the_trunc',
    'single_word_trunc',
]

for i, s in enumerate(samples):
    # Random: ~20 words from unrelated passage
    other_idx = (i + N_SAMPLES // 2) % len(samples)
    other_passage = samples[other_idx]['passage']
    s['surr_random'] = " ".join(other_passage.split()[:20])

    # Scrambled oracle: same words, random order
    query_words = s['query'].split()
    rng = np.random.RandomState(SEED + i)
    shuffled = list(query_words)
    rng.shuffle(shuffled)
    s['scrambled_oracle'] = " ".join(shuffled)

    # Template: "What is [kw]?"
    s['surr_template'] = make_surrogate_template(s['passage'])

    # Repeat the: "the" x 10
    s['repeat_the'] = " ".join(["the"] * 10)

    # Single word: "X"
    s['single_word'] = "X"

# Show examples
print(f"CONDITION EXAMPLES")
print("=" * 70)
ex = samples[0]
print(f"\nQuery:   {ex['query'][:80]}")
print(f"Answer:  {ex['answer'][:80]}")
print(f"Passage: {ex['passage'][:80]}...")

# Verify token counts
print(f"\n--- Token counts (T5-XL tokenizer) ---")
for cond in COND_NAMES:
    if cond == 'bare':
        toks = len(tokenizer(ex['passage'], add_special_tokens=True).input_ids)
        print(f"  {cond:<28s}: {toks} total tokens")
    else:
        key = cond.replace('_trunc', '')
        surr_text = ex.get(key, '') if key != 'oracle' else ex['query']
        full_text = surr_text + "\n" + ex['passage']
        ptoks = count_prefix_tokens(surr_text if key != 'oracle' else ex['query'],
                                    ex['passage'])
        total = len(tokenizer(full_text, add_special_tokens=True).input_ids)
        print(f"  {cond:<28s}: {total} total, {ptoks} prefix | "
              f"prefix='{str(surr_text)[:40]}...'")""")

# ============================================================
# Cell 7: Scoring loop
# ============================================================
code(r"""# Cell 7: Scoring loop (8 conditions x 500 samples)
print("=" * 70)
print("RUNNING T5-XL SCORING")
print("=" * 70)

def build_condition(cond_name, sample):
    # Returns (encoder_text, prefix_token_count, truncate)
    passage = sample['passage']

    if cond_name == 'bare':
        return passage, 0, False
    elif cond_name == 'oracle_trunc':
        surr = sample['query']
    elif cond_name == 'random_trunc':
        surr = sample['surr_random']
    elif cond_name == 'scrambled_oracle_trunc':
        surr = sample['scrambled_oracle']
    elif cond_name == 'surr_template_trunc':
        surr = sample['surr_template']
    elif cond_name == 'static_fact_trunc':
        surr = STATIC_FACT
    elif cond_name == 'repeat_the_trunc':
        surr = sample['repeat_the']
    elif cond_name == 'single_word_trunc':
        surr = sample['single_word']
    else:
        raise ValueError(f"Unknown condition: {cond_name}")

    enc_text = surr + "\n" + passage
    ptoks = count_prefix_tokens(surr, passage)
    return enc_text, ptoks, True


# Resume from checkpoint
all_results = []
start_idx = 0

if CHECKPOINT_PATH.exists():
    ckpt = json.loads(CHECKPOINT_PATH.read_text())
    if ckpt.get('n_total') == N_SAMPLES and len(ckpt.get('results', [])) > 0:
        saved_queries = [r['query'][:50] for r in ckpt['results']]
        current_queries = [s['query'][:50] for s in samples[:len(saved_queries)]]
        if saved_queries == current_queries:
            all_results = ckpt['results']
            start_idx = len(all_results)
            print(f"Resuming from checkpoint: {start_idx}/{N_SAMPLES}")

if start_idx == 0:
    total_calls = len(COND_NAMES) * N_SAMPLES
    print(f"Starting fresh: {len(COND_NAMES)} conditions x {N_SAMPLES} samples "
          f"= {total_calls} scorings")
    print(f"Estimated runtime: ~{total_calls * 0.1 / 60:.0f} min")

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
              desc="Scoring"):
    s = samples[i]
    result = {
        'query': s['query'],
        'answer': s['answer'],
        'passage_words': s['word_count'],
    }

    for cond_name in COND_NAMES:
        enc_text, ptoks, trunc = build_condition(cond_name, s)
        nll = score_nll(enc_text, s['answer'], ptoks, trunc)
        result[f'nll_{cond_name}'] = nll

    all_results.append(result)

    if (i + 1) % 20 == 0 or i == N_SAMPLES - 1:
        ckpt = {
            'n_total': N_SAMPLES,
            'results': all_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        CHECKPOINT_PATH.write_text(json.dumps(ckpt))
        elapsed = time.time() - t0
        done = i - start_idx + 1
        eta = (N_SAMPLES - i - 1) * elapsed / done if done > 0 else 0
        tqdm.write(f"  Checkpoint {i+1}/{N_SAMPLES} | {elapsed/60:.1f}m | ETA {eta/60:.1f}m")

    gc.collect()
    torch.cuda.empty_cache()

elapsed = time.time() - t0
print(f"\nScoring complete: {len(all_results)} samples in {elapsed/60:.1f} min")""")

# ============================================================
# Cell 8: Main results table
# ============================================================
code(r"""# Cell 8: Main results table
from lib.analysis import cohens_d

print("=" * 70)
print(f"RESULTS: T5-XL All Conditions vs Bare (N={len(all_results)})")
print("=" * 70)

bare_nlls = np.array([r['nll_bare'] for r in all_results])

print(f"\n{'Condition':<28} {'Mean NLL':>10} {'vs Bare':>10} {'d':>8} {'Win%':>8} "
      f"{'p':>12} {'sig':>5}")
print("-" * 86)

analysis = {}

for cond in COND_NAMES:
    nlls = np.array([r[f'nll_{cond}'] for r in all_results])
    mean_nll = nlls.mean()

    if cond == 'bare':
        print(f"{cond:<28} {mean_nll:>10.4f} {'--':>10} {'--':>8} {'--':>8} "
              f"{'--':>12} {'--':>5}")
        analysis[cond] = {'mean_nll': float(mean_nll)}
    else:
        diff = bare_nlls - nlls
        d = cohens_d(diff)
        win_pct = 100 * np.mean(diff > 0)
        t_stat, p_val = stats.ttest_1samp(diff, 0)
        sig = ('***' if p_val < 0.001/N_BONFERRONI else
               '**' if p_val < 0.01/N_BONFERRONI else
               '*' if p_val < 0.05/N_BONFERRONI else 'ns')
        print(f"{cond:<28} {mean_nll:>10.4f} {diff.mean():>+10.4f} {d:>+8.3f} "
              f"{win_pct:>7.1f}% {p_val:>12.2e} {sig:>5}")
        analysis[cond] = {
            'mean_nll': float(mean_nll), 'delta': float(diff.mean()),
            'd': float(d), 'win_pct': float(win_pct), 'p': float(p_val),
        }

# Sanity check: bare NLL should be reasonable (not near 0 = ceiling)
print(f"\nSanity check: bare NLL = {bare_nlls.mean():.3f} "
      f"(std={bare_nlls.std():.3f}, range=[{bare_nlls.min():.3f}, {bare_nlls.max():.3f}])")
if bare_nlls.mean() < 0.5:
    print("  WARNING: Bare NLL very low -- possible ceiling effect")
elif bare_nlls.mean() > 10:
    print("  WARNING: Bare NLL very high -- model may not understand task well")
else:
    print("  OK: reasonable range for NLL scoring")""")

# ============================================================
# Cell 9: 3-way decomposition
# ============================================================
code(r"""# Cell 9: 3-way decomposition (structure / vocabulary / semantics)
# Compare to T5Gemma's 85% / 6% / 10% from Exp 2B

print("=" * 70)
print("3-WAY DECOMPOSITION: Structure / Vocabulary / Semantics")
print("=" * 70)
print("Same decomposition as Exp 2B:")
print("  bare -> random_trunc -> scrambled_oracle -> oracle")
print("  Structure:  bare -> random (any prefix helps)")
print("  Vocabulary: random -> scrambled (right words, wrong order)")
print("  Semantics:  scrambled -> oracle (right word order)\n")

oracle_nlls = np.array([r['nll_oracle_trunc'] for r in all_results])
random_nlls = np.array([r['nll_random_trunc'] for r in all_results])
scrambled_nlls = np.array([r['nll_scrambled_oracle_trunc'] for r in all_results])

# Component deltas (positive = improvement)
struct_comp = bare_nlls - random_nlls
vocab_comp = random_nlls - scrambled_nlls
sem_comp = scrambled_nlls - oracle_nlls
total_comp = bare_nlls - oracle_nlls

total_mean = total_comp.mean()
struct_pct = struct_comp.mean() / total_mean * 100 if total_mean != 0 else 0
vocab_pct = vocab_comp.mean() / total_mean * 100 if total_mean != 0 else 0
sem_pct = sem_comp.mean() / total_mean * 100 if total_mean != 0 else 0

print(f"{'Component':<30} {'Mean NLL':>10} {'Delta':>8} {'% total':>9} "
      f"{'d':>8} {'p':>12} {'sig':>5}")
print("-" * 85)

steps = [
    ('bare (baseline)', bare_nlls, None),
    ('+ Structure (random)', random_nlls, struct_comp),
    ('+ Vocabulary (scrambled)', scrambled_nlls, vocab_comp),
    ('+ Semantics (oracle)', oracle_nlls, sem_comp),
]

for name, nlls, component in steps:
    if component is None:
        print(f"  {name:<28} {nlls.mean():>10.4f}")
    else:
        mu = component.mean()
        pct = mu / total_mean * 100 if total_mean != 0 else 0
        d = cohens_d(component)
        _, p = stats.ttest_1samp(component, 0)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {name:<28} {nlls.mean():>10.4f} {mu:>+8.4f} {pct:>8.1f}% "
              f"{d:>+8.3f} {p:>12.2e} {sig}")

print(f"  {'TOTAL':<28} {'':>10} {total_mean:>+8.4f} {'100.0%':>9}")

# Verify decomposition sums correctly
residual = total_mean - (struct_comp.mean() + vocab_comp.mean() + sem_comp.mean())
print(f"\n  Decomposition residual: {residual:.6f} (should be ~0)")

# T5Gemma comparison
print(f"\n--- T5Gemma (Exp 2B) comparison ---")
print(f"  {'':>20} {'T5Gemma':>12} {'T5-XL':>12}")
print(f"  {'Structure':>20} {'84.7%':>12} {struct_pct:>11.1f}%")
print(f"  {'Vocabulary':>20} {'5.5%':>12} {vocab_pct:>11.1f}%")
print(f"  {'Semantics':>20} {'9.7%':>12} {sem_pct:>11.1f}%")

decomp_analysis = {
    'structure_pct': float(struct_pct),
    'vocabulary_pct': float(vocab_pct),
    'semantics_pct': float(sem_pct),
    'structure_d': float(cohens_d(struct_comp)),
    'vocabulary_d': float(cohens_d(vocab_comp)),
    'semantics_d': float(cohens_d(sem_comp)),
}""")

# ============================================================
# Cell 10: Prefix length analysis
# ============================================================
code(r"""# Cell 10: Prefix length analysis
# single_word (1 token) vs repeat_the (10 tokens) vs random (~20 words)
# Does the switch mechanism (1 word = 85% of benefit) hold in T5-XL?

print("=" * 70)
print("PREFIX LENGTH ANALYSIS")
print("=" * 70)
print("Does 1 token suffice (switch mechanism)?\n")

length_conds = ['single_word_trunc', 'repeat_the_trunc', 'random_trunc']
oracle_d_val = cohens_d(bare_nlls - oracle_nlls)

print(f"{'Condition':<28} {'~Prefix toks':>12} {'d':>8} {'% Oracle':>10}")
print("-" * 65)

length_results = {}
for cond in length_conds:
    nlls = np.array([r[f'nll_{cond}'] for r in all_results])
    benefit = bare_nlls - nlls
    d = cohens_d(benefit)
    pct = d / oracle_d_val * 100 if oracle_d_val > 0 else 0

    # Estimate prefix tokens
    ptoks_sample = [count_prefix_tokens(
        samples[j].get(cond.replace('_trunc', ''), 'X'), samples[j]['passage'])
        for j in range(50)]
    mean_ptoks = np.mean(ptoks_sample)

    print(f"  {cond:<26} {mean_ptoks:>12.1f} {d:>+8.3f} {pct:>9.1f}%")
    length_results[cond] = {'d': float(d), 'mean_ptoks': float(mean_ptoks),
                             'pct_oracle': float(pct)}

# Switch mechanism check
single_d = length_results['single_word_trunc']['d']
random_d = length_results['random_trunc']['d']
if random_d > 0 and single_d / random_d > 0.7:
    print(f"\n  --> SWITCH MECHANISM: single word captures {single_d/random_d*100:.0f}% "
          f"of random benefit")
    print(f"      Same as T5Gemma (1 random word gets 85%)")
elif random_d > 0 and single_d / random_d > 0.4:
    print(f"\n  --> GRADUAL: single word captures {single_d/random_d*100:.0f}%, "
          f"benefit scales with length")
else:
    print(f"\n  --> LENGTH-DEPENDENT: single word insufficient "
          f"({single_d/random_d*100:.0f}% of random)")

# Pairwise: single_word vs repeat_the
sw_nlls = np.array([r['nll_single_word_trunc'] for r in all_results])
rt_nlls = np.array([r['nll_repeat_the_trunc'] for r in all_results])
diff_sr = sw_nlls - rt_nlls  # positive = repeat_the better
d_sr = cohens_d(diff_sr)
_, p_sr = stats.ttest_1samp(diff_sr, 0)
sig_sr = '***' if p_sr < 0.001 else '**' if p_sr < 0.01 else '*' if p_sr < 0.05 else 'ns'
print(f"\n  single_word vs repeat_the: d={d_sr:+.3f}, p={p_sr:.2e} {sig_sr}")
if d_sr > 0.05:
    print(f"    10 tokens > 1 token: more tokens help")
else:
    print(f"    1 token ~= 10 tokens: switch mechanism confirmed")""")

# ============================================================
# Cell 11: Attention sink probe
# ============================================================
code(r"""# Cell 11: Attention sink probe (50 samples)
# Extract encoder attention weights to check for attention sink pattern

print("=" * 70)
print("ATTENTION SINK PROBE (50 samples)")
print("=" * 70)
print("Does T5-XL have a BOS/position-0 attention sink like T5Gemma?\n")

N_PROBE = 50
probe_results = []

# Probe layers: first, middle, last
probe_layers = [0, n_enc_layers // 2, n_enc_layers - 1]
print(f"Probing layers: {probe_layers}")

for sample_idx in tqdm(range(N_PROBE), desc="Attention probe"):
    s = samples[sample_idx]
    result = {'layers': {}}

    for cond in ['bare', 'oracle_trunc', 'random_trunc']:
        if cond == 'bare':
            enc_text = s['passage']
            ptoks = 0
        elif cond == 'oracle_trunc':
            enc_text = s['query'] + "\n" + s['passage']
            ptoks = count_prefix_tokens(s['query'], s['passage'])
        else:
            enc_text = s['surr_random'] + "\n" + s['passage']
            ptoks = count_prefix_tokens(s['surr_random'], s['passage'])

        enc_ids = tokenizer(enc_text, return_tensors="pt",
                            add_special_tokens=True, truncation=True,
                            max_length=1024).input_ids.to(DEVICE)
        enc_mask = torch.ones(1, enc_ids.shape[1], device=DEVICE, dtype=torch.long)

        with torch.no_grad():
            enc_out = model.encoder(
                input_ids=enc_ids, attention_mask=enc_mask,
                output_attentions=True,
            )

        # enc_out.attentions: tuple of (batch, heads, seq, seq) per layer
        for l_idx, l in enumerate(probe_layers):
            attn = enc_out.attentions[l][0].float()  # (heads, seq, seq)
            seq_len = attn.shape[1]

            # Mean attention received by each position (averaged over heads and source)
            received = attn.mean(dim=0).sum(dim=0) / seq_len  # (seq,)

            key = f'{cond}_layer{l}'
            result[key] = {
                'pos0_received': received[0].item(),
                'mean_received': received.mean().item(),
                'pos0_ratio': (received[0] / received.mean()).item() if received.mean() > 0 else 0,
                'seq_len': seq_len,
                'prefix_tokens': ptoks,
            }

            if cond != 'bare' and ptoks > 0:
                # How much attention do prefix tokens receive?
                prefix_received = received[:ptoks].mean().item()
                doc_received = received[ptoks:].mean().item()
                result[key]['prefix_received'] = prefix_received
                result[key]['doc_received'] = doc_received

        del enc_out
    gc.collect()
    torch.cuda.empty_cache()
    probe_results.append(result)

# Summarize
print(f"\n--- Position-0 attention sink (bare condition) ---")
for l in probe_layers:
    key = f'bare_layer{l}'
    ratios = [r[key]['pos0_ratio'] for r in probe_results if key in r]
    mean_ratio = np.mean(ratios)
    print(f"  Layer {l}: pos0 receives {mean_ratio:.1f}x average attention")

print(f"\n--- Prefix attention (oracle_trunc condition) ---")
for l in probe_layers:
    key = f'oracle_trunc_layer{l}'
    prefix_recv = [r[key]['prefix_received'] for r in probe_results
                   if key in r and 'prefix_received' in r[key]]
    doc_recv = [r[key]['doc_received'] for r in probe_results
                if key in r and 'doc_received' in r[key]]
    if prefix_recv and doc_recv:
        ratio = np.mean(prefix_recv) / np.mean(doc_recv) if np.mean(doc_recv) > 0 else 0
        print(f"  Layer {l}: prefix receives {ratio:.1f}x doc attention")

# T5Gemma comparison
print(f"\n--- T5Gemma comparison ---")
print(f"  T5Gemma Exp 3E: <bos> absorbs 67-87x average attention at layer 23")
print(f"  T5-XL: see results above")

probe_summary = {}
for l in probe_layers:
    key = f'bare_layer{l}'
    ratios = [r[key]['pos0_ratio'] for r in probe_results if key in r]
    probe_summary[f'layer{l}_pos0_ratio'] = float(np.mean(ratios))""")

# ============================================================
# Cell 12: Cross-model comparison table
# ============================================================
code(r"""# Cell 12: Cross-model comparison table
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("CROSS-MODEL COMPARISON: T5-XL vs T5Gemma")
print("=" * 70)

# T5Gemma reference values from Exp 02/2B (on same samples)
# These are the actual measured values from those experiments
t5gemma_ref = {
    'oracle_trunc': {'d': 0.374},         # Exp 02
    'random_trunc': {'d': 0.302},          # Exp 02
    'scrambled_oracle_trunc': {'d': 0.352},  # Exp 2B
    'surr_template_trunc': {'d': 0.336},   # Exp 02
    'static_fact_trunc': {'d': 0.372},     # Exp 02
    'repeat_the_trunc': {'d': 0.337},      # Exp 2B
    'single_word_trunc': {'d': 0.254},     # Exp 2B
    'structure_pct': 84.7,
    'vocabulary_pct': 5.5,
    'semantics_pct': 9.7,
}

print(f"\n{'Condition':<28} {'T5Gemma d':>12} {'T5-XL d':>10} {'Ratio':>8}")
print("-" * 65)

shared_conds = [c for c in COND_NAMES if c != 'bare' and c in t5gemma_ref]
for cond in shared_conds:
    gemma_d = t5gemma_ref[cond]['d']
    xl_d = analysis.get(cond, {}).get('d', 0)
    ratio = xl_d / gemma_d if gemma_d != 0 else 0
    print(f"  {cond:<26} {gemma_d:>+12.3f} {xl_d:>+10.3f} {ratio:>7.2f}x")

print(f"\n--- Decomposition comparison ---")
print(f"  {'':>20} {'T5Gemma':>12} {'T5-XL':>12}")
print(f"  {'Structure':>20} {t5gemma_ref['structure_pct']:>11.1f}% {struct_pct:>11.1f}%")
print(f"  {'Vocabulary':>20} {t5gemma_ref['vocabulary_pct']:>11.1f}% {vocab_pct:>11.1f}%")
print(f"  {'Semantics':>20} {t5gemma_ref['semantics_pct']:>11.1f}% {sem_pct:>11.1f}%")

# Plot: side-by-side bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: effect sizes
ax = axes[0]
conds_plot = shared_conds
x = np.arange(len(conds_plot))
w = 0.35
gemma_ds = [t5gemma_ref[c]['d'] for c in conds_plot]
xl_ds = [analysis.get(c, {}).get('d', 0) for c in conds_plot]
ax.bar(x - w/2, gemma_ds, w, label='T5Gemma', color='steelblue')
ax.bar(x + w/2, xl_ds, w, label='T5-XL', color='coral')
ax.set_ylabel("Cohen's d vs bare")
ax.set_title("Effect Size Comparison")
short_labels = [c.replace('_trunc', '').replace('surr_', '') for c in conds_plot]
ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Right: decomposition
ax = axes[1]
categories = ['Structure', 'Vocabulary', 'Semantics']
gemma_pcts = [t5gemma_ref['structure_pct'], t5gemma_ref['vocabulary_pct'],
              t5gemma_ref['semantics_pct']]
xl_pcts = [struct_pct, vocab_pct, sem_pct]
x = np.arange(len(categories))
ax.bar(x - w/2, gemma_pcts, w, label='T5Gemma', color='steelblue')
ax.bar(x + w/2, xl_pcts, w, label='T5-XL', color='coral')
ax.set_ylabel("% of total benefit")
ax.set_title("Mechanism Decomposition")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_path = RESULTS_DIR / 'cross_model_comparison.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved to {plot_path}")""")

# ============================================================
# Cell 13: Hardness stratification
# ============================================================
code(r"""# Cell 13: Hardness stratification
print("=" * 70)
print("HARDNESS STRATIFICATION")
print("=" * 70)

quintile_bounds = np.percentile(bare_nlls, [20, 40, 60, 80])
quintiles = np.digitize(bare_nlls, quintile_bounds)
quintile_labels = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard']

print(f"\n{'Quintile':<12} {'N':>4} {'bare NLL':>10}", end='')
for cond in ['oracle_trunc', 'random_trunc', 'surr_template_trunc']:
    short = cond.replace('_trunc', '')
    print(f" {short:>12}", end='')
print()
print("-" * 70)

hardness_analysis = {}
for q in range(5):
    mask = quintiles == q
    n_q = mask.sum()
    bare_q = bare_nlls[mask].mean()
    row = f"{quintile_labels[q]:<12} {n_q:>4} {bare_q:>10.3f}"
    hardness_analysis[quintile_labels[q]] = {'n': int(n_q), 'bare_nll': float(bare_q)}

    for cond in ['oracle_trunc', 'random_trunc', 'surr_template_trunc']:
        nlls = np.array([r[f'nll_{cond}'] for r in all_results])
        benefit = bare_nlls[mask] - nlls[mask]
        d = cohens_d(benefit)
        row += f" {d:>+12.3f}"
        hardness_analysis[quintile_labels[q]][cond] = float(d)

    print(row)

# Correlation: hardness vs benefit
print(f"\n--- Hardness correlations ---")
for cond in ['oracle_trunc', 'random_trunc']:
    nlls = np.array([r[f'nll_{cond}'] for r in all_results])
    benefit = bare_nlls - nlls
    r, p = stats.pearsonr(bare_nlls, benefit)
    print(f"  {cond}: r={r:+.3f} (p={p:.2e})")""")

# ============================================================
# Cell 14: Verdict + save + cleanup
# ============================================================
code(r"""# Cell 14: Verdict + save + cleanup
print("=" * 70)
print("VERDICT -- Exp 09: Model Generality (T5-XL)")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"N: {len(all_results)} samples")

# Key results
oracle_d = analysis.get('oracle_trunc', {}).get('d', 0)
oracle_p = analysis.get('oracle_trunc', {}).get('p', 1)
random_d = analysis.get('random_trunc', {}).get('d', 0)
random_p = analysis.get('random_trunc', {}).get('p', 1)

print(f"\n--- Key effect sizes ---")
print(f"  oracle_trunc: d={oracle_d:+.3f} (p={oracle_p:.2e})")
print(f"  random_trunc: d={random_d:+.3f} (p={random_p:.2e})")

# Q1: Does the mechanism exist in T5-XL?
print(f"\nQ1: Does the co-encoding mechanism exist in T5-XL?")
if oracle_p < 0.001 and oracle_d > 0.1:
    print(f"  >>> YES: oracle benefit d={oracle_d:+.3f} (***)")
    mechanism_exists = True
elif oracle_p < 0.05 and oracle_d > 0:
    print(f"  >>> MARGINAL: oracle benefit d={oracle_d:+.3f} (*)")
    mechanism_exists = True
else:
    print(f"  >>> NO: oracle benefit d={oracle_d:+.3f} (ns)")
    mechanism_exists = False

# Q2: Is it primarily structural?
print(f"\nQ2: Is the mechanism structural (~85% in T5Gemma)?")
print(f"  Structure: {struct_pct:.1f}%")
print(f"  Vocabulary: {vocab_pct:.1f}%")
print(f"  Semantics: {sem_pct:.1f}%")
if struct_pct > 70:
    print(f"  >>> YES: primarily structural ({struct_pct:.0f}%)")
    structural = True
elif struct_pct > 50:
    print(f"  >>> MIXED: structural ({struct_pct:.0f}%) but content matters more")
    structural = False
else:
    print(f"  >>> NO: primarily semantic ({100-struct_pct:.0f}%)")
    structural = False

# Q3: Switch mechanism?
sw_d = analysis.get('single_word_trunc', {}).get('d', 0)
print(f"\nQ3: Does the switch mechanism (1 word = most benefit) hold?")
if random_d > 0 and sw_d / random_d > 0.7:
    print(f"  >>> YES: single word captures {sw_d/random_d*100:.0f}% of random benefit")
    switch = True
else:
    print(f"  >>> NO: single word captures only {sw_d/random_d*100:.0f}%"
          if random_d > 0 else "  >>> NO: no benefit detected")
    switch = False

# Q4: Scale comparison
print(f"\nQ4: Absolute scale comparison")
gemma_oracle_d = 0.374  # Exp 02 reference
ratio = oracle_d / gemma_oracle_d if gemma_oracle_d > 0 else 0
print(f"  T5Gemma oracle d: +0.374")
print(f"  T5-XL oracle d:   {oracle_d:+.3f} ({ratio:.0f}% of T5Gemma)")

# Overall verdict
print(f"\n--- OVERALL VERDICT ---")
if mechanism_exists and structural:
    print(f"  The attention redistribution mechanism is a GENERAL property of")
    print(f"  encoder-decoder architectures, not specific to T5Gemma.")
    if switch:
        print(f"  The switch mechanism (1 word suffices) also generalizes.")
    print(f"  This suggests the benefit comes from the fundamental structure of")
    print(f"  bidirectional self-attention, not from RoPE, sliding windows,")
    print(f"  or merged attention -- all of which T5-XL lacks.")
elif mechanism_exists:
    print(f"  The co-encoding mechanism exists in T5-XL but with different")
    print(f"  characteristics. The decomposition differs from T5Gemma,")
    print(f"  suggesting model-specific factors influence the balance.")
else:
    print(f"  The mechanism does NOT transfer to T5-XL. This suggests it is")
    print(f"  specific to T5Gemma's architecture (possibly RoPE, sliding windows,")
    print(f"  or merged attention). Not a general encoder-decoder property.")

print(f"\n{'='*70}")

# Save results
final_results = {
    'experiment': 'exp09_model_generality',
    'model': MODEL_NAME,
    'n_samples': len(all_results),
    'n_bonferroni': N_BONFERRONI,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'analysis': analysis,
    'decomposition': decomp_analysis,
    'length_analysis': length_results,
    'hardness_analysis': hardness_analysis,
    'attention_probe': probe_summary,
    'verdict': {
        'mechanism_exists': mechanism_exists,
        'primarily_structural': structural,
        'switch_mechanism': switch,
        'oracle_d_ratio_vs_t5gemma': float(ratio),
    },
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

# Cleanup
print("Cleaning up GPU memory...")
mem_before = torch.cuda.memory_allocated() / 1e9
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()
gc.collect()
mem_after = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory: {mem_before:.2f} GB -> {mem_after:.2f} GB")
print("Done!")""")

# ============================================================
# Write notebook
# ============================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4, "nbformat_minor": 5
}

outpath = "experiments/09/09_model_generality.ipynb"
with open(outpath, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"Notebook written to {outpath} ({len(cells)} cells)")
