#!/usr/bin/env python3
# Generate Exp 10 notebook: Selective Truncation.
#
# Exp 01: full visibility (d=+0.345) < truncation (d=+0.408). The decoder reading
# ALL prefix tokens hurts. But what about reading JUST the keyword? The encoder gets
# full structural + semantic enrichment via bidirectional attention, and the decoder
# gets targeted value access to the most informative token.
#
# 10 conditions testing whether selective keyword visibility helps.
import json

cells = []


def md(source):
    lines = source.split("\n") if isinstance(source, str) else source
    formatted = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type": "markdown", "metadata": {}, "source": formatted})
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
md(r"""# Experiment 10: Selective Truncation
## Does letting the decoder see keyword tokens help beyond standard truncation?

### Motivation
Exp 01 established that truncation (masking prefix from decoder cross-attention) is
strictly better than full visibility:
- `oracle_full` (decoder sees query + doc): d=+0.345
- `oracle_trunc` (decoder sees doc only): d=+0.408

The decoder reading ALL prefix tokens creates noise/interference. But what about
reading JUST the keyword? The encoder gets full structural + semantic enrichment
via bidirectional attention, and the decoder gets targeted value access to the
single most informative token.

### Key idea: selective cross-attention masking
```
Encoder input:   [prefix tokens] [document tokens]
Standard trunc:  [masked........] [visible........]
Selective:       [masked.][KW][.] [visible........]
                          ^^^^
                    keyword unmasked
```

### Conditions (10)
| # | Condition | Encoder prefix | Decoder sees | Purpose |
|---|-----------|---------------|-------------|---------|
| 1 | bare | (none) | doc only | lower bound |
| 2 | oracle\_trunc | query | doc only | upper bound (Exp 01) |
| 3 | oracle\_full | query | query + doc | Exp 01 comparison |
| 4 | oracle\_kw\_visible | query | top query keyword + doc | **KEY TEST** |
| 5 | template\_trunc | "What is [kw]?" | doc only | Exp 02 best |
| 6 | template\_kw\_visible | "What is [kw]?" | [kw] + doc | **KEY TEST** |
| 7 | pad\_kw\_trunc | "the the the [kw]" | doc only | structural + vocab |
| 8 | pad\_kw\_kw\_visible | "the the the [kw]" | [kw] + doc | **KEY TEST** |
| 9 | random\_trunc | random text | doc only | structural control |
| 10 | keyword\_only\_visible | [kw] | [kw] + doc | minimal prefix + direct access |

### Key comparisons
- `oracle_trunc` vs `oracle_kw_visible`: does seeing 1 keyword help beyond truncation?
- `template_trunc` vs `template_kw_visible`: same for doc-derived heuristic
- `pad_kw_trunc` vs `pad_kw_kw_visible`: structural + vocab enrichment + direct access?
- `oracle_trunc` vs `oracle_full`: replicates Exp 01 (trunc > full)

### N=500 (same samples as Exp 02)
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
from scipy.stats import wilcoxon
from tqdm.auto import tqdm

sys.path.insert(0, "../..")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("../../results/exp10")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

N_SAMPLES = 500
MODEL_NAME = "google/t5gemma-2-4b-4b"
N_BONFERRONI = 9  # 9 non-bare conditions

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"Exp 10: Selective Truncation")
print(f"Model: {MODEL_NAME}")
print(f"N: {N_SAMPLES}")
print(f"Bonferroni comparisons: {N_BONFERRONI}")
print(f"CUDA: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")""")

# ============================================================
# Cell 3: Load model
# ============================================================
code("""# Cell 3: Load model
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

print(f"Loading {MODEL_NAME}...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer = processor.tokenizer

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
)
model.eval()

DEVICE = next(model.parameters()).device
print(f"Model loaded. dtype={next(model.parameters()).dtype}")
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")""")

# ============================================================
# Cell 4: Scoring helpers
# ============================================================
code(r"""# Cell 4: Scoring helpers with selective visibility

def score_nll_selective(encoder_text, answer_text, prefix_token_count=0,
                        truncate=False, visible_positions=None):
    # Score NLL of answer tokens with optional truncation and selective visibility.
    #
    # Args:
    #   encoder_text: Full text for encoder (e.g., "[prefix]\n[document]")
    #   answer_text: Answer text for decoder
    #   prefix_token_count: Number of prefix tokens to potentially mask
    #   truncate: If True, mask all prefix tokens from decoder cross-attention
    #   visible_positions: List of prefix positions to UNMASK (override truncation).
    #       Only used when truncate=True. These positions within the prefix
    #       remain visible to the decoder.
    enc_ids = tokenizer(encoder_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids.to(DEVICE)
    total_enc_len = enc_ids.shape[1]

    # Full mask for encoder (bidirectional, sees everything)
    enc_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        encoder_outputs = model.get_encoder()(
            input_ids=enc_ids, attention_mask=enc_mask
        )

    # Build cross-attention mask for decoder
    if truncate and prefix_token_count > 0:
        cross_attn_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)
        cross_attn_mask[:, :prefix_token_count] = 0
        # Selectively unmask keyword positions
        if visible_positions:
            for pos in visible_positions:
                if 0 <= pos < prefix_token_count:
                    cross_attn_mask[:, pos] = 1
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
    # Count how many tokens the prefix occupies in [prefix + newline + document].
    full_text = prefix_text + "\n" + document_text
    full_ids = tokenizer(full_text, add_special_tokens=True).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True).input_ids
    return len(full_ids) - len(doc_ids)


def find_keyword_positions(full_text, keyword, document_text):
    # Find the token positions of `keyword` within the prefix portion of full_text.
    #
    # Returns: (n_prefix, kw_positions)
    #   n_prefix: number of prefix tokens
    #   kw_positions: list of token indices within the prefix that correspond to keyword
    full_ids = tokenizer(full_text, add_special_tokens=True).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True).input_ids
    n_prefix = len(full_ids) - len(doc_ids)

    kw_ids = tokenizer(keyword, add_special_tokens=False).input_ids
    prefix_ids = full_ids[:n_prefix]

    positions = []
    if len(kw_ids) == 0:
        return n_prefix, positions

    # Search for keyword token sequence within prefix
    for i in range(len(prefix_ids) - len(kw_ids) + 1):
        if prefix_ids[i:i + len(kw_ids)] == kw_ids:
            positions = list(range(i, i + len(kw_ids)))
            break

    # Fallback: if exact match fails, try matching just the first kw token
    if not positions and len(kw_ids) > 0:
        for i in range(len(prefix_ids)):
            if prefix_ids[i] == kw_ids[0]:
                positions = [i]
                break

    return n_prefix, positions


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

def get_top_keyword(text):
    # Get the single most frequent content word from text.
    content_words = extract_keywords(text)
    if not content_words:
        return "information"
    counts = Counter(content_words)
    return counts.most_common(1)[0][0]

def make_surrogate_template(passage):
    kw = get_top_keyword(passage)
    return f"What is {kw}?", kw

print("Helpers defined.")
print("  score_nll_selective: supports visible_positions for keyword unmasking")
print("  find_keyword_positions: locates keyword tokens within prefix")""")

# ============================================================
# Cell 5: Load data + precompute keywords
# ============================================================
code(r"""# Cell 5: Load MS MARCO data (same 500 as Exp 02) + precompute keywords
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

# Precompute keywords and surrogates for each sample
for i, s in enumerate(samples):
    # Top query keyword
    query_kws = extract_keywords(s['query'])
    s['query_kw'] = query_kws[0] if query_kws else "information"

    # Document keyword
    s['doc_kw'] = get_top_keyword(s['passage'])

    # Template surrogate (uses doc keyword)
    s['surr_template'], s['template_kw'] = make_surrogate_template(s['passage'])

    # Pad + keyword: "the the the [kw]"
    s['pad_kw'] = "the the the " + s['doc_kw']

    # Keyword only
    s['keyword_only'] = s['doc_kw']

    # Random surrogate (from another sample)
    other_idx = (i + N_SAMPLES // 2) % len(samples)
    other_passage = samples[other_idx]['passage']
    s['surr_random'] = " ".join(other_passage.split()[:20])

print(f"Selected {len(samples)} samples")
print(f"Mean words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Sample keywords: {[s['doc_kw'] for s in samples[:5]]}")""")

# ============================================================
# Cell 6: Condition examples + keyword position verification
# ============================================================
code(r"""# Cell 6: Condition examples + keyword position verification
print("=" * 70)
print("CONDITION EXAMPLES + KEYWORD POSITION VERIFICATION")
print("=" * 70)

COND_NAMES = [
    'bare', 'oracle_trunc', 'oracle_full', 'oracle_kw_visible',
    'template_trunc', 'template_kw_visible',
    'pad_kw_trunc', 'pad_kw_kw_visible',
    'random_trunc', 'keyword_only_visible',
]

ex = samples[0]
print(f"\nQuery: {ex['query'][:80]}")
print(f"Answer: {ex['answer'][:80]}")
print(f"Passage: {ex['passage'][:80]}...")
print(f"Query keyword: '{ex['query_kw']}'")
print(f"Doc keyword: '{ex['doc_kw']}'")

# Show each condition
for cond in COND_NAMES:
    if cond == 'bare':
        enc = ex['passage'][:60] + "..."
        dec = "doc only"
    elif cond == 'oracle_trunc':
        enc = ex['query'] + " | " + ex['passage'][:40] + "..."
        dec = "doc only (prefix masked)"
    elif cond == 'oracle_full':
        enc = ex['query'] + " | " + ex['passage'][:40] + "..."
        dec = "query + doc (all visible)"
    elif cond == 'oracle_kw_visible':
        enc = ex['query'] + " | " + ex['passage'][:40] + "..."
        dec = f"'{ex['query_kw']}' + doc (keyword visible)"
    elif cond == 'template_trunc':
        enc = ex['surr_template'] + " | " + ex['passage'][:40] + "..."
        dec = "doc only"
    elif cond == 'template_kw_visible':
        enc = ex['surr_template'] + " | " + ex['passage'][:40] + "..."
        dec = f"'{ex['template_kw']}' + doc"
    elif cond == 'pad_kw_trunc':
        enc = ex['pad_kw'] + " | " + ex['passage'][:40] + "..."
        dec = "doc only"
    elif cond == 'pad_kw_kw_visible':
        enc = ex['pad_kw'] + " | " + ex['passage'][:40] + "..."
        dec = f"'{ex['doc_kw']}' + doc"
    elif cond == 'random_trunc':
        enc = ex['surr_random'][:30] + "... | " + ex['passage'][:30] + "..."
        dec = "doc only"
    elif cond == 'keyword_only_visible':
        enc = ex['doc_kw'] + " | " + ex['passage'][:40] + "..."
        dec = f"'{ex['doc_kw']}' + doc (keyword visible)"
    print(f"  {cond:<24s}: enc=[{enc[:55]}]  dec=[{dec}]")

# Keyword position verification on first 20 samples
print(f"\n--- Keyword position verification (first 20 samples) ---")
hit_counts = {'oracle_kw': 0, 'template_kw': 0, 'pad_kw': 0, 'keyword_only': 0}
total = 20

for i in range(total):
    s = samples[i]

    # oracle_kw_visible: find query keyword in "query\npassage"
    full_text = s['query'] + "\n" + s['passage']
    n_pfx, positions = find_keyword_positions(full_text, s['query_kw'], s['passage'])
    if positions:
        hit_counts['oracle_kw'] += 1
    if i < 3:
        print(f"  Sample {i} oracle_kw: kw='{s['query_kw']}', n_prefix={n_pfx}, "
              f"positions={positions}")

    # template_kw_visible: find doc keyword in "What is [kw]?\npassage"
    full_text = s['surr_template'] + "\n" + s['passage']
    n_pfx, positions = find_keyword_positions(full_text, s['template_kw'], s['passage'])
    if positions:
        hit_counts['template_kw'] += 1
    if i < 3:
        print(f"  Sample {i} template_kw: kw='{s['template_kw']}', n_prefix={n_pfx}, "
              f"positions={positions}")

    # pad_kw: find doc keyword in "the the the [kw]\npassage"
    full_text = s['pad_kw'] + "\n" + s['passage']
    n_pfx, positions = find_keyword_positions(full_text, s['doc_kw'], s['passage'])
    if positions:
        hit_counts['pad_kw'] += 1

    # keyword_only: find doc keyword in "[kw]\npassage"
    full_text = s['keyword_only'] + "\n" + s['passage']
    n_pfx, positions = find_keyword_positions(full_text, s['doc_kw'], s['passage'])
    if positions:
        hit_counts['keyword_only'] += 1

print(f"\nKeyword hit rates (first {total} samples):")
for key, count in hit_counts.items():
    print(f"  {key}: {count}/{total} ({100*count/total:.0f}%)")""")

# ============================================================
# Cell 7: Scoring loop
# ============================================================
code(r"""# Cell 7: Scoring loop (10 conditions x 500 samples)
print("=" * 70)
print("RUNNING SELECTIVE TRUNCATION EXPERIMENT")
print("=" * 70)

def build_condition(cond_name, sample):
    # Returns (encoder_text, prefix_token_count, truncate, visible_positions)
    passage = sample['passage']

    if cond_name == 'bare':
        return passage, 0, False, None

    elif cond_name == 'oracle_trunc':
        surr = sample['query']
        enc_text = surr + "\n" + passage
        ptoks = count_prefix_tokens(surr, passage)
        return enc_text, ptoks, True, None

    elif cond_name == 'oracle_full':
        surr = sample['query']
        enc_text = surr + "\n" + passage
        return enc_text, 0, False, None

    elif cond_name == 'oracle_kw_visible':
        surr = sample['query']
        enc_text = surr + "\n" + passage
        n_pfx, kw_pos = find_keyword_positions(enc_text, sample['query_kw'], passage)
        return enc_text, n_pfx, True, kw_pos if kw_pos else None

    elif cond_name == 'template_trunc':
        surr = sample['surr_template']
        enc_text = surr + "\n" + passage
        ptoks = count_prefix_tokens(surr, passage)
        return enc_text, ptoks, True, None

    elif cond_name == 'template_kw_visible':
        surr = sample['surr_template']
        enc_text = surr + "\n" + passage
        n_pfx, kw_pos = find_keyword_positions(enc_text, sample['template_kw'], passage)
        return enc_text, n_pfx, True, kw_pos if kw_pos else None

    elif cond_name == 'pad_kw_trunc':
        surr = sample['pad_kw']
        enc_text = surr + "\n" + passage
        ptoks = count_prefix_tokens(surr, passage)
        return enc_text, ptoks, True, None

    elif cond_name == 'pad_kw_kw_visible':
        surr = sample['pad_kw']
        enc_text = surr + "\n" + passage
        n_pfx, kw_pos = find_keyword_positions(enc_text, sample['doc_kw'], passage)
        return enc_text, n_pfx, True, kw_pos if kw_pos else None

    elif cond_name == 'random_trunc':
        surr = sample['surr_random']
        enc_text = surr + "\n" + passage
        ptoks = count_prefix_tokens(surr, passage)
        return enc_text, ptoks, True, None

    elif cond_name == 'keyword_only_visible':
        surr = sample['keyword_only']
        enc_text = surr + "\n" + passage
        n_pfx, kw_pos = find_keyword_positions(enc_text, sample['doc_kw'], passage)
        return enc_text, n_pfx, True, kw_pos if kw_pos else None

    else:
        raise ValueError(f"Unknown condition: {cond_name}")


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
    print(f"Estimated runtime: ~{total_calls * 0.2 / 60:.0f} min")

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
              desc="Scoring"):
    s = samples[i]
    result = {
        'query': s['query'],
        'answer': s['answer'],
        'passage_words': s['word_count'],
        'query_kw': s['query_kw'],
        'doc_kw': s['doc_kw'],
    }

    for cond_name in COND_NAMES:
        enc_text, ptoks, trunc, vis_pos = build_condition(cond_name, s)
        nll = score_nll_selective(enc_text, s['answer'], ptoks, trunc, vis_pos)
        result[f'nll_{cond_name}'] = nll

        # Track keyword hit for _kw_visible conditions
        if vis_pos is not None:
            result[f'kw_hit_{cond_name}'] = True
            result[f'kw_ntoks_{cond_name}'] = len(vis_pos)
        elif cond_name.endswith('_visible'):
            result[f'kw_hit_{cond_name}'] = False
            result[f'kw_ntoks_{cond_name}'] = 0

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
print(f"RESULTS: All Conditions vs Bare (N={len(all_results)})")
print("=" * 70)

bare_nlls = np.array([r['nll_bare'] for r in all_results])

print(f"\n{'Condition':<24} {'Mean NLL':>10} {'vs Bare':>10} {'d':>8} {'Win%':>8} "
      f"{'p':>12} {'sig':>5}")
print("-" * 82)

analysis = {}

for cond in COND_NAMES:
    nlls = np.array([r[f'nll_{cond}'] for r in all_results])
    mean_nll = nlls.mean()

    if cond == 'bare':
        print(f"{cond:<24} {mean_nll:>10.4f} {'--':>10} {'--':>8} {'--':>8} "
              f"{'--':>12} {'--':>5}")
        analysis[cond] = {'mean_nll': float(mean_nll)}
    else:
        diff = bare_nlls - nlls  # positive = condition is better
        d = cohens_d(diff)
        win_pct = 100 * np.mean(diff > 0)
        t_stat, p_val = stats.ttest_1samp(diff, 0)
        sig = ('***' if p_val < 0.001/N_BONFERRONI else
               '**' if p_val < 0.01/N_BONFERRONI else
               '*' if p_val < 0.05/N_BONFERRONI else 'ns')
        print(f"{cond:<24} {mean_nll:>10.4f} {diff.mean():>+10.4f} {d:>+8.3f} "
              f"{win_pct:>7.1f}% {p_val:>12.2e} {sig:>5}")
        analysis[cond] = {
            'mean_nll': float(mean_nll), 'delta': float(diff.mean()),
            'd': float(d), 'win_pct': float(win_pct), 'p': float(p_val),
        }""")

# ============================================================
# Cell 9: Key pairwise comparisons
# ============================================================
code(r"""# Cell 9: Key pairwise comparisons
print("=" * 70)
print("KEY PAIRWISE COMPARISONS")
print("=" * 70)
print("Does selective keyword visibility improve beyond standard truncation?\n")

pairwise_tests = [
    ('oracle_trunc', 'oracle_kw_visible',
     "Oracle: does seeing query keyword help?"),
    ('template_trunc', 'template_kw_visible',
     "Template: does seeing doc keyword help?"),
    ('pad_kw_trunc', 'pad_kw_kw_visible',
     "Pad+KW: does seeing keyword help?"),
    ('oracle_trunc', 'oracle_full',
     "Replication: trunc > full? (Exp 01)"),
    ('oracle_kw_visible', 'oracle_full',
     "Selective vs full visibility?"),
    ('keyword_only_visible', 'random_trunc',
     "Keyword-only visible vs random trunc?"),
    ('keyword_only_visible', 'template_trunc',
     "Keyword-only visible vs template trunc?"),
]

pairwise_analysis = {}

for cond_a, cond_b, question in pairwise_tests:
    nlls_a = np.array([r[f'nll_{cond_a}'] for r in all_results])
    nlls_b = np.array([r[f'nll_{cond_b}'] for r in all_results])
    diff = nlls_a - nlls_b  # positive = B is better (lower NLL)
    d = cohens_d(diff)
    win_b = 100 * np.mean(diff > 0)
    t_stat, p_val = stats.ttest_1samp(diff, 0)
    sig = ('***' if p_val < 0.001 else '**' if p_val < 0.01 else
           '*' if p_val < 0.05 else 'ns')
    winner = cond_b if d > 0 else cond_a

    print(f"  {question}")
    print(f"    {cond_a} vs {cond_b}: d={d:+.3f}, {cond_b} wins {win_b:.1f}%, "
          f"p={p_val:.2e} {sig}")
    print(f"    Winner: {winner}")
    print()

    pairwise_analysis[f'{cond_a}_vs_{cond_b}'] = {
        'd': float(d), 'win_b_pct': float(win_b),
        'p': float(p_val), 'winner': winner,
    }

# Headline
print("=" * 70)
oracle_trunc_d = analysis.get('oracle_trunc', {}).get('d', 0)
oracle_kw_d = analysis.get('oracle_kw_visible', {}).get('d', 0)
oracle_full_d = analysis.get('oracle_full', {}).get('d', 0)
print(f"HEADLINE:")
print(f"  oracle_trunc:      d={oracle_trunc_d:+.3f} (Exp 01 reference)")
print(f"  oracle_kw_visible: d={oracle_kw_d:+.3f}")
print(f"  oracle_full:       d={oracle_full_d:+.3f}")
if oracle_kw_d > oracle_trunc_d:
    print(f"  >>> Selective visibility HELPS (+{oracle_kw_d - oracle_trunc_d:.3f})")
elif oracle_kw_d < oracle_trunc_d:
    print(f"  >>> Selective visibility HURTS ({oracle_kw_d - oracle_trunc_d:+.3f})")
else:
    print(f"  >>> Selective visibility has no effect")""")

# ============================================================
# Cell 10: Mechanism decomposition chain
# ============================================================
code(r"""# Cell 10: Mechanism decomposition chain
print("=" * 70)
print("MECHANISM DECOMPOSITION CHAIN")
print("=" * 70)
print("bare -> random_trunc -> pad_kw_trunc -> pad_kw_kw_visible -> oracle_kw_visible\n")

chain = ['bare', 'random_trunc', 'pad_kw_trunc', 'pad_kw_kw_visible', 'oracle_kw_visible']
chain_labels = {
    'bare': 'bare (baseline)',
    'random_trunc': '+ structure (random prefix)',
    'pad_kw_trunc': '+ vocabulary (keyword in prefix)',
    'pad_kw_kw_visible': '+ direct access (keyword visible to decoder)',
    'oracle_kw_visible': '+ semantics (real query + keyword visible)',
}

nlls_by_cond = {c: np.array([r[f'nll_{c}'] for r in all_results]) for c in chain}

print(f"{'Step':<50} {'NLL':>8} {'Incremental':>12} {'d':>8} {'p':>12} {'sig':>5}")
print("-" * 98)

prev_nlls = nlls_by_cond['bare']
for cond in chain:
    nlls = nlls_by_cond[cond]
    label = chain_labels[cond]

    if cond == 'bare':
        print(f"  {label:<48} {nlls.mean():>8.4f}")
    else:
        increment = prev_nlls - nlls  # positive = improvement
        d = cohens_d(increment)
        t_stat, p_val = stats.ttest_1samp(increment, 0)
        sig = ('***' if p_val < 0.001 else '**' if p_val < 0.01 else
               '*' if p_val < 0.05 else 'ns')
        print(f"  {label:<48} {nlls.mean():>8.4f} {increment.mean():>+12.4f} "
              f"{d:>+8.3f} {p_val:>12.2e} {sig:>5}")
    prev_nlls = nlls

# Cumulative from bare
print(f"\n--- Cumulative from bare ---")
total_benefit = nlls_by_cond['bare'] - nlls_by_cond['oracle_kw_visible']
total_d = cohens_d(total_benefit)
for cond in chain[1:]:
    cum_benefit = nlls_by_cond['bare'] - nlls_by_cond[cond]
    cum_d = cohens_d(cum_benefit)
    pct = cum_d / total_d * 100 if total_d > 0 else 0
    print(f"  bare -> {cond:<28s}: d={cum_d:+.3f} ({pct:.0f}% of total chain)")

# Also show the oracle_trunc reference
oracle_trunc_benefit = nlls_by_cond['bare'] - np.array([r['nll_oracle_trunc'] for r in all_results])
oracle_trunc_d = cohens_d(oracle_trunc_benefit)
print(f"\n  Reference: bare -> oracle_trunc: d={oracle_trunc_d:+.3f}")""")

# ============================================================
# Cell 11: Hardness stratification
# ============================================================
code(r"""# Cell 11: Hardness stratification
print("=" * 70)
print("HARDNESS STRATIFICATION")
print("=" * 70)
print("Does keyword visibility help more for hard samples?\n")

quintile_bounds = np.percentile(bare_nlls, [20, 40, 60, 80])
quintiles = np.digitize(bare_nlls, quintile_bounds)
quintile_labels = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard']

# Key comparison: oracle_trunc vs oracle_kw_visible by quintile
oracle_trunc_nlls = np.array([r['nll_oracle_trunc'] for r in all_results])
oracle_kw_nlls = np.array([r['nll_oracle_kw_visible'] for r in all_results])
template_trunc_nlls = np.array([r['nll_template_trunc'] for r in all_results])
template_kw_nlls = np.array([r['nll_template_kw_visible'] for r in all_results])

print(f"{'Quintile':<12} {'N':>4} {'bare':>8} "
      f"{'orc_trunc d':>12} {'orc_kw d':>10} {'kw gain':>10} "
      f"{'tpl_trunc d':>12} {'tpl_kw d':>10} {'kw gain':>10}")
print("-" * 100)

hardness_analysis = {}
for q in range(5):
    mask = quintiles == q
    n_q = mask.sum()
    bare_q = bare_nlls[mask].mean()

    ot_d = cohens_d(bare_nlls[mask] - oracle_trunc_nlls[mask])
    ok_d = cohens_d(bare_nlls[mask] - oracle_kw_nlls[mask])
    ok_gain = ok_d - ot_d

    tt_d = cohens_d(bare_nlls[mask] - template_trunc_nlls[mask])
    tk_d = cohens_d(bare_nlls[mask] - template_kw_nlls[mask])
    tk_gain = tk_d - tt_d

    print(f"{quintile_labels[q]:<12} {n_q:>4} {bare_q:>8.3f} "
          f"{ot_d:>+12.3f} {ok_d:>+10.3f} {ok_gain:>+10.3f} "
          f"{tt_d:>+12.3f} {tk_d:>+10.3f} {tk_gain:>+10.3f}")

    hardness_analysis[quintile_labels[q]] = {
        'oracle_trunc_d': float(ot_d), 'oracle_kw_d': float(ok_d),
        'oracle_kw_gain': float(ok_gain),
        'template_trunc_d': float(tt_d), 'template_kw_d': float(tk_d),
        'template_kw_gain': float(tk_gain),
    }

# Correlation: hardness vs keyword visibility gain
oracle_kw_gain_per_sample = (bare_nlls - oracle_kw_nlls) - (bare_nlls - oracle_trunc_nlls)
r, p = stats.pearsonr(bare_nlls, oracle_kw_gain_per_sample)
print(f"\nCorrelation: hardness vs oracle_kw_gain: r={r:+.3f} (p={p:.2e})")
if r > 0.1:
    print("  --> Keyword visibility helps MORE for harder samples")
elif r < -0.1:
    print("  --> Keyword visibility helps MORE for easier samples")
else:
    print("  --> Keyword visibility benefit is uniform across difficulty")""")

# ============================================================
# Cell 12: Keyword position diagnostics
# ============================================================
code(r"""# Cell 12: Keyword position diagnostics
print("=" * 70)
print("KEYWORD POSITION DIAGNOSTICS")
print("=" * 70)

kw_visible_conds = ['oracle_kw_visible', 'template_kw_visible',
                    'pad_kw_kw_visible', 'keyword_only_visible']

for cond in kw_visible_conds:
    hits = [r.get(f'kw_hit_{cond}', False) for r in all_results]
    ntoks = [r.get(f'kw_ntoks_{cond}', 0) for r in all_results]
    hit_rate = 100 * np.mean(hits)
    mean_toks = np.mean([n for n in ntoks if n > 0]) if any(n > 0 for n in ntoks) else 0

    print(f"\n  {cond}:")
    print(f"    Hit rate: {hit_rate:.1f}% ({sum(hits)}/{len(hits)})")
    print(f"    Mean keyword tokens (when found): {mean_toks:.1f}")

    # Performance comparison: hit vs miss
    hit_mask = np.array(hits)
    if hit_mask.sum() > 10 and (~hit_mask).sum() > 10:
        hit_nlls = np.array([r[f'nll_{cond}'] for r in all_results])[hit_mask]
        miss_nlls = np.array([r[f'nll_{cond}'] for r in all_results])[~hit_mask]
        hit_bare = bare_nlls[hit_mask]
        miss_bare = bare_nlls[~hit_mask]
        hit_d = cohens_d(hit_bare - hit_nlls)
        miss_d = cohens_d(miss_bare - miss_nlls)
        print(f"    Hit d={hit_d:+.3f} vs Miss d={miss_d:+.3f} (gap={hit_d - miss_d:+.3f})")
    elif hit_mask.sum() > 0:
        print(f"    Not enough misses for comparison ({(~hit_mask).sum()} misses)")

# Keyword token distribution
print(f"\n--- Token count for keywords ---")
for s in samples[:5]:
    qk_toks = len(tokenizer(s['query_kw'], add_special_tokens=False).input_ids)
    dk_toks = len(tokenizer(s['doc_kw'], add_special_tokens=False).input_ids)
    print(f"  query_kw='{s['query_kw']}' ({qk_toks} toks), "
          f"doc_kw='{s['doc_kw']}' ({dk_toks} toks)")""")

# ============================================================
# Cell 13: Verdict + save + cleanup
# ============================================================
code(r"""# Cell 13: Verdict + save + cleanup
print("=" * 70)
print("VERDICT -- Exp 10: Selective Truncation")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"N: {len(all_results)} samples")

# Key effect sizes
print(f"\n--- Effect sizes (d vs bare) ---")
for cond in COND_NAMES[1:]:
    a = analysis.get(cond, {})
    d = a.get('d', 0)
    p = a.get('p', 1)
    sig = ('***' if p < 0.001/N_BONFERRONI else '**' if p < 0.01/N_BONFERRONI
           else '*' if p < 0.05/N_BONFERRONI else 'ns')
    print(f"  {cond:<24s}: d={d:+.3f} {sig}")

# Key question answers
print(f"\n--- KEY QUESTIONS ---")

# Q1: Does keyword visibility help?
ot_d = analysis.get('oracle_trunc', {}).get('d', 0)
ok_d = analysis.get('oracle_kw_visible', {}).get('d', 0)
pw = pairwise_analysis.get('oracle_trunc_vs_oracle_kw_visible', {})
print(f"\nQ1: Does selective keyword visibility help beyond truncation?")
print(f"  oracle_trunc: d={ot_d:+.3f}")
print(f"  oracle_kw_visible: d={ok_d:+.3f}")
print(f"  Pairwise: d={pw.get('d', 0):+.3f}, p={pw.get('p', 1):.2e}")
if ok_d > ot_d + 0.02 and pw.get('p', 1) < 0.05:
    print(f"  >>> YES: keyword visibility provides additional benefit")
elif ok_d < ot_d - 0.02 and pw.get('p', 1) < 0.05:
    print(f"  >>> NO: keyword visibility HURTS (like full visibility)")
else:
    print(f"  >>> NO significant difference: keyword visibility has minimal effect")

# Q2: Does Exp 01 replicate?
of_d = analysis.get('oracle_full', {}).get('d', 0)
print(f"\nQ2: Does Exp 01 replicate (trunc > full)?")
print(f"  oracle_trunc: d={ot_d:+.3f}")
print(f"  oracle_full:  d={of_d:+.3f}")
if ot_d > of_d:
    print(f"  >>> YES: truncation is better (replicates Exp 01)")
else:
    print(f"  >>> NO: full visibility is better (contradicts Exp 01)")

# Q3: Is there a sweet spot between full and trunc?
print(f"\nQ3: Is oracle_kw_visible a sweet spot?")
if ok_d > ot_d and ok_d > of_d:
    print(f"  >>> YES: selective > trunc ({ok_d:+.3f} > {ot_d:+.3f}) "
          f"AND selective > full ({ok_d:+.3f} > {of_d:+.3f})")
elif ok_d > of_d:
    print(f"  >>> PARTIAL: selective > full but not > trunc")
else:
    print(f"  >>> NO: truncation remains optimal")

# Q4: Template keyword results
tt_d = analysis.get('template_trunc', {}).get('d', 0)
tk_d = analysis.get('template_kw_visible', {}).get('d', 0)
print(f"\nQ4: Template keyword visibility?")
print(f"  template_trunc: d={tt_d:+.3f}")
print(f"  template_kw_visible: d={tk_d:+.3f}")

print(f"\n{'='*70}")

# Save results
final_results = {
    'experiment': 'exp10_selective_truncation',
    'model': MODEL_NAME,
    'n_samples': len(all_results),
    'n_bonferroni': N_BONFERRONI,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'analysis': analysis,
    'pairwise_analysis': pairwise_analysis,
    'hardness_analysis': hardness_analysis,
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

# Cleanup
print("Cleaning up GPU memory...")
mem_before = torch.cuda.memory_allocated() / 1e9
del model, processor, tokenizer
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

outpath = "experiments/10/10_selective_truncation.ipynb"
with open(outpath, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"Notebook written to {outpath} ({len(cells)} cells)")
