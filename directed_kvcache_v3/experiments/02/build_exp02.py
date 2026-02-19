#!/usr/bin/env python3
"""Generate Exp 02 notebook: Surrogate Type Sweep.

Which surrogate produces the best document representations when co-encoded
with T5Gemma's bidirectional encoder?

Exp 01 proved truncation works (doc reps genuinely improved). But we only tested
2 surrogates: doc keywords and paraphrased query. v2 showed surprising results --
static phrases beat LLM-generated surrogates 2x on Mistral (Exp 07). The mechanism
in T5Gemma is fundamentally different (bidirectional co-encoding, not value
contamination), so the surrogate hierarchy may change.

Key question: Does T5Gemma show a genuine CONTENT GRADIENT (more relevant surrogates
produce better document representations)? v2 found NO gradient (Spearman r=+0.036).
"""
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
md("""# Experiment 02: Surrogate Type Sweep
## Which surrogate works best for bidirectional co-encoding?

### Background
Exp 01 proved that document representations are genuinely improved by co-encoding
with a query/surrogate (truncation made the benefit STRONGER: d=+0.408 trunc vs
+0.345 full). But we only tested 2 surrogate types.

v2 showed surprising results on decoder-only models:
- Static "What are the key facts?" beat LLM-generated surrogates 2x (Exp 07)
- No semantic content gradient detected (Exp 10, Spearman r=+0.036)
- Content barely mattered -- the mechanism was structural (value contamination)

T5Gemma's bidirectional encoder should be fundamentally different. If the encoder
creates genuine query-document interactions, then content-specific surrogates
should outperform content-agnostic ones.

### Conditions (9 total, all truncated)
| # | Condition | Source | Semantic relevance |
|---|-----------|--------|--------------------|
| 1 | bare | -- | Lower bound |
| 2 | oracle_trunc | Real query | Upper bound (100%) |
| 3 | surr_doc_trunc | Top-5 TF keywords from document | Doc-specific |
| 4 | surr_para_trunc | Query keywords reversed | Query-specific |
| 5 | static_fact_trunc | "What are the key facts?" | Content-agnostic |
| 6 | static_howto_trunc | "How do I do this?" | Content-agnostic |
| 7 | random_trunc | Passage from unrelated sample | Structural control |
| 8 | surr_lead_trunc | First sentence of document | Doc-specific (rich) |
| 9 | surr_template_trunc | "What is [top_keyword]?" | Doc-specific (minimal) |

### Success criteria
- oracle_trunc replicates Exp 01 (d ~ +0.41)
- Content gradient: oracle > para > doc_kw > lead > template > static > random
- If gradient exists: confirms bidirectional mechanism is semantic
- If no gradient: mechanism is structural (like v2), just stronger in encoder-decoder
""")

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

RESULTS_DIR = Path("../../results/exp02")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

N_SAMPLES = 500
MODEL_NAME = "google/t5gemma-2-4b-4b"

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"Exp 02: Surrogate Type Sweep")
print(f"Model: {MODEL_NAME}")
print(f"N: {N_SAMPLES}")
print(f"CUDA: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")""")

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
code(r"""# Cell 4: Scoring and surrogate helpers

def score_nll(encoder_text, answer_text, prefix_token_count=0, truncate=False):
    # Score NLL of answer tokens with optional truncation.
    # When truncate=True: encoder processes full input bidirectionally,
    # but decoder can only cross-attend to document positions (prefix masked).
    enc_ids = tokenizer(encoder_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids.to(DEVICE)
    total_enc_len = enc_ids.shape[1]

    enc_mask = torch.ones(1, total_enc_len, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        encoder_outputs = model.get_encoder()(
            input_ids=enc_ids, attention_mask=enc_mask
        )

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
    # Count how many tokens the prefix occupies in the concatenated encoding.
    full_text = prefix_text + "\n" + document_text
    full_ids = tokenizer(full_text, add_special_tokens=True).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True).input_ids
    return len(full_ids) - len(doc_ids)


# === Surrogate generators ===
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


def make_surrogate_paraphrase(query):
    # Reversed query keywords (paraphrase proxy)
    keywords = extract_keywords(query)
    return " ".join(keywords[::-1]) if keywords else query


def make_surrogate_doc_kw(passage):
    # Top-5 TF keywords from the document
    content_words = extract_keywords(passage)
    if not content_words:
        return "information"
    counts = Counter(content_words)
    return " ".join(w for w, _ in counts.most_common(5))


def make_surrogate_lead(passage):
    # First sentence of the document
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', passage.strip())
    first = sentences[0] if sentences else passage[:100]
    # Cap length to avoid very long surrogates
    words = first.split()
    if len(words) > 25:
        first = " ".join(words[:25])
    return first


def make_surrogate_template(passage):
    # Auto-generated question template: 'What is [top_keyword]?'
    content_words = extract_keywords(passage)
    if not content_words:
        return "What is this about?"
    counts = Counter(content_words)
    top_word = counts.most_common(1)[0][0]
    return f"What is {top_word}?"


STATIC_FACT = "What are the key facts I need to know?"
STATIC_HOWTO = "How do I do this?"

print("Helpers defined. Surrogate types:")
print("  1. oracle      - real query (upper bound)")
print("  2. surr_doc    - top-5 TF keywords from document")
print("  3. surr_para   - reversed query keywords")
print("  4. static_fact - 'What are the key facts I need to know?'")
print("  5. static_howto- 'How do I do this?'")
print("  6. random      - passage from unrelated sample")
print("  7. surr_lead   - first sentence of document")
print("  8. surr_template - 'What is [top_keyword]?'")""")

# ============================================================
code("""# Cell 5: Load data
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
                'word_count': wc
            })
            break

np.random.seed(SEED)
np.random.shuffle(samples)
samples = samples[:N_SAMPLES]
del ds
gc.collect()

# Pre-compute surrogates
for i, s in enumerate(samples):
    s['surr_para'] = make_surrogate_paraphrase(s['query'])
    s['surr_doc_kw'] = make_surrogate_doc_kw(s['passage'])
    s['surr_lead'] = make_surrogate_lead(s['passage'])
    s['surr_template'] = make_surrogate_template(s['passage'])
    # Random: use passage from a different sample (circular offset)
    other_idx = (i + N_SAMPLES // 2) % len(samples)
    other_passage = samples[other_idx]['passage']
    # Use first ~20 words of the other passage as the random surrogate
    s['surr_random'] = " ".join(other_passage.split()[:20])

print(f"Selected {len(samples)} samples, mean words={np.mean([s['word_count'] for s in samples]):.0f}")""")

# ============================================================
code(r"""# Cell 6: Explain conditions with concrete examples
print("=" * 70)
print("EXPERIMENTAL CONDITIONS (all with truncation)")
print("=" * 70)

ex = samples[0]
print(f"\nExample query:     {ex['query'][:80]}")
print(f"Example answer:    {ex['answer'][:80]}")
print(f"Example passage:   {ex['passage'][:80]}...")
print()

surrogates = {
    'oracle':        ex['query'],
    'surr_doc':      ex['surr_doc_kw'],
    'surr_para':     ex['surr_para'],
    'static_fact':   STATIC_FACT,
    'static_howto':  STATIC_HOWTO,
    'random':        ex['surr_random'],
    'surr_lead':     ex['surr_lead'],
    'surr_template': ex['surr_template'],
}

# Expected semantic relevance ranking (for gradient test)
SEMANTIC_RANK = {
    'oracle_trunc': 1,        # exact match
    'surr_para_trunc': 2,     # query-derived
    'surr_doc_trunc': 3,      # doc-specific keywords
    'surr_lead_trunc': 4,     # doc-specific sentence
    'surr_template_trunc': 5, # doc-specific minimal
    'static_fact_trunc': 6,   # content-agnostic (best v2)
    'static_howto_trunc': 7,  # content-agnostic
    'random_trunc': 8,        # structural control
}

print(f"{'Condition':<22} {'Prefix tokens':>14} {'Surrogate text (first 60 chars)'}")
print("-" * 100)

for name, surr_text in surrogates.items():
    ptoks = count_prefix_tokens(surr_text, ex['passage'])
    display = surr_text[:60] + ('...' if len(surr_text) > 60 else '')
    print(f"  {name:<20} {ptoks:>14} {display}")

print(f"\n  bare                            0 (no surrogate -- lower bound)")

print(f"\n--- Semantic relevance ranking (for gradient test) ---")
print("  1=most relevant (oracle) ... 8=least relevant (random)")
for name, rank in sorted(SEMANTIC_RANK.items(), key=lambda x: x[1]):
    print(f"  {rank}. {name}")

print("\n--- Key question ---")
print("  v2 Mistral: NO content gradient (Spearman r=+0.036)")
print("  v3 T5Gemma: Does bidirectional attention create a genuine gradient?")""")

# ============================================================
code(r"""# Cell 7: Run scoring
print("=" * 70)
print("RUNNING EXPERIMENT")
print("=" * 70)

COND_NAMES = [
    'bare',
    'oracle_trunc',
    'surr_doc_trunc',
    'surr_para_trunc',
    'static_fact_trunc',
    'static_howto_trunc',
    'random_trunc',
    'surr_lead_trunc',
    'surr_template_trunc',
]


def make_conditions(sample):
    # Return dict of {name: (encoder_text, prefix_token_count, truncate)}
    query = sample['query']
    passage = sample['passage']

    surr_map = {
        'oracle':        query,
        'surr_doc':      sample['surr_doc_kw'],
        'surr_para':     sample['surr_para'],
        'static_fact':   STATIC_FACT,
        'static_howto':  STATIC_HOWTO,
        'random':        sample['surr_random'],
        'surr_lead':     sample['surr_lead'],
        'surr_template': sample['surr_template'],
    }

    conditions = {
        'bare': (passage, 0, False),
    }

    for surr_name, surr_text in surr_map.items():
        cond_name = f'{surr_name}_trunc'
        enc_text = surr_text + "\n" + passage
        prefix_count = count_prefix_tokens(surr_text, passage)
        conditions[cond_name] = (enc_text, prefix_count, True)

    return conditions


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
            print(f"Resuming: {start_idx}/{N_SAMPLES}")

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES, desc="Scoring"):
    s = samples[i]
    conditions = make_conditions(s)

    result = {
        'query': s['query'], 'answer': s['answer'],
        'passage_words': s['word_count'],
    }

    for cond_name in COND_NAMES:
        enc_text, prefix_count, trunc = conditions[cond_name]
        nll = score_nll(enc_text, s['answer'], prefix_count, trunc)
        result[f'nll_{cond_name}'] = nll

    all_results.append(result)

    if (i + 1) % 20 == 0 or i == N_SAMPLES - 1:
        ckpt = {
            'n_total': N_SAMPLES, 'results': all_results,
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
code(r"""# Cell 8: Results
from lib.analysis import cohens_d

print("=" * 70)
print(f"RESULTS (N={len(all_results)})")
print("=" * 70)

bare_nlls = np.array([r['nll_bare'] for r in all_results])

print(f"\n{'Condition':<25} {'Mean NLL':>10} {'vs Bare':>10} {'d':>8} {'Win%':>8} {'p':>12} {'sig':>5} {'% oracle':>10}")
print("-" * 95)

analysis = {}
oracle_d = None

for cond in COND_NAMES:
    nlls = np.array([r[f'nll_{cond}'] for r in all_results])
    mean_nll = nlls.mean()
    diff = bare_nlls - nlls
    d = cohens_d(diff)
    win_pct = 100 * np.mean(diff > 0)

    if cond == 'bare':
        print(f"{cond:<25} {mean_nll:>10.4f} {'--':>10} {'--':>8} {'--':>8} {'--':>12} {'--':>5} {'--':>10}")
        analysis[cond] = {'mean_nll': float(mean_nll)}
    else:
        t_stat, p_val = stats.ttest_1samp(diff, 0)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

        if cond == 'oracle_trunc':
            oracle_d = d
            pct_oracle = '100% (UB)'
        elif oracle_d and oracle_d > 0:
            pct_oracle = f"{d / oracle_d * 100:.0f}%"
        else:
            pct_oracle = '--'

        print(f"{cond:<25} {mean_nll:>10.4f} {diff.mean():>+10.4f} {d:>+8.3f} {win_pct:>7.1f}% {p_val:>12.2e} {sig:>5} {pct_oracle:>10}")
        analysis[cond] = {
            'mean_nll': float(mean_nll), 'delta': float(diff.mean()),
            'd': float(d), 'win_pct': float(win_pct), 'p': float(p_val),
        }

# Bonferroni correction
n_tests = len(COND_NAMES) - 1  # exclude bare
bonferroni_threshold = 0.05 / n_tests
print(f"\nBonferroni threshold: p < {bonferroni_threshold:.4f} (alpha=0.05, {n_tests} tests)")
for cond, a in analysis.items():
    if cond != 'bare' and 'p' in a:
        bf_sig = a['p'] < bonferroni_threshold
        if bf_sig:
            print(f"  {cond}: p={a['p']:.2e} -- SIGNIFICANT after Bonferroni")""")

# ============================================================
code(r"""# Cell 9: Content gradient analysis
print("=" * 70)
print("CONTENT GRADIENT ANALYSIS")
print("=" * 70)
print("Does more relevant content produce better document representations?")

# Spearman rank correlation: semantic relevance rank vs effect size (d)
ranks = []
ds = []
cond_labels = []

for cond, rank in sorted(SEMANTIC_RANK.items(), key=lambda x: x[1]):
    if cond in analysis and 'd' in analysis[cond]:
        ranks.append(rank)
        ds.append(analysis[cond]['d'])
        cond_labels.append(cond)

r_spearman, p_spearman = stats.spearmanr(ranks, ds)
r_pearson, p_pearson = stats.pearsonr(ranks, ds)

print(f"\nSemantic relevance rank vs Cohen's d:")
print(f"  {'Rank':<6} {'Condition':<25} {'d':>8}")
print(f"  {'-'*42}")
for rank, cond, d_val in zip(ranks, cond_labels, ds):
    print(f"  {rank:<6} {cond:<25} {d_val:>+8.3f}")

print(f"\n  Spearman rho = {r_spearman:+.3f} (p={p_spearman:.3f})")
print(f"  Pearson r    = {r_pearson:+.3f} (p={p_pearson:.3f})")

if r_spearman < -0.5 and p_spearman < 0.05:
    print(f"\n  STRONG CONTENT GRADIENT: More relevant surrogates produce")
    print(f"  significantly better representations. The bidirectional mechanism")
    print(f"  is genuinely semantic -- content matters.")
elif r_spearman < -0.3:
    print(f"\n  MODERATE GRADIENT: Some evidence that content helps, but")
    print(f"  not a clean monotonic relationship.")
else:
    print(f"\n  NO CONTENT GRADIENT (like v2): Surrogate content does not")
    print(f"  predict effect size. The mechanism may be structural rather")
    print(f"  than semantic, even with bidirectional attention.")

# v2 comparison
print(f"\n  v2 Mistral (Exp 10): Spearman r=+0.036 (no gradient)")
print(f"  v3 T5Gemma (this):   Spearman r={r_spearman:+.3f}")

# --- Group comparison: content-specific vs content-agnostic ---
print(f"\n{'='*70}")
print("GROUP COMPARISON: Content-specific vs Content-agnostic")
print("=" * 70)

content_specific = ['oracle_trunc', 'surr_para_trunc', 'surr_doc_trunc',
                    'surr_lead_trunc', 'surr_template_trunc']
content_agnostic = ['static_fact_trunc', 'static_howto_trunc', 'random_trunc']

# Per-sample comparison: average NLL across content-specific vs content-agnostic
specific_nlls = []
agnostic_nlls = []
for r in all_results:
    spec = np.mean([r[f'nll_{c}'] for c in content_specific])
    agn = np.mean([r[f'nll_{c}'] for c in content_agnostic])
    specific_nlls.append(spec)
    agnostic_nlls.append(agn)

specific_nlls = np.array(specific_nlls)
agnostic_nlls = np.array(agnostic_nlls)
diff_groups = agnostic_nlls - specific_nlls  # positive = specific better

d_groups = cohens_d(diff_groups)
win_groups = 100 * np.mean(diff_groups > 0)
t_groups, p_groups = stats.ttest_1samp(diff_groups, 0)

print(f"\n  Content-specific (oracle, para, doc_kw, lead, template):")
print(f"    Mean NLL = {specific_nlls.mean():.4f}")
print(f"  Content-agnostic (static_fact, static_howto, random):")
print(f"    Mean NLL = {agnostic_nlls.mean():.4f}")
print(f"\n  Difference: d={d_groups:+.3f}, win%={win_groups:.1f}%, p={p_groups:.2e}")
if d_groups > 0.1 and p_groups < 0.05:
    print(f"  --> Content-SPECIFIC surrogates are significantly better.")
    print(f"      Bidirectional co-encoding is genuinely semantic!")
elif d_groups > 0:
    print(f"  --> Content-specific slightly better but not significant.")
else:
    print(f"  --> Content-agnostic is as good or better (structural mechanism).")

# --- Document-derived vs query-derived ---
print(f"\n{'='*70}")
print("DOCUMENT-DERIVED vs QUERY-DERIVED (deployment question)")
print("=" * 70)
print("Can we get good results without any query information?")

doc_derived = ['surr_doc_trunc', 'surr_lead_trunc', 'surr_template_trunc']
query_derived = ['surr_para_trunc']  # only para uses query info (oracle excluded -- it IS the query)

doc_ds = [analysis[c]['d'] for c in doc_derived if c in analysis]
query_ds = [analysis[c]['d'] for c in query_derived if c in analysis]

print(f"\n  Document-derived surrogates (no query needed at build time):")
for c in doc_derived:
    if c in analysis:
        print(f"    {c}: d={analysis[c]['d']:+.3f}")
print(f"    Mean d: {np.mean(doc_ds):+.3f}")

print(f"\n  Query-derived surrogates (needs query proxy at build time):")
for c in query_derived:
    if c in analysis:
        print(f"    {c}: d={analysis[c]['d']:+.3f}")

best_doc = max(doc_derived, key=lambda c: analysis.get(c, {}).get('d', -999))
best_doc_d = analysis.get(best_doc, {}).get('d', 0)
oracle_d_val = analysis.get('oracle_trunc', {}).get('d', 0)
pct_of_oracle = best_doc_d / oracle_d_val * 100 if oracle_d_val > 0 else 0

print(f"\n  Best document-derived: {best_doc} (d={best_doc_d:+.3f}, {pct_of_oracle:.0f}% of oracle)")
print(f"  --> {'PRACTICAL: ' if pct_of_oracle > 70 else 'NEEDS QUERY INFO: '}",
      f"{'doc-derived surrogates capture enough of the oracle gap for deployment' if pct_of_oracle > 70 else 'may need query-like surrogates for practical benefit'}")""")

# ============================================================
code(r"""# Cell 10: Hardness stratification
print("=" * 70)
print("HARDNESS STRATIFICATION")
print("=" * 70)

quintile_bounds = np.percentile(bare_nlls, [20, 40, 60, 80])
quintiles = np.digitize(bare_nlls, quintile_bounds)

print(f"\n{'Quintile':<12} {'N':>4}", end="")
for cond in ['oracle_trunc', 'surr_doc_trunc', 'static_fact_trunc', 'random_trunc']:
    print(f" {cond.replace('_trunc',''):>12}", end="")
print()
print("-" * 60)

for q in range(5):
    mask = quintiles == q
    n_q = mask.sum()
    if n_q < 3:
        continue
    qlabel = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard'][q]
    print(f"{qlabel:<12} {n_q:>4}", end="")

    b_q = bare_nlls[mask]
    for cond in ['oracle_trunc', 'surr_doc_trunc', 'static_fact_trunc', 'random_trunc']:
        c_nlls = np.array([all_results[j][f'nll_{cond}'] for j in range(len(all_results)) if mask[j]])
        diff_q = b_q - c_nlls
        d_q = cohens_d(diff_q)
        print(f" {d_q:>+12.3f}", end="")
    print()

# Hardness-benefit correlation for each condition
print(f"\n--- Hardness-benefit correlation (r with bare NLL) ---")
for cond in COND_NAMES[1:]:  # skip bare
    cond_nlls = np.array([r[f'nll_{cond}'] for r in all_results])
    benefit = bare_nlls - cond_nlls
    r_hb, p_hb = stats.pearsonr(bare_nlls, benefit)
    print(f"  {cond:<25} r={r_hb:+.3f} (p={p_hb:.2e})")""")

# ============================================================
code(r"""# Cell 11: Pairwise comparisons (direct head-to-head)
print("=" * 70)
print("PAIRWISE HEAD-TO-HEAD COMPARISONS")
print("=" * 70)
print("Direct comparisons between key surrogate pairs.\n")

pairs = [
    ('oracle_trunc', 'surr_doc_trunc', "Oracle vs doc keywords (how close?)"),
    ('oracle_trunc', 'surr_para_trunc', "Oracle vs paraphrase"),
    ('surr_doc_trunc', 'static_fact_trunc', "Doc keywords vs static (content matters?)"),
    ('surr_doc_trunc', 'surr_lead_trunc', "Doc keywords vs lead sentence"),
    ('surr_doc_trunc', 'random_trunc', "Doc keywords vs random (semantic signal?)"),
    ('static_fact_trunc', 'random_trunc', "Static fact vs random (structural baseline)"),
    ('surr_lead_trunc', 'surr_template_trunc', "Lead sentence vs question template"),
]

print(f"{'Comparison':<50} {'d':>8} {'Win%':>8} {'p':>12} {'sig':>5}")
print("-" * 85)

for cond_a, cond_b, desc in pairs:
    nlls_a = np.array([r[f'nll_{cond_a}'] for r in all_results])
    nlls_b = np.array([r[f'nll_{cond_b}'] for r in all_results])
    diff = nlls_b - nlls_a  # positive = A is better (lower NLL)
    d = cohens_d(diff)
    win = 100 * np.mean(diff > 0)
    t, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    winner = cond_a.replace('_trunc', '') if d > 0 else cond_b.replace('_trunc', '')
    print(f"  {desc:<48} {d:>+8.3f} {win:>7.1f}% {p:>12.2e} {sig:>5}  [{winner}]")""")

# ============================================================
code(r"""# Cell 12: Verdict and save
print("=" * 70)
print("VERDICT -- Exp 02: Surrogate Type Sweep")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"N: {len(all_results)} samples")

# Rank conditions by d
ranked = [(c, analysis[c]['d']) for c in COND_NAMES[1:] if 'd' in analysis.get(c, {})]
ranked.sort(key=lambda x: -x[1])

print(f"\n--- Rankings (by Cohen's d) ---")
oracle_d_val = analysis.get('oracle_trunc', {}).get('d', 0)
for i, (cond, d_val) in enumerate(ranked, 1):
    pct = d_val / oracle_d_val * 100 if oracle_d_val > 0 else 0
    print(f"  {i}. {cond:<25} d={d_val:+.3f} ({pct:.0f}% of oracle)")

# Best document-derived (for Exp 03/04)
doc_surrogates = ['surr_doc_trunc', 'surr_lead_trunc', 'surr_template_trunc']
best_doc = max(doc_surrogates, key=lambda c: analysis.get(c, {}).get('d', -999))
best_doc_d = analysis.get(best_doc, {}).get('d', 0)
best_doc_pct = best_doc_d / oracle_d_val * 100 if oracle_d_val > 0 else 0

print(f"\n--- Recommendation for Exps 03-04 ---")
print(f"  Best document-derived surrogate: {best_doc}")
print(f"  d={best_doc_d:+.3f} ({best_doc_pct:.0f}% of oracle)")

# Content gradient conclusion
print(f"\n--- Content gradient ---")
print(f"  Spearman rho = {r_spearman:+.3f} (p={p_spearman:.3f})")
if r_spearman < -0.5 and p_spearman < 0.05:
    print(f"  CONTENT GRADIENT CONFIRMED: T5Gemma's bidirectional encoding is semantic.")
    print(f"  More relevant surrogates genuinely produce better document representations.")
else:
    print(f"  NO CONTENT GRADIENT (same as v2): Content does not predict effect size.")
    print(f"  The benefit may be structural rather than semantic.")

# Upper/lower/middle bound framing
print(f"\n--- Three-point spectrum ---")
print(f"  Upper bound (oracle_trunc):      d={oracle_d_val:+.3f}")
print(f"  Middle ground (best doc-derived): d={best_doc_d:+.3f} ({best_doc_pct:.0f}%)")
print(f"  Lower bound (bare):              d=0.000")
print(f"  Structural floor (random):       d={analysis.get('random_trunc', {}).get('d', 0):+.3f}")

print(f"\n{'='*70}")

# Save
final_results = {
    'experiment': 'exp02_surrogate_type_sweep',
    'model': MODEL_NAME,
    'n_samples': len(all_results),
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'analysis': analysis,
    'content_gradient': {
        'spearman_rho': float(r_spearman),
        'spearman_p': float(p_spearman),
        'pearson_r': float(r_pearson),
        'pearson_p': float(p_pearson),
    },
    'best_doc_derived': best_doc,
    'best_doc_derived_d': float(best_doc_d),
    'best_doc_derived_pct_oracle': float(best_doc_pct),
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")""")

# ============================================================
code("""# Cell 13: Cleanup
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
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4, "nbformat_minor": 5
}

outpath = "experiments/02/02_surrogate_type_sweep.ipynb"
with open(outpath, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"Notebook written to {outpath} ({len(cells)} cells)")
