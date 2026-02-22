#!/usr/bin/env python3
# Build Exp 05 notebook: Truncation Wound Mechanism.
#
# Exp 04 revealed that the first sentence of a document is CATASTROPHIC as a prefix
# (d=-0.298 ***), while disconnected keywords help. This experiment dissects WHY.
#
# Two hypotheses:
#   H1: OVERLAP — The first sentence shares content with the document. During
#       bidirectional encoding, prefix-document attention connections are very strong.
#       After truncation, document tokens that "leaned on" the prefix lose those
#       attention targets, creating representation gaps.
#   H2: COHERENCE — Any coherent natural text (even non-overlapping) creates deep
#       attention patterns during encoding that are disrupted by truncation.
#       Keywords work because they're disconnected tokens that don't form deep deps.
#
# To distinguish: test first sentence from WRONG document (coherent, non-overlapping)
# and shuffled first sentence (same words, broken coherence).
#
# 10 conditions, N=500.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 05: Truncation Wound Mechanism

## Motivation

Exp 04 revealed a striking finding: the first sentence of the document used as an
encoder prefix is **catastrophic** (d=-0.298 ***), while disconnected keywords help
(kw10 d=+0.186). This is the "truncation wound" phenomenon — the prefix creates
strong bidirectional attention connections during encoding, and masking those tokens
from cross-attention leaves the document representations *worse* than bare encoding.

But what specifically causes the wound? Two hypotheses:

1. **Overlap hypothesis**: The first sentence shares vocabulary and semantics with the
   document. This creates unusually strong attention connections during encoding. When
   truncated, the "wound" is proportional to connection strength.

2. **Coherence hypothesis**: Any coherent natural text creates deep attention dependencies
   during encoding, regardless of overlap. Keywords work specifically because they're
   disconnected tokens that perturb without creating deep dependencies.

## Conditions (10 total)

### Text type sweep (all with query in decoder + truncation):

| # | Condition | Prefix | Tests |
|---|-----------|--------|-------|
| 1 | bare | (none) | Baseline |
| 2 | oracle_trunc | real query | Ceiling |
| 3 | surr_kw10 | top-10 kw from THIS doc | Best surrogate (Exp 04) |
| 4 | surr_first_sent | first sent of THIS doc | Catastrophic in Exp 04 |
| 5 | surr_wrong_first_sent | first sent of WRONG doc | Coherence without overlap |
| 6 | surr_shuffled_sent | shuffled first sent of THIS doc | Overlap without coherence |
| 7 | surr_wrong_kw10 | top-10 kw from WRONG doc | Keywords without specificity |
| 8 | surr_generic_sent | fixed generic sentence | Coherent, no overlap, no info |

### Controls:

| # | Condition | Purpose |
|---|-----------|---------|
| 9 | bare_nq | v3 baseline |
| 10 | oracle_trunc_nq | v3 replication |

## Key comparisons

- **(4) vs (5)**: Same-doc vs wrong-doc first sentence → isolates **overlap**
- **(4) vs (6)**: Coherent vs shuffled same-doc first sentence → isolates **coherence**
- **(5) vs (8)**: Wrong-doc sentence vs generic sentence → information content within coherent text
- **(3) vs (7)**: Same-doc vs wrong-doc keywords → replicates Exp 04 specificity test
- **(6) vs (3)**: Shuffled sentence (long) vs keywords (short) → length/format comparison""")


# ===== Cell 2: Setup + model loading =====
code(r"""# Cell 2: Setup and model loading
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
MODEL_NAME = "google/t5gemma-2-4b-4b"

RESULTS_DIR = Path("../../../results/exp05")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

from transformers import AutoProcessor, AutoModelForSeq2SeqLM

print(f"Loading {MODEL_NAME}...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer = processor.tokenizer

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN,
)
model.eval()

DEVICE = next(model.parameters()).device
BOS_ID = getattr(model.config, 'decoder_start_token_id', None) or tokenizer.bos_token_id

print(f"Exp 05: Truncation Wound Mechanism")
print(f"N: {N_SAMPLES}, Model: {MODEL_NAME}")
print(f"DEVICE: {DEVICE}, dtype: {next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
""")


# ===== Cell 3: Scoring helpers =====
code(r"""# Cell 3: Scoring helpers

def count_prefix_tokens(prefix_text, document_text):
    full_text = prefix_text + "\n" + document_text
    full_ids = tokenizer(full_text, add_special_tokens=True, truncation=True,
                         max_length=2048).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids
    return len(full_ids) - len(doc_ids)


def score_nll(encoder_text, answer_text, prefix_token_count=0, truncate=False):
    # No query in decoder — used for _nq conditions.
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
        del encoder_outputs
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


def score_nll_query_prefix(encoder_text, query_text, answer_text,
                           prefix_token_count=0, truncate=False):
    # Query as decoder prefix — production-realistic.
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

    query_ids = tokenizer(query_text, add_special_tokens=False, truncation=True,
                          max_length=512).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False, truncation=True,
                           max_length=256).input_ids

    if len(answer_ids) == 0:
        del encoder_outputs
        return 0.0

    dec_ids = [BOS_ID] + query_ids + answer_ids
    dec_tensor = torch.tensor([dec_ids], dtype=torch.long, device=DEVICE)

    n_query = len(query_ids)
    n_answer = len(answer_ids)

    with torch.no_grad():
        outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=cross_attn_mask,
            decoder_input_ids=dec_tensor,
        )

    logits = outputs.logits
    answer_logits = logits[0, n_query:n_query + n_answer, :]

    answer_targets = torch.tensor(answer_ids, dtype=torch.long, device=DEVICE)
    log_probs = F.log_softmax(answer_logits, dim=-1)
    token_log_probs = log_probs.gather(1, answer_targets.unsqueeze(1)).squeeze(1)
    mean_nll = -token_log_probs.mean().item()

    del encoder_outputs, outputs, logits, log_probs
    return mean_nll


# === Surrogate helpers ===
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

def make_kw_surrogate(passage, n_keywords):
    content_words = extract_keywords(passage)
    if not content_words:
        return "information"
    counts = Counter(content_words)
    return " ".join(w for w, _ in counts.most_common(n_keywords))

def get_first_sentence(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    if parts:
        return parts[0]
    return text[:100]

def shuffle_sentence(text):
    # Shuffle words in a sentence, preserving the word set but breaking coherence.
    words = text.split()
    pyrandom.shuffle(words)
    return " ".join(words)

# Fixed generic sentence — coherent but zero information about any document.
GENERIC_SENTENCE = "The following passage contains relevant information about the topic."

print("Scoring functions defined.")
""")


# ===== Cell 4: Load data + generate surrogates =====
code(r"""# Cell 4: Load MS MARCO data and generate all prefix variants
from lib.data import count_words
from datasets import load_dataset

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

# Generate all prefix variants
pyrandom.seed(SEED)
for i, s in enumerate(samples):
    passage = s['passage']
    wrong_idx = (i + N_SAMPLES // 2) % len(samples)

    # Keywords from THIS document (best surrogate from Exp 04)
    s['surr_kw10'] = make_kw_surrogate(passage, 10)

    # First sentence of THIS document (catastrophic in Exp 04)
    s['surr_first_sent'] = get_first_sentence(passage)

    # First sentence of WRONG document (coherent, non-overlapping)
    s['surr_wrong_first_sent'] = get_first_sentence(samples[wrong_idx]['passage'])

    # Shuffled first sentence of THIS document (overlap without coherence)
    s['surr_shuffled_sent'] = shuffle_sentence(get_first_sentence(passage))

    # Keywords from WRONG document (replicates Exp 04 control)
    s['surr_wrong_kw10'] = make_kw_surrogate(samples[wrong_idx]['passage'], 10)

    # Generic sentence (coherent, zero information)
    s['surr_generic_sent'] = GENERIC_SENTENCE

    # Count prefix tokens for each
    s['n_pfx_oracle'] = count_prefix_tokens(s['query'], passage)
    s['n_pfx_kw10'] = count_prefix_tokens(s['surr_kw10'], passage)
    s['n_pfx_first_sent'] = count_prefix_tokens(s['surr_first_sent'], passage)
    s['n_pfx_wrong_first_sent'] = count_prefix_tokens(s['surr_wrong_first_sent'], passage)
    s['n_pfx_shuffled_sent'] = count_prefix_tokens(s['surr_shuffled_sent'], passage)
    s['n_pfx_wrong_kw10'] = count_prefix_tokens(s['surr_wrong_kw10'], passage)
    s['n_pfx_generic_sent'] = count_prefix_tokens(s['surr_generic_sent'], passage)

# Statistics
print(f"\nLoaded {len(samples)} samples")

print(f"\nPrefix token counts (mean):")
for key, label in [
    ('n_pfx_oracle', 'oracle (query)'),
    ('n_pfx_kw10', 'kw10 (this doc)'),
    ('n_pfx_first_sent', 'first_sent (this)'),
    ('n_pfx_wrong_first_sent', 'wrong_first_sent'),
    ('n_pfx_shuffled_sent', 'shuffled_sent'),
    ('n_pfx_wrong_kw10', 'wrong_kw10'),
    ('n_pfx_generic_sent', 'generic_sent'),
]:
    vals = [s[key] for s in samples]
    print(f"  {label:<22}: mean={np.mean(vals):.1f}, min={np.min(vals)}, max={np.max(vals)}")

print(f"\nFirst sample:")
print(f"  Query:             {samples[0]['query']}")
print(f"  kw10:              {samples[0]['surr_kw10']}")
print(f"  first_sent:        {samples[0]['surr_first_sent'][:80]}")
print(f"  wrong_first_sent:  {samples[0]['surr_wrong_first_sent'][:80]}")
print(f"  shuffled_sent:     {samples[0]['surr_shuffled_sent'][:80]}")
print(f"  wrong_kw10:        {samples[0]['surr_wrong_kw10']}")
print(f"  generic_sent:      {samples[0]['surr_generic_sent']}")
""")


# ===== Cell 5: Scoring loop =====
code(r"""# Cell 5: Scoring loop — 10 conditions x 500 samples
print("=" * 70)
print("SCORING ALL CONDITIONS")
print("=" * 70)

COND_NAMES = [
    'bare', 'oracle_trunc', 'surr_kw10',
    'surr_first_sent', 'surr_wrong_first_sent', 'surr_shuffled_sent',
    'surr_wrong_kw10', 'surr_generic_sent',
    'bare_nq', 'oracle_trunc_nq',
]

results = []
start_idx = 0

if CHECKPOINT_PATH.exists():
    ckpt = json.loads(CHECKPOINT_PATH.read_text())
    if ckpt.get('n_total') == N_SAMPLES and len(ckpt.get('results', [])) > 0:
        saved_queries = [r['query'][:50] for r in ckpt['results']]
        current_queries = [s['query'][:50] for s in samples[:len(saved_queries)]]
        if saved_queries == current_queries:
            results = ckpt['results']
            start_idx = len(results)
            print(f"Resuming from checkpoint: {start_idx}/{N_SAMPLES}")

if start_idx == 0:
    print(f"Starting fresh: {len(COND_NAMES)} conditions x {N_SAMPLES} samples "
          f"= {len(COND_NAMES) * N_SAMPLES} forward passes")

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
              desc="Scoring"):
    s = samples[i]
    query = s['query']
    passage = s['passage']
    answer = s['answer']

    result = {
        'query': query,
        'answer': answer,
        'passage_words': s['word_count'],
    }

    # --- With query in decoder ---

    # 1. bare
    result['nll_bare'] = score_nll_query_prefix(passage, query, answer)

    # 2. oracle_trunc
    result['nll_oracle_trunc'] = score_nll_query_prefix(
        query + "\n" + passage, query, answer,
        prefix_token_count=s['n_pfx_oracle'], truncate=True)

    # 3. surr_kw10 (this doc — best surrogate from Exp 04)
    result['nll_surr_kw10'] = score_nll_query_prefix(
        s['surr_kw10'] + "\n" + passage, query, answer,
        prefix_token_count=s['n_pfx_kw10'], truncate=True)

    # 4. surr_first_sent (this doc — catastrophic in Exp 04)
    result['nll_surr_first_sent'] = score_nll_query_prefix(
        s['surr_first_sent'] + "\n" + passage, query, answer,
        prefix_token_count=s['n_pfx_first_sent'], truncate=True)

    # 5. surr_wrong_first_sent (wrong doc — coherent, non-overlapping)
    result['nll_surr_wrong_first_sent'] = score_nll_query_prefix(
        s['surr_wrong_first_sent'] + "\n" + passage, query, answer,
        prefix_token_count=s['n_pfx_wrong_first_sent'], truncate=True)

    # 6. surr_shuffled_sent (this doc shuffled — overlap without coherence)
    result['nll_surr_shuffled_sent'] = score_nll_query_prefix(
        s['surr_shuffled_sent'] + "\n" + passage, query, answer,
        prefix_token_count=s['n_pfx_shuffled_sent'], truncate=True)

    # 7. surr_wrong_kw10 (wrong doc keywords)
    result['nll_surr_wrong_kw10'] = score_nll_query_prefix(
        s['surr_wrong_kw10'] + "\n" + passage, query, answer,
        prefix_token_count=s['n_pfx_wrong_kw10'], truncate=True)

    # 8. surr_generic_sent (fixed generic sentence)
    result['nll_surr_generic_sent'] = score_nll_query_prefix(
        s['surr_generic_sent'] + "\n" + passage, query, answer,
        prefix_token_count=s['n_pfx_generic_sent'], truncate=True)

    # --- Without query in decoder ---

    # 9. bare_nq
    result['nll_bare_nq'] = score_nll(passage, answer)

    # 10. oracle_trunc_nq
    result['nll_oracle_trunc_nq'] = score_nll(
        query + "\n" + passage, answer,
        prefix_token_count=s['n_pfx_oracle'], truncate=True)

    results.append(result)

    if (i + 1) % 20 == 0 or i == N_SAMPLES - 1:
        ckpt = {
            'n_total': N_SAMPLES,
            'results': results,
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
print(f"\nScoring complete: {len(results)} samples, "
      f"{len(COND_NAMES)} conditions in {elapsed/60:.1f} min")
""")


# ===== Cell 6: Results table =====
code(r"""# Cell 6: Results table
print("=" * 70)
print(f"RESULTS (N={len(results)})")
print("=" * 70)

# Extract NLL arrays
bare = np.array([r['nll_bare'] for r in results])
oracle_trunc = np.array([r['nll_oracle_trunc'] for r in results])
kw10 = np.array([r['nll_surr_kw10'] for r in results])
first_sent = np.array([r['nll_surr_first_sent'] for r in results])
wrong_first_sent = np.array([r['nll_surr_wrong_first_sent'] for r in results])
shuffled_sent = np.array([r['nll_surr_shuffled_sent'] for r in results])
wrong_kw10 = np.array([r['nll_surr_wrong_kw10'] for r in results])
generic_sent = np.array([r['nll_surr_generic_sent'] for r in results])
bare_nq = np.array([r['nll_bare_nq'] for r in results])
oracle_nq = np.array([r['nll_oracle_trunc_nq'] for r in results])

N_BONF = 7  # 7 comparisons vs bare (excluding bare itself and _nq conditions)
d_oracle = cohens_d(bare - oracle_trunc)

print(f"\n--- With query in decoder ---")
print(f"  Bonferroni: {N_BONF} comparisons")
print(f"\n  {'Condition':<24} {'NLL':>8} {'delta':>8} {'d':>8} {'Win%':>8} {'p':>12} {'sig':>5}")
print(f"  {'-'*78}")

analysis = {}
for name, nlls in [
    ('bare', bare),
    ('oracle_trunc', oracle_trunc),
    ('surr_kw10', kw10),
    ('surr_wrong_kw10', wrong_kw10),
    ('surr_shuffled_sent', shuffled_sent),
    ('surr_generic_sent', generic_sent),
    ('surr_wrong_first_sent', wrong_first_sent),
    ('surr_first_sent', first_sent),
]:
    mean_nll = nlls.mean()
    if name == 'bare':
        print(f"  {name:<24} {mean_nll:>8.4f} {'--':>8} {'--':>8} {'--':>8} {'--':>12} {'--':>5}")
        analysis[name] = {'mean_nll': float(mean_nll)}
    else:
        diff = bare - nlls
        d = cohens_d(diff)
        win_pct = 100 * np.mean(diff > 0)
        _, p_val = stats.ttest_1samp(diff, 0)
        sig = '***' if p_val < 0.001/N_BONF else '**' if p_val < 0.01/N_BONF else '*' if p_val < 0.05/N_BONF else 'ns'
        print(f"  {name:<24} {mean_nll:>8.4f} {diff.mean():>+8.4f} {d:>+8.3f} {win_pct:>7.1f}% {p_val:>12.2e} {sig:>5}")
        analysis[name] = {
            'mean_nll': float(mean_nll), 'delta': float(diff.mean()),
            'd': float(d), 'win_pct': float(win_pct), 'p': float(p_val),
        }

# v3 replication
diff_nq = bare_nq - oracle_nq
d_nq = cohens_d(diff_nq)
_, p_nq = stats.ttest_1samp(diff_nq, 0)
sig_nq = '***' if p_nq < 0.001 else 'ns'
print(f"\n--- v3 replication ---")
print(f"  oracle_trunc_nq: d={d_nq:+.3f} ({sig_nq})")

analysis['bare_nq'] = {'mean_nll': float(bare_nq.mean())}
analysis['oracle_trunc_nq'] = {
    'mean_nll': float(oracle_nq.mean()), 'd': float(d_nq), 'p': float(p_nq),
}
""")


# ===== Cell 7: Hypothesis testing =====
code(r"""# Cell 7: Hypothesis testing — overlap vs coherence
print("=" * 70)
print("HYPOTHESIS TESTING: What causes the truncation wound?")
print("=" * 70)

d_first = cohens_d(bare - first_sent)
d_wrong_first = cohens_d(bare - wrong_first_sent)
d_shuffled = cohens_d(bare - shuffled_sent)
d_generic = cohens_d(bare - generic_sent)
d_kw10_val = cohens_d(bare - kw10)
d_wrong_kw10 = cohens_d(bare - wrong_kw10)

# === Test 1: Overlap hypothesis ===
# If overlap causes the wound, same-doc first_sent should hurt MORE than wrong-doc first_sent
print(f"\n--- Test 1: Does overlap matter for sentences? ---")
print(f"  first_sent (this doc):    d={d_first:+.3f}")
print(f"  wrong_first_sent (other): d={d_wrong_first:+.3f}")
diff_overlap = wrong_first_sent - first_sent  # positive = this-doc is worse
d_overlap = cohens_d(diff_overlap)
_, p_overlap = stats.ttest_1samp(diff_overlap, 0)
sig_overlap = '***' if p_overlap < 0.001 else '**' if p_overlap < 0.01 else '*' if p_overlap < 0.05 else 'ns'
print(f"  Overlap effect (same vs wrong doc): d={d_overlap:+.3f} ({sig_overlap})")
if d_overlap > 0.05:
    print(f"  -> Same-doc sentence is WORSE. Overlap amplifies the wound.")
elif d_overlap < -0.05:
    print(f"  -> Wrong-doc sentence is WORSE. Overlap is NOT the driver.")
else:
    print(f"  -> No significant overlap effect. Both sentences hurt equally.")

# === Test 2: Coherence hypothesis ===
# If coherence causes the wound, coherent first_sent should hurt MORE than shuffled
print(f"\n--- Test 2: Does coherence matter? ---")
print(f"  first_sent (coherent):    d={d_first:+.3f}")
print(f"  shuffled_sent (broken):   d={d_shuffled:+.3f}")
diff_coherence = shuffled_sent - first_sent  # positive = coherent is worse
d_coherence = cohens_d(diff_coherence)
_, p_coherence = stats.ttest_1samp(diff_coherence, 0)
sig_coherence = '***' if p_coherence < 0.001 else '**' if p_coherence < 0.01 else '*' if p_coherence < 0.05 else 'ns'
print(f"  Coherence effect (coherent vs shuffled): d={d_coherence:+.3f} ({sig_coherence})")
if d_coherence > 0.05:
    print(f"  -> Coherent text is WORSE. Word order creates deeper attention dependencies.")
elif d_coherence < -0.05:
    print(f"  -> Shuffled is WORSE. Coherence is NOT the problem.")
else:
    print(f"  -> No coherence effect. The wound is not about word order.")

# === Test 3: Generic sentence ===
print(f"\n--- Test 3: Does information content matter within coherent text? ---")
print(f"  wrong_first_sent (informative): d={d_wrong_first:+.3f}")
print(f"  generic_sent (zero info):       d={d_generic:+.3f}")
diff_info = generic_sent - wrong_first_sent  # positive = informative is worse
d_info = cohens_d(diff_info)
_, p_info = stats.ttest_1samp(diff_info, 0)
sig_info = '***' if p_info < 0.001 else '**' if p_info < 0.01 else '*' if p_info < 0.05 else 'ns'
print(f"  Info content effect: d={d_info:+.3f} ({sig_info})")

# === Test 4: Keywords specificity (replicates Exp 04) ===
print(f"\n--- Test 4: Keyword specificity (Exp 04 replication) ---")
print(f"  kw10 (this doc):   d={d_kw10_val:+.3f}")
print(f"  wrong_kw10 (other): d={d_wrong_kw10:+.3f}")
diff_kw_spec = wrong_kw10 - kw10  # positive = this-doc kw is better
d_kw_spec = cohens_d(diff_kw_spec)
_, p_kw_spec = stats.ttest_1samp(diff_kw_spec, 0)
sig_kw_spec = '***' if p_kw_spec < 0.001 else '**' if p_kw_spec < 0.01 else '*' if p_kw_spec < 0.05 else 'ns'
print(f"  Specificity effect: d={d_kw_spec:+.3f} ({sig_kw_spec})")

# === Summary: 2x2 decomposition ===
print(f"\n--- 2x2 Decomposition: Coherence x Overlap ---")
print(f"  (using d vs bare as the outcome)")
print(f"")
print(f"  {'':>24} {'Same-doc':>12} {'Wrong-doc':>12} {'Difference':>12}")
print(f"  {'Coherent sentence':<24} {d_first:>+12.3f} {d_wrong_first:>+12.3f} {d_first-d_wrong_first:>+12.3f}")
print(f"  {'Shuffled/Keywords':<24} {d_shuffled:>+12.3f} {d_wrong_kw10:>+12.3f} {d_shuffled-d_wrong_kw10:>+12.3f}")
print(f"  {'Difference':<24} {d_first-d_shuffled:>+12.3f} {d_wrong_first-d_wrong_kw10:>+12.3f}")
print(f"")
print(f"  Coherence main effect: {(d_first + d_wrong_first)/2 - (d_shuffled + d_wrong_kw10)/2:+.3f}")
print(f"  Overlap main effect:   {(d_first + d_shuffled)/2 - (d_wrong_first + d_wrong_kw10)/2:+.3f}")
""")


# ===== Cell 8: Prefix length confound check =====
code(r"""# Cell 8: Is the wound just a length effect?
print("=" * 70)
print("LENGTH CONFOUND CHECK")
print("=" * 70)

# Sentences are longer than keywords — is length the real driver?
print(f"\nMean prefix tokens and d for each condition:")
for name, nlls, pfx_key in [
    ('oracle_trunc', oracle_trunc, 'n_pfx_oracle'),
    ('surr_kw10', kw10, 'n_pfx_kw10'),
    ('surr_wrong_kw10', wrong_kw10, 'n_pfx_wrong_kw10'),
    ('surr_shuffled_sent', shuffled_sent, 'n_pfx_shuffled_sent'),
    ('surr_generic_sent', generic_sent, 'n_pfx_generic_sent'),
    ('surr_wrong_first_sent', wrong_first_sent, 'n_pfx_wrong_first_sent'),
    ('surr_first_sent', first_sent, 'n_pfx_first_sent'),
]:
    pfx = np.array([s[pfx_key] for s in samples])
    d = cohens_d(bare - nlls)
    print(f"  {name:<24}: {pfx.mean():>5.1f} tokens, d={d:+.3f}")

# Key comparison: shuffled_sent has SAME length as first_sent (same words)
print(f"\nCritical length-controlled comparison:")
pfx_first = np.array([s['n_pfx_first_sent'] for s in samples])
pfx_shuffled = np.array([s['n_pfx_shuffled_sent'] for s in samples])
print(f"  first_sent tokens:   mean={pfx_first.mean():.1f}")
print(f"  shuffled_sent tokens: mean={pfx_shuffled.mean():.1f}")
print(f"  (Same words, same length — only word order differs)")
print(f"  first_sent d:   {cohens_d(bare - first_sent):+.3f}")
print(f"  shuffled_sent d: {cohens_d(bare - shuffled_sent):+.3f}")
print(f"  -> Any difference is purely due to coherence, not length.")

# Within first_sent: does per-sample prefix length predict damage?
pfx_len = np.array([s['n_pfx_first_sent'] for s in samples])
damage = first_sent - bare  # positive = first_sent is worse
r_len, p_len = stats.pearsonr(pfx_len, damage)
print(f"\nWithin first_sent: correlation of prefix length with damage:")
print(f"  r={r_len:.3f}, p={p_len:.3e}")
if r_len > 0.1:
    print(f"  -> Longer first sentences cause MORE damage.")
else:
    print(f"  -> Length does not predict damage magnitude.")
""")


# ===== Cell 9: Verdict + save =====
code(r"""# Cell 9: Verdict and save
print("=" * 70)
print("VERDICT — Exp 05: Truncation Wound Mechanism")
print("=" * 70)

d_first = cohens_d(bare - first_sent)
d_wrong_first = cohens_d(bare - wrong_first_sent)
d_shuffled = cohens_d(bare - shuffled_sent)
d_generic = cohens_d(bare - generic_sent)
d_kw10_val = cohens_d(bare - kw10)

print(f"\nModel: {MODEL_NAME}")
print(f"N: {len(results)} (MS MARCO v1.1)")

print(f"\n--- Condition summary ---")
print(f"  {'Condition':<24} {'d vs bare':>10} {'Interpretation'}")
print(f"  {'-'*70}")
for name, d, interp in [
    ('oracle_trunc', cohens_d(bare - oracle_trunc), 'Ceiling'),
    ('surr_kw10 (this)', d_kw10_val, 'Best surrogate'),
    ('surr_wrong_kw10', cohens_d(bare - wrong_kw10), 'Keywords without specificity'),
    ('surr_shuffled_sent', d_shuffled, 'Overlap without coherence'),
    ('surr_generic_sent', d_generic, 'Coherence without information'),
    ('surr_wrong_first_sent', d_wrong_first, 'Coherence without overlap'),
    ('surr_first_sent', d_first, 'Coherence + overlap (catastrophic)'),
]:
    print(f"  {name:<24} {d:>+10.3f}   {interp}")

# Determine which hypothesis won
overlap_effect = d_first - d_wrong_first
coherence_effect = d_first - d_shuffled
coherence_main = (d_first + d_wrong_first)/2 - (d_shuffled + cohens_d(bare - wrong_kw10))/2
overlap_main = (d_first + d_shuffled)/2 - (d_wrong_first + cohens_d(bare - wrong_kw10))/2

print(f"\n--- Hypothesis verdict ---")
print(f"  Coherence main effect: {coherence_main:+.3f}")
print(f"  Overlap main effect:   {overlap_main:+.3f}")

if abs(coherence_main) > abs(overlap_main) and coherence_main < -0.05:
    print(f"  -> H2 (COHERENCE) is the primary driver.")
    print(f"     Coherent text creates deep attention dependencies that truncation disrupts.")
elif abs(overlap_main) > abs(coherence_main) and overlap_main < -0.05:
    print(f"  -> H1 (OVERLAP) is the primary driver.")
    print(f"     Shared content creates extra-strong attention connections.")
elif coherence_main < -0.05 and overlap_main < -0.05:
    print(f"  -> BOTH coherence and overlap contribute to the wound.")
else:
    print(f"  -> Neither factor dominates clearly. The wound mechanism is more complex.")

# Save
final_results = {
    'experiment': 'v4_exp05_truncation_wound',
    'model': MODEL_NAME,
    'dataset': 'ms_marco_v1.1',
    'n_samples': len(results),
    'seed': SEED,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'bonferroni': N_BONF,
    'conditions': analysis,
    'hypothesis_tests': {
        'overlap_effect': float(overlap_effect),
        'coherence_effect': float(coherence_effect),
        'coherence_main_effect': float(coherence_main),
        'overlap_main_effect': float(overlap_main),
    },
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")

# Cleanup
print(f"\nCleaning up GPU memory...")
mem_before = torch.cuda.memory_allocated() / 1e9
del model, processor, tokenizer
gc.collect()
torch.cuda.empty_cache()
gc.collect()
mem_after = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory: {mem_before:.2f} GB -> {mem_after:.2f} GB")
print("Done!")
""")


# ===== Write notebook =====
out_path = "experiments/encoder_decoder/05/05_truncation_wound.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
