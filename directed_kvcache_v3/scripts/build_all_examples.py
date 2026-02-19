#!/usr/bin/env python3
# build_all_examples.py — Generate one small example notebook per experiment.
#
# Each notebook reconstructs sample 0 from the dataset (same seed/filtering),
# generates all condition texts, and prints them. No GPU or model needed.
# Cross-checks against checkpoint queries where available.

import json
import nbformat as nbf

# ============================================================
# Shared code blocks
# ============================================================

MSMARCO_LOAD_SEED42 = r"""import os, sys, json, re
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, "../..")
from lib.data import count_words

SEED = 42

# ---- Load MS MARCO (same reconstruction as Exp 01/02/etc.) ----
from datasets import load_dataset
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

samples = []
for item in ds:
    if len(samples) >= 1500:
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
samples = samples[:500]
del ds

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
    keywords = extract_keywords(query)
    return " ".join(keywords[::-1]) if keywords else query

def make_surrogate_from_doc(passage):
    content_words = extract_keywords(passage)
    if not content_words:
        return "information"
    counts = Counter(content_words)
    return " ".join(w for w, _ in counts.most_common(5))

def make_surrogate_template(passage):
    content_words = extract_keywords(passage)
    if not content_words:
        return "What is this about?"
    counts = Counter(content_words)
    top_word = counts.most_common(1)[0][0]
    return f"What is {top_word}?"

# Verify against checkpoint
def verify_checkpoint(exp_name):
    ckpt_path = Path(f"../../results/{exp_name}/checkpoint.json")
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        results = ckpt.get('results', [])
        if results and results[0].get('query', '')[:50] == samples[0]['query'][:50]:
            print(f"  Checkpoint verification: MATCH ({exp_name})")
            return True
        elif results:
            print(f"  Checkpoint verification: MISMATCH ({exp_name})")
            print(f"    Checkpoint: {results[0].get('query', '')[:50]}")
            print(f"    Samples:    {samples[0]['query'][:50]}")
            return False
    else:
        print(f"  No checkpoint found for {exp_name}")
    return None

print(f"Loaded {len(samples)} MS MARCO samples (SEED={SEED})")
print(f"Sample 0 query: {samples[0]['query'][:70]}")
"""

NEURAL_BRIDGE_LOAD = r"""import os, sys, json, re
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, "../..")

SEED = 42
N_SAMPLES = 500

from datasets import load_dataset
ds = load_dataset("neural-bridge/rag-dataset-12000", split="train")

all_candidates = []
for row in ds:
    q = row.get("question", "")
    doc = row.get("context", "")
    answer = row.get("answer", "")
    if not q or not doc or not answer:
        continue
    q_words = len(q.split())
    a_words = len(answer.split())
    if q_words >= 15 and a_words >= 5:
        all_candidates.append({
            "query": q, "document": doc, "answer": answer,
            "query_words": q_words, "doc_words": len(doc.split()),
            "answer_words": a_words,
        })

np.random.seed(SEED)
indices = np.random.permutation(len(all_candidates))
samples = [all_candidates[i] for i in indices[:N_SAMPLES]]
del ds

# Verify against checkpoint
def verify_checkpoint(exp_name):
    ckpt_path = Path(f"../../results/{exp_name}/checkpoint.json")
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        meta = ckpt.get('sample_meta', ckpt.get('results', []))
        if meta and meta[0].get('query', '')[:50] == samples[0]['query'][:50]:
            print(f"  Checkpoint verification: MATCH ({exp_name})")
            return True
    return None

print(f"Loaded {len(samples)} neural-bridge samples (SEED={SEED})")
print(f"Sample 0 query ({samples[0]['query_words']}w): {samples[0]['query'][:70]}")
"""

DISPLAY_HEADER = r"""
def show_sample(s, doc_key='passage', n=0):
    # Show sample info
    doc = s[doc_key]
    print(f"{'='*80}")
    print(f"SAMPLE {n}")
    print(f"{'='*80}")
    print(f"  Query:    {s['query']}")
    print(f"  Answer:   {s['answer']}")
    print(f"  Document: {doc[:100]}...")
    print(f"  Doc words: {len(doc.split())}")
    print()

def show_conditions(conditions, doc_text):
    # conditions: list of (name, description, encoder_prefix_text_or_None)
    # For bare conditions, encoder_prefix_text is None
    print(f"{'Condition':<30} {'Prefix':<14} {'Encoder input (first 70 chars)'}")
    print(f"{'-'*100}")
    for name, desc, prefix_text in conditions:
        if prefix_text is None:
            enc_preview = doc_text[:70]
            print(f"{name:<30} {'(none)':<14} {enc_preview}...")
        else:
            enc_text = prefix_text + "\n" + doc_text
            print(f"{name:<30} {str(len(prefix_text.split()))+'w':<14} {enc_text[:70]}...")
        if desc:
            print(f"  {'':>28} ^ {desc}")
    print()
"""


# ============================================================
# Per-experiment condition generators
# ============================================================

def make_exp01_display():
    return r"""
show_sample(samples[0])
verify_checkpoint("exp01")

ex = samples[0]
surr_para = make_surrogate_paraphrase(ex['query'])
surr_doc_kw = make_surrogate_from_doc(ex['passage'])

print("CONDITIONS: 7 total (3 prefixes x full/trunc + bare)")
print("  _full: decoder cross-attends to ALL encoder tokens (prefix + doc)")
print("  _trunc: decoder cross-attends to doc tokens ONLY (prefix masked)")
print()

conditions = [
    ("bare", "Baseline — document only", None),
    ("oracle_full", "Real query, decoder sees query+doc", ex['query']),
    ("oracle_trunc", "Real query, decoder sees doc ONLY", ex['query']),
    ("surr_para_full", "Reversed query keywords, decoder sees all", surr_para),
    ("surr_para_trunc", "Reversed query keywords, decoder sees doc ONLY", surr_para),
    ("surr_doc_full", "Top-5 TF keywords from doc, decoder sees all", surr_doc_kw),
    ("surr_doc_trunc", "Top-5 TF keywords from doc, decoder sees doc ONLY", surr_doc_kw),
]
show_conditions(conditions, ex['passage'])

print("KEY QUESTION: If _trunc ~= _full, the benefit is from improved doc representations.")
print("              If _trunc ~= bare, the decoder was just reading the prefix.")
"""


def make_exp02_display():
    return r"""
show_sample(samples[0])
verify_checkpoint("exp02")

ex = samples[0]
surr_doc_kw = make_surrogate_from_doc(ex['passage'])
surr_para = make_surrogate_paraphrase(ex['query'])
surr_template = make_surrogate_template(ex['passage'])
surr_lead = ex['passage'].split('.')[0] + '.'
other_idx = (0 + 250) % len(samples)
surr_random = " ".join(samples[other_idx]['passage'].split()[:20])

conditions = [
    ("bare", "Baseline", None),
    ("oracle_trunc", "Real query", ex['query']),
    ("surr_doc_trunc", "Top-5 TF keywords from doc", surr_doc_kw),
    ("surr_para_trunc", "Reversed query keywords", surr_para),
    ("static_fact_trunc", "Fixed question", "What are the key facts I need to know?"),
    ("static_howto_trunc", "Fixed question", "How do I do this?"),
    ("random_trunc", "~20 words from unrelated passage", surr_random),
    ("surr_lead_trunc", "First sentence of document", surr_lead),
    ("surr_template_trunc", "'What is [keyword]?'", surr_template),
]
show_conditions(conditions, ex['passage'])

print("KEY QUESTION: Is there a content gradient? (more relevant = bigger effect)")
print("              Or is the benefit mostly structural? (any prefix helps)")
"""


def make_exp02b_display():
    return r"""
show_sample(samples[0])
verify_checkpoint("exp02b")

ex = samples[0]
rng = np.random.RandomState(SEED + 1)

# Build vocab pool
all_words = []
for s in samples:
    all_words.extend(extract_keywords(s['passage']))
vocab_pool = np.array(list(set(all_words)))

query_words = ex['query'].split()
n_qw = len(query_words)

# Surrogates
shuffled = list(query_words)
rng_tmp = np.random.RandomState(SEED + 1)
rng_tmp.shuffle(shuffled)
scrambled_oracle = " ".join(shuffled)

rand_matched = " ".join(rng.choice(vocab_pool, size=min(n_qw, len(vocab_pool)), replace=True).tolist())

conditions = [
    ("bare", "Baseline", None),
    ("oracle_trunc", "Real query", ex['query']),
    ("scrambled_oracle_trunc", "Query words, random order", scrambled_oracle),
    ("rand_matched_trunc", f"Random words, query-length matched ({n_qw}w)", rand_matched),
    ("rand_1w_trunc", "1 random word", rng.choice(vocab_pool)),
    ("rand_3w_trunc", "3 random words", " ".join(rng.choice(vocab_pool, 3))),
    ("rand_5w_trunc", "5 random words", " ".join(rng.choice(vocab_pool, 5))),
    ("rand_10w_trunc", "10 random words", " ".join(rng.choice(vocab_pool, 10))),
    ("rand_20w_trunc", "20 random words", " ".join(rng.choice(vocab_pool, 20))),
    ("rand_50w_trunc", "50 random words", " ".join(rng.choice(vocab_pool, 50))),
    ("repeat_the_trunc", "'the' repeated 10x", " ".join(["the"] * 10)),
    ("repeat_kw_trunc", "Top doc keyword repeated 10x",
     " ".join([extract_keywords(ex['passage'])[0]] * 10) if extract_keywords(ex['passage']) else "information " * 10),
]
show_conditions(conditions, ex['passage'])

print("THREE-WAY DECOMPOSITION:")
print("  Structure  = bare -> random_matched  (any prefix helps)")
print("  Vocabulary = random_matched -> scrambled_oracle  (right words, wrong order)")
print("  Semantics  = scrambled_oracle -> oracle  (right word order)")
"""


def make_exp03_display():
    return r"""
show_sample(samples[0])

ex = samples[0]
surr_para = make_surrogate_paraphrase(ex['query'])
surr_doc_kw = make_surrogate_from_doc(ex['passage'])

print("CONDITIONS: 4 (all with truncation) x 6 LENGTH BINS")
print("  Length bins: original (~130 tok), 256, 384, 512, 1024, 2048")
print("  Same passage padded with unrelated text to reach target length.")
print()

conditions = [
    ("bare", "Document only", None),
    ("oracle_trunc", "Real query", ex['query']),
    ("surr_doc_trunc", "Top-5 TF keywords from doc", surr_doc_kw),
    ("surr_para_trunc", "Reversed query keywords", surr_para),
]
show_conditions(conditions, ex['passage'])

# Show what padding looks like
print("PADDING EXAMPLE (512 token bin):")
print(f"  Original passage ({ex['word_count']}w): {ex['passage'][:80]}...")
print(f"  After padding: [original passage] \\n\\n [unrelated passage 1] \\n\\n [unrelated passage 2] ...")
print(f"  Target: 512 tokens total. Question and answer unchanged.")
print()
print("KEY QUESTION: Does the benefit survive longer documents?")
print("  v2 (decoder-only): benefit collapsed by 256 tokens")
print("  v3 (encoder-decoder): ???")
"""


def make_exp03b_display():
    return r"""
show_sample(samples[0])

ex = samples[0]
rng = np.random.RandomState(SEED + 1)

all_words = []
for s in samples:
    all_words.extend(extract_keywords(s['passage']))
vocab_pool = np.array(list(set(all_words)))

n_qw = len(ex['query'].split())
shuffled = list(ex['query'].split())
rng.shuffle(shuffled)
scrambled = " ".join(shuffled)
rand_matched = " ".join(rng.choice(vocab_pool, size=min(n_qw, len(vocab_pool)), replace=True).tolist())
other_idx = (0 + 250) % len(samples)
surr_random = " ".join(samples[other_idx]['passage'].split()[:20])

conditions = [
    ("bare", "Baseline", None),
    ("oracle_trunc", "Real query", ex['query']),
    ("scrambled_oracle_trunc", "Query words shuffled", scrambled),
    ("random_matched_trunc", f"Random words, {n_qw}w matched", rand_matched),
    ("random_trunc", "~20w from unrelated passage", surr_random),
    ("static_fact_trunc", "Fixed question", "What are the key facts I need to know?"),
    ("surr_template_trunc", "'What is [kw]?'", make_surrogate_template(ex['passage'])),
    ("surr_doc_trunc", "Top-5 TF keywords", make_surrogate_from_doc(ex['passage'])),
]
show_conditions(conditions, ex['passage'])

print("8 CONDITIONS x 7 LENGTH BINS: original, 512, 1024, 2048, 3072, 4096, 6144")
print("  Encoder sliding window = 1024 tokens, full attention every 6th layer.")
print("  At 6144 tokens, only ~17% of tokens in each window layer can see the prefix.")
print()
print("THREE-WAY DECOMPOSITION at each length:")
print("  Structure  = bare -> random_matched")
print("  Vocabulary = random_matched -> scrambled_oracle")
print("  Semantics  = scrambled_oracle -> oracle")
"""


def make_exp03d_display():
    return NEURAL_BRIDGE_LOAD + "\n" + DISPLAY_HEADER + r"""
show_sample(samples[0], doc_key='document')
verify_checkpoint("exp03d")

ex = samples[0]
query_words = ex['query'].split()
n_qw = len(query_words)

other_idx = (0 + N_SAMPLES // 2) % N_SAMPLES
other_doc = samples[other_idx]['document']
other_words = other_doc.split()

rand_matched = " ".join(other_words[:n_qw])

rng = np.random.RandomState(SEED + 1)
shuffled = list(query_words)
rng.shuffle(shuffled)
scrambled = " ".join(shuffled)

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
import re
def extract_keywords(text):
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]

doc_kws = extract_keywords(ex['document'])
counts = Counter(doc_kws) if doc_kws else Counter()
top_kw = counts.most_common(1)[0][0] if counts else "topic"
surr_template = f"What is {top_kw}?"

conditions = [
    ("bare", "Baseline", None),
    ("oracle_trunc", "Real query", ex['query']),
    ("random_matched_trunc", f"Random words, {n_qw}w matched", rand_matched),
    ("scrambled_oracle_trunc", "Query words shuffled", scrambled),
    ("surr_template_trunc", "'What is [kw]?'", surr_template),
    ("repeat_the_trunc", f"'the' x {n_qw}", " ".join(["the"] * n_qw)),
    ("repeat_kw_trunc", f"Top doc kw x {n_qw}",
     " ".join([top_kw] * n_qw)),
]
show_conditions(conditions, ex['document'])

print("CROSS-DATASET TEST: Same decomposition as Exp 2B but on neural-bridge.")
print(f"  Queries are ~{n_qw} words (vs ~6w in MS MARCO)")
print(f"  Documents are ~{ex['doc_words']} words (vs ~90w in MS MARCO)")
"""


def make_exp03e_display():
    return NEURAL_BRIDGE_LOAD + "\n" + DISPLAY_HEADER + r"""
show_sample(samples[0], doc_key='document')
verify_checkpoint("exp03e")

ex = samples[0]
query_words = ex['query'].split()
n_qw = len(query_words)

other_idx = (0 + N_SAMPLES // 2) % N_SAMPLES
other_doc = samples[other_idx]['document']
rand_matched = " ".join(other_doc.split()[:n_qw])
repeat_the = " ".join(["the"] * n_qw)

conditions = [
    ("bare", "Document only — baseline encoder attention", None),
    ("oracle_trunc", "Real query — do prefix tokens get special attention?", ex['query']),
    ("random_matched_trunc", f"Random words, {n_qw}w — structural control", rand_matched),
    ("repeat_the_trunc", f"'the' x {n_qw} — minimal diversity control", repeat_the),
]

show_conditions(conditions, ex['document'])

print("PROBES (all measured at encoder attention layers):")
print("  A: Attention mass — how much doc-token attention goes to prefix?")
print("  B: Entropy — does prefix increase or decrease attention entropy?")
print("  C: Doc-doc redistribution — KL divergence of doc-doc attention pattern")
print("  D: Shift magnitude — L2 distance of doc representations (bare vs prefixed)")
print("  E: Shift direction — cosine similarity of shift vectors across conditions")
print("  F: Attention sinks — does prefix take over the sink role from <bos>?")
"""


def make_exp03f_display():
    return r"""
show_sample(samples[0])
verify_checkpoint("exp03f")

ex = samples[0]
query = ex['query']
query_words = query.split()
n_qw = len(query_words)
content_words = [w for w in query_words if w.lower() not in STOP_WORDS]
content_text = " ".join(content_words) if content_words else query_words[0]

other_idx = (0 + 250) % len(samples)
rand_matched = " ".join(samples[other_idx]['passage'].split()[:n_qw])

rng = np.random.RandomState(SEED + 0)
shuffled = list(query_words)
rng.shuffle(shuffled)
scrambled = " ".join(shuffled)

rand_content = " ".join(samples[other_idx]['passage'].split()[:len(content_words)])

conditions = [
    ("bare", "Baseline", None),
    ("oracle_x1_trunc", "Query x 1", query),
    ("oracle_x3_trunc", "Query x 3", " ".join([query] * 3)),
    ("oracle_x5_trunc", "Query x 5", " ".join([query] * 5)),
    ("oracle_x10_trunc", "Query x 10", " ".join([query] * 10)),
    ("random_x1_trunc", "Random matched x 1", rand_matched),
    ("random_x5_trunc", "Random matched x 5", " ".join([rand_matched] * 5)),
    ("random_x10_trunc", "Random matched x 10", " ".join([rand_matched] * 10)),
    ("scrambled_x5_trunc", "Scrambled query x 5", " ".join([scrambled] * 5)),
    ("content_x5_trunc", f"Content words only x 5 ({len(content_words)}w)",
     " ".join([content_text] * 5)),
    ("random_content_x5_trunc", f"Random matched to content len x 5",
     " ".join([rand_content] * 5)),
    ("the_matched10_trunc", "'the' x M (token-matched to oracle_x10)",
     " ".join(["the"] * (n_qw * 10))),  # approximate
    ("bare_short", "Doc truncated to 30 words", None),
    ("oracle_short_trunc", "Query x 1, doc = first 30 words", query),
]

print("20 CONDITIONS — testing semantic amplification via repetition")
print()
show_conditions(conditions, ex['passage'])

print("Short doc (30 words):", " ".join(ex['passage'].split()[:30]))
print()
print("KEY QUESTION: Does repeating the prefix amplify the semantic component?")
print("  Exp 2B: ~10% semantic at x1. Does x5/x10/x20 push it higher?")
"""


def make_exp04a_display():
    # This one needs passage pools - slightly different data structure
    return r"""
show_sample(samples[0])

ex = samples[0]
surr_doc_kw = make_surrogate_from_doc(ex['passage'])
surr_template = make_surrogate_template(ex['passage'])
other_idx = (0 + 250) % len(samples)
surr_random = " ".join(samples[other_idx]['passage'].split()[:20])

print("RANKING EXPERIMENT: Does co-encoding create differential signal?")
print("  Each query has ~10 candidate passages. One is relevant.")
print("  Surrogates are per-PASSAGE (not per-query), except oracle and random.")
print()
print("  For the RELEVANT passage:")
conditions = [
    ("bare", "Passage only", None),
    ("oracle_trunc", "Real query", ex['query']),
    ("surr_template_trunc", "'What is [kw]?' (per-passage)", surr_template),
    ("surr_doc_trunc", "Top-5 TF keywords (per-passage)", surr_doc_kw),
    ("random_trunc", "~20w from unrelated passage", surr_random),
    ("static_fact_trunc", "Fixed question", "What are the key facts I need to know?"),
]
show_conditions(conditions, ex['passage'])

print("KEY QUESTION: Does oracle help the relevant passage MORE than irrelevant ones?")
print("  If yes: co-encoding creates ranking signal.")
print("  If no: the structural boost is document-independent (dead end for ranking).")
"""


def make_exp04b_display():
    return r"""import os, sys, json
import numpy as np
from pathlib import Path

sys.path.insert(0, "../..")

SEED = 42
N_QUERIES = 400

from datasets import load_dataset
ds = load_dataset("tasksource/esci", split="train")

# Reconstruct ESCI samples
queries = {}
for row in ds:
    q = row.get('query', '')
    label = row.get('esci_label', '')
    product_title = row.get('product_title', '')
    product_desc = row.get('product_description', '')
    if not q or not product_title:
        continue
    text = product_title
    if product_desc and len(product_desc) > 20:
        text = product_title + " " + product_desc
    if len(text.split()) < 10:
        continue
    if q not in queries:
        queries[q] = {'products': [], 'labels': []}
    queries[q]['products'].append(text)
    queries[q]['labels'].append(label)

# Filter: need at least 1 exact + 1 irrelevant
usable = []
for q, data in queries.items():
    has_exact = 'E' in data['labels'] or 'Exact' in data['labels']
    has_irrel = 'I' in data['labels'] or 'Irrelevant' in data['labels']
    if has_exact and has_irrel and len(data['products']) >= 3:
        usable.append({'query': q, **data})

if not usable:
    # Debug: show what labels look like
    all_labels = set()
    for data in queries.values():
        all_labels.update(data['labels'])
    print(f"WARNING: No usable queries found! Unique labels in dataset: {all_labels}")
    print(f"Total queries collected: {len(queries)}")

np.random.seed(SEED)
np.random.shuffle(usable)
usable = usable[:N_QUERIES]

print(f"Loaded {len(usable)} ESCI queries with products")
del ds

ex = usable[0]
exact_idx = ex['labels'].index('Exact') if 'Exact' in ex['labels'] else ex['labels'].index('E')
irrel_idx = ex['labels'].index('Irrelevant') if 'Irrelevant' in ex['labels'] else ex['labels'].index('I')
rel_product = ex['products'][exact_idx]
irrel_product = ex['products'][irrel_idx]

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
import re
def extract_keywords(text):
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]
from collections import Counter
def make_surrogate_from_doc(passage):
    content_words = extract_keywords(passage)
    if not content_words:
        return "information"
    counts = Counter(content_words)
    return " ".join(w for w, _ in counts.most_common(5))
def make_surrogate_template(passage):
    content_words = extract_keywords(passage)
    if not content_words:
        return "What is this about?"
    counts = Counter(content_words)
    return f"What is {counts.most_common(1)[0][0]}?"

# Product title as surrogate (natural for ad-serving)
title = rel_product.split()[0:8]  # approximate title
surr_title = " ".join(title)

print(f"Query: {ex['query']}")
print(f"Relevant product (E): {rel_product[:120]}...")
print(f"Irrelevant product (I): {irrel_product[:120]}...")
print()

print("QUERY-LIKELIHOOD RANKING on graded-relevance product search.")
print("  Decoder scores the QUERY (not an answer) given encoded product.")
print()

""" + DISPLAY_HEADER + r"""
conditions = [
    ("bare", "Product text only", None),
    ("oracle_trunc", "Real search query", ex['query']),
    ("surr_title_trunc", "Product title (natural surrogate)", surr_title),
    ("surr_doc_trunc", "Top-5 TF keywords from product", make_surrogate_from_doc(rel_product)),
    ("surr_template_trunc", "'What is [kw]?'", make_surrogate_template(rel_product)),
    ("random_trunc", "~20w from another query's product",
     " ".join(usable[1]['products'][0].split()[:20])),
]
print("For the RELEVANT product:")
show_conditions(conditions, rel_product)

print("KEY DIFFERENCE from Exp 04A: decoder scores the query, not a gold answer.")
"""


def make_exp05_display():
    return r"""
show_sample(samples[0])
verify_checkpoint("exp05")

ex = samples[0]

# Load LLM-generated surrogates
surr_path = Path("../../results/exp05/surrogates.json")
if surr_path.exists():
    all_surrogates = json.loads(surr_path.read_text())
    surr = all_surrogates[0]
    print(f"Loaded LLM surrogates from {surr_path}")
    print(f"  Prompt A (need):     {surr['llm_need']}")
    print(f"  Prompt B (question): {surr['llm_question']}")
    print(f"  Prompt C (keywords): {surr['llm_keywords']}")
    print()
else:
    surr = {'llm_need': '(not available)', 'llm_question': '(not available)', 'llm_keywords': '(not available)'}
    print("LLM surrogates not found — showing placeholders")

surr_template = make_surrogate_template(ex['passage'])
other_idx = (0 + 250) % len(samples)
rand_matched = " ".join(samples[other_idx]['passage'].split()[:len(ex['query'].split())])

rng = np.random.RandomState(SEED + 0)
shuffled_need = surr['llm_need'].split()
rng.shuffle(shuffled_need)
scrambled_need = " ".join(shuffled_need)

conditions = [
    ("bare", "Baseline", None),
    ("oracle_x1_trunc", "Real query x 1", ex['query']),
    ("oracle_x4_trunc", "Real query x 4", " ".join([ex['query']] * 4)),
    ("llm_need_x1_trunc", "Gemma 2 9B-IT: need-focused x 1", surr['llm_need']),
    ("llm_need_x4_trunc", "Gemma 2 9B-IT: need-focused x 4", " ".join([surr['llm_need']] * 4)),
    ("llm_question_x1_trunc", "Gemma 2 9B-IT: question x 1", surr['llm_question']),
    ("llm_question_x4_trunc", "Gemma 2 9B-IT: question x 4", " ".join([surr['llm_question']] * 4)),
    ("llm_keywords_x1_trunc", "Gemma 2 9B-IT: keywords x 1", surr['llm_keywords']),
    ("llm_keywords_x4_trunc", "Gemma 2 9B-IT: keywords x 4", " ".join([surr['llm_keywords']] * 4)),
    ("surr_template_x1_trunc", "'What is [kw]?' x 1", surr_template),
    ("surr_template_x4_trunc", "'What is [kw]?' x 4", " ".join([surr_template] * 4)),
    ("random_x1_trunc", "Random matched x 1", rand_matched),
    ("random_x4_trunc", "Random matched x 4", " ".join([rand_matched] * 4)),
    ("scrambled_llm_need_x4_trunc", "Shuffled need x 4 (vocab control)",
     " ".join([scrambled_need] * 4)),
]
show_conditions(conditions, ex['passage'])

print("KEY QUESTION: Can an LLM generate better surrogates than 'What is [kw]?'")
"""


def make_exp06_display():
    return r"""import os, sys, json, re
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, "../..")
from lib.data import count_words

SEED = 43  # Different seed from main experiments!

from datasets import load_dataset
ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

samples = []
for item in ds:
    if len(samples) >= 1500:
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
    if not answer or len(answer.split()) > 5:
        continue  # FACTOID FILTER: answer <= 5 words
    for pt, sel in zip(ptexts, is_sel):
        wc = count_words(pt)
        if sel == 1 and 30 <= wc <= 300:
            samples.append({
                'passage': pt, 'query': query, 'answer': answer,
                'word_count': wc, 'answer_words': len(answer.split()),
            })
            break

np.random.seed(SEED)
np.random.shuffle(samples)
samples = samples[:500]
del ds

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
    return f"What is {counts.most_common(1)[0][0]}?"

# Verify
ckpt_path = Path("../../results/exp06/checkpoint.json")
if ckpt_path.exists():
    ckpt = json.loads(ckpt_path.read_text())
    if ckpt['results'][0]['query'][:50] == samples[0]['query'][:50]:
        print("Checkpoint verification: MATCH")

print(f"Loaded {len(samples)} FACTOID samples (answer <= 5 words, SEED={SEED})")
print()

ex = samples[0]

""" + DISPLAY_HEADER + r"""
print(f"FACTOID SAMPLE (answer <= 5 words):")
show_sample(ex, n=0)
print(f"  Answer length: {ex['answer_words']} words")
print()

query_words = ex['query'].split()
n_qw = len(query_words)
other_idx = (0 + 250) % len(samples)
rand_matched = " ".join(samples[other_idx]['passage'].split()[:n_qw])
rng = np.random.RandomState(SEED + 0)
shuffled = list(query_words)
rng.shuffle(shuffled)
scrambled = " ".join(shuffled)
surr_template = make_surrogate_template(ex['passage'])

conditions = [
    ("bare", "Baseline", None),
    ("oracle_x1_trunc", "Real query x 1", ex['query']),
    ("oracle_x4_trunc", "Real query x 4", " ".join([ex['query']] * 4)),
    ("random_x1_trunc", "Random matched x 1", rand_matched),
    ("random_x4_trunc", "Random matched x 4", " ".join([rand_matched] * 4)),
    ("scrambled_oracle_trunc", "Query words shuffled", scrambled),
    ("surr_template_x1_trunc", "'What is [kw]?' x 1", surr_template),
    ("surr_template_x4_trunc", "'What is [kw]?' x 4", " ".join([surr_template] * 4)),
]
show_conditions(conditions, ex['passage'])

print("KEY QUESTION: Does the two-population structure exist?")
print("  Factoid QA (short answers) might have more semantic headroom.")
print("  Overall 85% structural might be a Simpson's paradox average.")
"""


def make_exp07_display():
    return r"""
show_sample(samples[0])
verify_checkpoint("exp07")

ex = samples[0]
n_qw = len(ex['query'].split())
other_idx = (0 + 250) % len(samples)
rand_matched = " ".join(samples[other_idx]['passage'].split()[:n_qw])
doc_len = len(ex['passage'].split())

print("2x2 FACTORIAL: {RoPE shift ON/OFF} x {Prefix in attention ON/OFF}")
print()

print(f"{'Condition':<28} {'Encoder text':<22} {'position_ids':<30} {'Decoder sees'}")
print(f"{'-'*100}")
print(f"{'bare':<28} {'[doc]':<22} {'[0..L-1] (standard)':<30} {'all'}")
print(f"{'oracle_trunc':<28} {'[query + doc]':<22} {'[0..N+L-1] (default)':<30} {'doc only'}")
print(f"  prefix: {ex['query'][:60]}")
print(f"{'random_trunc':<28} {'[random + doc]':<22} {'[0..N+L-1] (default)':<30} {'doc only'}")
print(f"  prefix: {rand_matched[:60]}")
print(f"{'shifted_bare':<28} {'[doc]':<22} {'[N..N+L-1] (shifted)':<30} {'all'}")
print(f"  Pure absolute RoPE shift, no prefix tokens")
print(f"{'prefix_encoder_blocked':<28} {'[random + doc]':<22} {'[0..N+L-1] (default)':<30} {'doc only'}")
print(f"  prefix: {rand_matched[:60]}")
print(f"  BUT: doc tokens CANNOT attend to prefix in encoder (blocked)")
print(f"{'random_rope_neutralized':<28} {'[random + doc]':<22} {'prefix@[L..L+N],':<30} {'doc only'}")
print(f"  prefix: {rand_matched[:60]}")
print(f"  {'':>28} {'':>22} {'doc@[0..L-1]':<30}")
print(f"  Doc keeps original positions — no RoPE shift!")
print()

print("2x2 FACTORIAL TABLE:")
print(f"  {'':>25} | Prefix in attn (ON) | Prefix NOT in attn")
print(f"  {'-'*70}")
print(f"  {'RoPE shift ON':<25} | random_trunc        | prefix_encoder_blocked")
print(f"  {'RoPE shift OFF':<25} | rope_neutralized    | bare (baseline)")
print()
print("PREDICTIONS:")
print("  shifted_bare ~= bare (RoPE is relative, absolute shift irrelevant)")
print("  prefix_encoder_blocked ~= bare (invisible prefix has no effect)")
print("  rope_neutralized ~= random_trunc (benefit is from attention, not RoPE)")
"""


def make_exp08_display():
    return r"""
show_sample(samples[0])
verify_checkpoint("exp08")

ex = samples[0]
surr_doc_kw = make_surrogate_from_doc(ex['passage'])
surr_template = make_surrogate_template(ex['passage'])
other_idx = (0 + 250) % len(samples)
surr_random = " ".join(samples[other_idx]['passage'].split()[:20])

# Cross-query: a different sample's query
cross_query = samples[1]['query']

conditions = [
    ("bare", "Passage only", None),
    ("oracle_trunc", "Real query (matched to passage)", ex['query']),
    ("oracle_cross_trunc", "DIFFERENT query (structural only)", cross_query),
    ("surr_template_trunc", "'What is [kw]?' (per-passage)", surr_template),
    ("surr_doc_trunc", "Top-5 TF keywords (per-passage)", surr_doc_kw),
    ("random_trunc", "~20w from unrelated passage", surr_random),
    ("static_fact_trunc", "Fixed question", "What are the key facts I need to know?"),
]
show_conditions(conditions, ex['passage'])

print("KEY NEW CONDITION: oracle_cross_trunc uses a DIFFERENT query's text.")
print(f"  oracle:       '{ex['query'][:50]}...'")
print(f"  oracle_cross: '{cross_query[:50]}...'")
print()
print("  oracle - oracle_cross isolates semantic interaction (same structure).")
print("  If oracle >> oracle_cross: semantic signal exists for ranking.")
print("  If oracle ~= oracle_cross: structural mechanism is query-independent.")
"""


def make_exp09_display():
    return r"""
show_sample(samples[0])

ex = samples[0]
surr_template = make_surrogate_template(ex['passage'])
surr_doc_kw = make_surrogate_from_doc(ex['passage'])
other_idx = (0 + 250) % len(samples)
rand_matched = " ".join(samples[other_idx]['passage'].split()[:len(ex['query'].split())])
rng = np.random.RandomState(SEED + 1)
shuffled = list(ex['query'].split())
rng.shuffle(shuffled)
scrambled = " ".join(shuffled)
top_kw = extract_keywords(ex['passage'])
top_kw_word = top_kw[0] if top_kw else "topic"

conditions = [
    ("bare", "Baseline", None),
    ("oracle_trunc", "Real query", ex['query']),
    ("random_trunc", "~20w from unrelated passage",
     " ".join(samples[other_idx]['passage'].split()[:20])),
    ("scrambled_oracle_trunc", "Query words shuffled", scrambled),
    ("surr_template_trunc", "'What is [kw]?'", surr_template),
    ("static_fact_trunc", "Fixed question", "What are the key facts I need to know?"),
    ("repeat_the_trunc", "'the' x 10 (attention sink test)", " ".join(["the"] * 10)),
    ("single_word_trunc", "'X' (minimal prefix)", "X"),
]
show_conditions(conditions, ex['passage'])

print("MODEL GENERALITY TEST: Same conditions but on T5-XL (3B params).")
print("  T5-XL uses learned relative position bias (NOT RoPE).")
print("  Standard separated cross-attention (not merged like T5Gemma).")
print("  If structural mechanism replicates: it's a general enc-dec property.")
"""


def make_exp10_display():
    return r"""
show_sample(samples[0])

ex = samples[0]
query_kws = extract_keywords(ex['query'])
doc_kws = extract_keywords(ex['passage'])
query_kw = query_kws[0] if query_kws else "topic"
doc_kw = doc_kws[0] if doc_kws else "topic"

other_idx = (0 + 250) % len(samples)
rand_text = " ".join(samples[other_idx]['passage'].split()[:20])

print(f"  Query keyword: '{query_kw}'")
print(f"  Doc keyword:   '{doc_kw}'")
print()

print("SELECTIVE TRUNCATION: What if the decoder can see just ONE keyword from the prefix?")
print()

conditions = [
    ("bare", "Baseline", None),
    ("oracle_trunc", "Query, decoder sees doc only", ex['query']),
    ("oracle_full", "Query, decoder sees query+doc", ex['query']),
    ("oracle_kw_visible", f"Query, decoder sees doc + '{query_kw}' token", ex['query']),
    ("template_trunc", "'What is [kw]?', truncated", f"What is {doc_kw}?"),
    ("template_kw_visible", f"'What is [kw]?', decoder sees '{doc_kw}' token",
     f"What is {doc_kw}?"),
    ("pad_kw_trunc", f"'the the the {doc_kw}', truncated",
     f"the the the {doc_kw}"),
    ("pad_kw_kw_visible", f"'the the the {doc_kw}', decoder sees '{doc_kw}' token",
     f"the the the {doc_kw}"),
    ("random_trunc", "Random text, truncated", rand_text),
    ("keyword_only_visible", f"'{doc_kw}' as entire prefix, decoder sees it", doc_kw),
]
show_conditions(conditions, ex['passage'])

print("KEY QUESTION: Is there a sweet spot between full truncation and full visibility?")
print("  Full truncation (d=+0.408): decoder sees doc only")
print("  Full visibility (d=+0.345): decoder sees everything")
print("  Selective: decoder sees doc + just the keyword token")
"""


# ============================================================
# Build notebooks
# ============================================================

EXPERIMENTS = [
    ("01", "Truncation Test", make_exp01_display, True),   # uses MSMARCO std
    ("02", "Surrogate Type Sweep", make_exp02_display, True),
    ("02b", "Mechanism Decomposition", make_exp02b_display, True),
    ("03", "Length Scaling", make_exp03_display, True),
    ("03b", "Extended Length Scaling", make_exp03b_display, True),
    ("03d", "Cross-Dataset Ablation", make_exp03d_display, False),   # neural-bridge
    ("03e", "Attention Probing", make_exp03e_display, False),        # neural-bridge
    ("03f", "Semantic Amplification", make_exp03f_display, True),
    ("04a", "MS MARCO Ranking", make_exp04a_display, True),
    ("04b", "ESCI Ranking", make_exp04b_display, False),   # self-contained
    ("05", "LLM Surrogates", make_exp05_display, True),
    ("06", "Factoid Subsample", make_exp06_display, False),   # self-contained
    ("07", "RoPE Isolation", make_exp07_display, True),
    ("08", "Contrastive Ranking", make_exp08_display, True),
    ("09", "Model Generality (T5-XL)", make_exp09_display, True),
    ("10", "Selective Truncation", make_exp10_display, True),
]


for exp_id, title, display_fn, uses_msmarco_std in EXPERIMENTS:
    nb = nbf.v4.new_notebook()

    # Cell 1: Markdown header
    nb.cells.append(nbf.v4.new_markdown_cell(
        f"# Experiment {exp_id}: {title} — Condition Examples\n\n"
        f"This notebook shows the actual text for each experimental condition "
        f"using real data from the dataset. No GPU needed."
    ))

    # Cell 2: Data loading + condition display
    display_code = display_fn()

    if uses_msmarco_std:
        # Prepend MS MARCO loading + display helpers
        full_code = MSMARCO_LOAD_SEED42 + "\n" + DISPLAY_HEADER + "\n" + display_code
    else:
        # Self-contained — display_fn already includes its own loading + DISPLAY_HEADER
        full_code = display_code

    nb.cells.append(nbf.v4.new_code_cell(full_code))

    # Write notebook
    out_name = f"experiments/{exp_id}/{exp_id}_examples.ipynb"
    with open(out_name, 'w') as f:
        nbf.write(nb, f)
    print(f"  {out_name}")

print(f"\nDone: {len(EXPERIMENTS)} example notebooks generated.")
