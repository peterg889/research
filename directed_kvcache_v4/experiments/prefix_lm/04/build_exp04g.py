#!/usr/bin/env python3
# Build Prefix LM Exp 04g notebook: Value Storage Enhancement.
#
# Exp 04f showed: oracle creates measurably different attention patterns (d up to
# +1.34), but the signal doesn't survive truncation -- attention differences don't
# predict NLL benefit (r=+0.031, ns). Doc tokens attend only ~2-5% to prime, so
# little prime info gets stored in doc VALUE vectors. After truncation, only what's
# baked into doc KVs persists.
#
# GOAL: Force more prime information into doc value vectors during Phase A.
# Test 5 mechanistic approaches:
#   1. Attention logit boost (positive mask values force more doc->prime attention)
#   2. Bidirectional bridge (prime tokens can see doc tokens)
#   3. Chat format (IT formatting activates instruction-following circuits)
#   4. Interspersed repetition (prime copies between doc chunks)
#   5. Value injection (directly add prime V content to doc V vectors post-Phase-A)
#
# 17 conditions, N=500, selective attention probing on 7 conditions.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 0: Markdown =====
md(r"""# Prefix LM Exp 04g: Value Storage Enhancement

## Motivation

Exp 04f confirmed: oracle DOES create measurably different attention patterns than
random during Phase A (d up to +1.34 at peak layers, significant in all 13/13 probed
layers), but the semantic signal doesn't survive truncation -- the oracle-vs-random
attention difference doesn't predict oracle-vs-random NLL benefit (r=+0.031, ns).

**Root cause**: doc tokens attend only ~2-5% to the prime, so little prime information
gets stored in doc VALUE vectors. After truncation cuts prime KV positions from Phase B,
only what's baked into doc KVs persists.

**Goal**: Force more prime information into document value vectors during Phase A so
it survives truncation into Phase B.

## Five Mechanistic Approaches

| # | Approach | Mechanism | Key question |
|---|----------|-----------|-------------|
| 1 | Attention logit boost | Positive mask values at doc->prime positions | Does forcing more attention store more info in doc Vs? |
| 2 | Bidirectional bridge | Prime tokens can see doc tokens (upper triangle unmasked) | Does multi-layer cross-information help? |
| 3 | Chat format | IT turn markers activate instruction-following circuits | Does the model process content more deeply in chat mode? |
| 4 | Interspersed repetition | Prime copies between doc chunks | Does proximity to fresh prime copies help? |
| 5 | Value injection | Directly add prime V content to doc V vectors post-Phase-A | Is V content the actual bottleneck? |

## Conditions (17)

| # | Name | Category | Description |
|---|------|----------|-------------|
| 1 | `bare` | control | No prime |
| 2 | `random` | control | 8 random words |
| 3 | `oracle` | control | Real query |
| 4 | `oracle_boost2` | boost | Oracle, Phase A mask doc->prime = +2.0 |
| 5 | `oracle_boost4` | boost | Oracle, Phase A mask doc->prime = +4.0 |
| 6 | `oracle_boost8` | boost | Oracle, Phase A mask doc->prime = +8.0 |
| 7 | `random_boost4` | boost | Random, Phase A mask doc->prime = +4.0 |
| 8 | `oracle_bidir` | bidir | Oracle, prime->doc bidirectional in Phase A |
| 9 | `oracle_chat` | chat | Oracle in chat turn markers |
| 10 | `instr_chat` | chat | Instruction in chat turn markers |
| 11 | `oracle_inter` | interspersed | [BOS, oracle, doc1, oracle, doc2, oracle, doc3] |
| 12 | `random_inter` | interspersed | [BOS, random, doc1, random, doc2, random, doc3] |
| 13 | `oracle_vinj01` | vinject | Oracle + post-Phase-A value inject beta=0.1 |
| 14 | `oracle_vinj05` | vinject | Oracle + post-Phase-A value inject beta=0.5 |
| 15 | `oracle_vinj10` | vinject | Oracle + post-Phase-A value inject beta=1.0 |
| 16 | `pointer` | reference | "the answer is about [keywords]" (04e best vs random) |
| 17 | `oracle_plus_vocab` | reference | Query + answer-doc overlap words (04e overall best) |

## Key Analyses

- **A**: Full ranking (all 17 by mean NLL, d vs bare/random/oracle)
- **B**: Per-approach comparison (dose-response for boost and vinj; paired for bidir, chat, inter)
- **C**: Best-of-each approach vs oracle
- **D**: Attention verification (frac_prime monotonic with boost; correlation with NLL)
- **E**: Structural fraction per approach
- **F**: Per-sample heterogeneity (correlation with answer_wc, doc_wc, overlap)
- **G**: 04e replication check (pointer, oracle_plus_vocab)""")


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

RESULTS_DIR = Path("../../../results/prefix_lm_exp04g")
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
    "oracle_boost2",
    "oracle_boost4",
    "oracle_boost8",
    "random_boost4",
    "oracle_bidir",
    "oracle_chat",
    "instr_chat",
    "oracle_inter",
    "random_inter",
    "oracle_vinj01",
    "oracle_vinj05",
    "oracle_vinj10",
    "pointer",
    "oracle_plus_vocab",
]

CONDITION_CONFIG = {
    "bare":             {"prime": "bare",             "boost": 0, "bidir": False, "inter": False, "vinj": 0},
    "random":           {"prime": "random",           "boost": 0, "bidir": False, "inter": False, "vinj": 0},
    "oracle":           {"prime": "oracle",           "boost": 0, "bidir": False, "inter": False, "vinj": 0},
    "oracle_boost2":    {"prime": "oracle",           "boost": 2, "bidir": False, "inter": False, "vinj": 0},
    "oracle_boost4":    {"prime": "oracle",           "boost": 4, "bidir": False, "inter": False, "vinj": 0},
    "oracle_boost8":    {"prime": "oracle",           "boost": 8, "bidir": False, "inter": False, "vinj": 0},
    "random_boost4":    {"prime": "random",           "boost": 4, "bidir": False, "inter": False, "vinj": 0},
    "oracle_bidir":     {"prime": "oracle",           "boost": 0, "bidir": True,  "inter": False, "vinj": 0},
    "oracle_chat":      {"prime": "oracle_chat",      "boost": 0, "bidir": False, "inter": False, "vinj": 0},
    "instr_chat":       {"prime": "instr_chat",       "boost": 0, "bidir": False, "inter": False, "vinj": 0},
    "oracle_inter":     {"prime": "oracle",           "boost": 0, "bidir": False, "inter": True,  "vinj": 0},
    "random_inter":     {"prime": "random",           "boost": 0, "bidir": False, "inter": True,  "vinj": 0},
    "oracle_vinj01":    {"prime": "oracle",           "boost": 0, "bidir": False, "inter": False, "vinj": 0.1},
    "oracle_vinj05":    {"prime": "oracle",           "boost": 0, "bidir": False, "inter": False, "vinj": 0.5},
    "oracle_vinj10":    {"prime": "oracle",           "boost": 0, "bidir": False, "inter": False, "vinj": 1.0},
    "pointer":          {"prime": "pointer",          "boost": 0, "bidir": False, "inter": False, "vinj": 0},
    "oracle_plus_vocab":{"prime": "oracle_plus_vocab","boost": 0, "bidir": False, "inter": False, "vinj": 0},
}

# Attention probing: control + boost conditions at 4 late layers
ATTN_CONDITIONS = {"bare", "random", "oracle",
                   "oracle_boost2", "oracle_boost4", "oracle_boost8",
                   "random_boost4"}
PROBE_LAYERS = [36, 40, 44, 47]

# Chat token IDs (set after tokenizer loads in Cell 2)
CHAT_PREFIX_IDS = None
CHAT_SUFFIX_IDS = None
INSTR_CHAT_IDS = None

print(f"Prefix LM Exp 04g: Value Storage Enhancement")
print(f"N: {N_SAMPLES}, Conditions: {len(CONDITIONS)}")
print(f"Attention probing: {len(ATTN_CONDITIONS)} conditions x {len(PROBE_LAYERS)} layers")
print(f"DEVICE: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"\nConditions: {CONDITIONS}")
""")


# ===== Cell 2: Load model + verify =====
code(r"""# Cell 2: Load model + tokenizer, verify DynamicCache API + chat tokens
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

print(f"transformers version: {transformers.__version__}")

print(f"Loading {MODEL_NAME}...")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="eager",
    token=HF_TOKEN,
)
model.eval()

n_params = sum(p.numel() for p in model.parameters()) / 1e9
gpu_mem = torch.cuda.memory_allocated() / 1e9
print(f"Loaded: {n_params:.1f}B params, {gpu_mem:.1f} GB GPU, {time.time()-t0:.0f}s")

n_layers = model.config.text_config.num_hidden_layers
print(f"Model has {n_layers} layers")
print(f"BOS token id: {tokenizer.bos_token_id}")

# Verify PROBE_LAYERS are valid
PROBE_LAYERS = [l for l in PROBE_LAYERS if l < n_layers]
if (n_layers - 1) not in PROBE_LAYERS:
    PROBE_LAYERS.append(n_layers - 1)
PROBE_LAYERS = sorted(set(PROBE_LAYERS))
print(f"Probe layers: {PROBE_LAYERS}")

# --- Verify DynamicCache API ---
print(f"\n--- DynamicCache verification ---")
test_text = "The quick brown fox."
test_ids = tokenizer(test_text, return_tensors="pt",
                     add_special_tokens=True).input_ids.to(DEVICE)
with torch.no_grad():
    out = model(input_ids=test_ids, use_cache=True)
past_kv = out.past_key_values
print(f"  Cache type: {type(past_kv).__name__}")
print(f"  n_layers in cache: {len(past_kv.layers)}")
print(f"  Key shape [0]: {past_kv.layers[0].keys.shape}")
print(f"  Value shape [0]: {past_kv.layers[0].values.shape}")
del out, past_kv
gc.collect(); torch.cuda.empty_cache()

# --- Verify chat token handling ---
print(f"\n--- Chat token verification ---")
sot_id = tokenizer.convert_tokens_to_ids("<start_of_turn>")
eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
print(f"  <start_of_turn> ID: {sot_id}")
print(f"  <end_of_turn> ID: {eot_id}")

CHAT_PREFIX_IDS = tokenizer("<start_of_turn>user\n", add_special_tokens=False).input_ids
CHAT_SUFFIX_IDS = tokenizer("<end_of_turn>\n", add_special_tokens=False).input_ids
print(f"  Chat prefix IDs: {CHAT_PREFIX_IDS}")
print(f"  Chat suffix IDs: {CHAT_SUFFIX_IDS}")

instr_text = "Identify the key facts in this passage:"
instr_raw_ids = tokenizer(instr_text, add_special_tokens=False).input_ids
INSTR_CHAT_IDS = CHAT_PREFIX_IDS + instr_raw_ids + CHAT_SUFFIX_IDS
print(f"  Instr chat IDs ({len(INSTR_CHAT_IDS)} tokens): {INSTR_CHAT_IDS[:5]}...{INSTR_CHAT_IDS[-3:]}")

# Round-trip verify
test_chat = "<start_of_turn>user\nhello world<end_of_turn>\n"
test_chat_ids = tokenizer(test_chat, add_special_tokens=False).input_ids
assert test_chat_ids[0] == sot_id, f"Expected SOT at pos 0, got {test_chat_ids[0]}"
assert test_chat_ids[-2] == eot_id, f"Expected EOT at pos -2, got {test_chat_ids[-2]}"
print(f"  Chat tokenization verified.")
""")


# ===== Cell 3: Mask functions + helpers + sanity check =====
code(r"""# Cell 3: Mask functions + helpers + sanity check

# --- Base masks ---

def make_causal_mask(n, dtype=torch.bfloat16):
    # Standard causal mask: lower triangle = 0, upper triangle = min_val
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((n, n), min_val, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, n, n)


def make_phase_b_mask_general(n_prefix, prime_positions, n_q, n_a,
                              dtype=torch.bfloat16):
    # Phase B mask: continuation tokens see all prefix EXCEPT prime positions
    n_cont = n_q + n_a
    min_val = torch.finfo(dtype).min
    mask = torch.full((n_cont, n_prefix + n_cont), min_val, dtype=dtype)
    # Allow attending to all prefix positions
    mask[:, :n_prefix] = 0.0
    # Mask out prime positions (truncation)
    for p in prime_positions:
        mask[:, p] = min_val
    # Causal mask for continuation tokens among themselves
    mask[:, n_prefix:] = torch.triu(
        torch.full((n_cont, n_cont), min_val, dtype=dtype), diagonal=1
    )
    return mask.unsqueeze(0).unsqueeze(0)


def make_mask_dict(mask_4d):
    return {"full_attention": mask_4d, "sliding_attention": mask_4d}


# --- Approach helpers ---

def apply_boost(mask, doc_positions, prime_positions, boost):
    # Set doc->prime attention positions to +boost (positive boosts pre-softmax logits)
    # Works for standard layout where prime and doc are contiguous blocks
    dp_start = doc_positions[0]
    dp_end = doc_positions[-1] + 1
    pp_start = prime_positions[0]
    pp_end = prime_positions[-1] + 1
    mask[0, 0, dp_start:dp_end, pp_start:pp_end] = boost


def apply_bidir(mask, prime_positions, doc_positions):
    # Allow prime tokens to attend to doc tokens (breaks causal for prime->doc only)
    pp_start = prime_positions[0]
    pp_end = prime_positions[-1] + 1
    dp_start = doc_positions[0]
    dp_end = doc_positions[-1] + 1
    mask[0, 0, pp_start:pp_end, dp_start:dp_end] = 0.0


def make_interspersed_sequence(bos_id, prime_ids, doc_ids, n_chunks=3):
    # Split doc into n_chunks, insert prime copy before each chunk
    # Returns: (sequence, prime_positions, doc_positions)
    if not prime_ids:
        return ([bos_id] + doc_ids, [], list(range(1, 1 + len(doc_ids))))

    chunk_size = max(1, len(doc_ids) // n_chunks)
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < n_chunks - 1 else len(doc_ids)
        chunks.append(doc_ids[start:end])

    seq = [bos_id]
    prime_positions = []
    doc_positions = []

    for chunk in chunks:
        # Add prime copy
        p_start = len(seq)
        seq.extend(prime_ids)
        prime_positions.extend(range(p_start, p_start + len(prime_ids)))
        # Add doc chunk
        d_start = len(seq)
        seq.extend(chunk)
        doc_positions.extend(range(d_start, d_start + len(chunk)))

    return seq, prime_positions, doc_positions


def inject_values(past_kv, prime_positions, doc_positions, beta):
    # Add scaled mean of prime V vectors to doc V vectors (in-place)
    n_layers = len(past_kv.layers)
    dp_start = doc_positions[0]
    dp_end = doc_positions[-1] + 1
    for layer_idx in range(n_layers):
        v = past_kv.layers[layer_idx].values  # (1, n_heads, seq_len, head_dim)
        prime_v = v[:, :, prime_positions, :]  # (1, n_heads, n_prime, head_dim)
        mean_v = prime_v.mean(dim=2, keepdim=True)  # (1, n_heads, 1, head_dim)
        v[:, :, dp_start:dp_end, :] += beta * mean_v


# --- Sanity check: custom causal mask vs default forward ---
print("Mask sanity check: custom causal mask vs default forward...")
test_text = "The quick brown fox jumps over the lazy dog."
test_ids = tokenizer(test_text, return_tensors="pt",
                     add_special_tokens=True).input_ids.to(DEVICE)
Lt = test_ids.shape[1]

with torch.no_grad():
    out_default = model(input_ids=test_ids)

causal_mask = make_causal_mask(Lt)
causal_dict = make_mask_dict(causal_mask.to(DEVICE))
causal_pos = torch.arange(Lt, device=DEVICE).unsqueeze(0)

with torch.no_grad():
    out_custom = model(input_ids=test_ids, attention_mask=causal_dict,
                       position_ids=causal_pos)

max_diff = (out_default.logits - out_custom.logits).abs().max().item()
print(f"  Max logit diff: {max_diff:.6f}")
assert max_diff < 0.1, f"FAIL: max_diff={max_diff:.4f}"
print(f"  PASS: Dict-based mask API verified.")

# Quick test of interspersed helper
seq, pp, dp = make_interspersed_sequence(2, [10, 11, 12], [20, 21, 22, 23, 24, 25], 3)
print(f"\n  Interspersed test: seq={seq}")
print(f"    prime_positions={pp}")
print(f"    doc_positions={dp}")
assert len(seq) == 1 + 3*3 + 6, f"Wrong length: {len(seq)}"
print(f"  PASS: Interspersed sequence verified.")

del out_default, out_custom
gc.collect(); torch.cuda.empty_cache()
print("All helpers verified.")
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

    # Query-doc overlap
    q_words = set(content_words(s['query']))
    d_words = set(content_words(s['passage']))
    a_words = set(content_words(s['answer']))
    s['query_doc_overlap'] = jaccard(q_words, d_words)

    # Answer-doc overlap words (for pointer + oracle_plus_vocab)
    overlap_words = sorted(a_words & d_words)
    if not overlap_words:
        overlap_words = content_words(s['answer'])[:5]

    # Pointer instruction
    kw = overlap_words[:5] if overlap_words else content_words(s['query'])[:3]
    s['pointer'] = "the answer is about " + " ".join(kw)

    # Oracle + vocabulary
    s['answer_vocab'] = " ".join(overlap_words[:10])
    s['oracle_plus_vocab'] = s['query'] + " " + s['answer_vocab']

    s['answer_wc'] = count_words(s['answer'])

print(f"\nLoaded {len(samples)} samples")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Mean query words: {np.mean([count_words(s['query']) for s in samples]):.0f}")
print(f"Mean answer words: {np.mean([s['answer_wc'] for s in samples]):.0f}")

print(f"\n--- Examples ---")
for j in range(3):
    print(f"\n  Sample {j}:")
    print(f"    Q: {samples[j]['query'][:70]}")
    print(f"    A: {samples[j]['answer'][:70]}")
    print(f"    pointer: {samples[j]['pointer']}")
    print(f"    oracle_plus_vocab: {samples[j]['oracle_plus_vocab'][:70]}")
""")


# ===== Cell 5: Scoring function =====
code(r"""# Cell 5: score_sample() -- 17 conditions, 5 approaches, selective attention probing

def score_sample(model, tokenizer, sample, device, probe_layers):
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
    pointer_ids = tokenizer(sample['pointer'], add_special_tokens=False).input_ids
    oracle_plus_vocab_ids = tokenizer(sample['oracle_plus_vocab'],
                                      add_special_tokens=False,
                                      truncation=True, max_length=256).input_ids

    # Chat-formatted primes
    oracle_chat_ids = CHAT_PREFIX_IDS + oracle_ids + CHAT_SUFFIX_IDS
    # INSTR_CHAT_IDS is a global constant (same for all samples)

    prime_map = {
        "bare": [],
        "random": random_ids,
        "oracle": oracle_ids,
        "oracle_chat": oracle_chat_ids,
        "instr_chat": INSTR_CHAT_IDS,
        "pointer": pointer_ids,
        "oracle_plus_vocab": oracle_plus_vocab_ids,
    }

    n_q = len(query_ids)
    n_a = len(answer_ids)
    n_d = len(doc_ids)

    targets = torch.tensor(answer_ids, dtype=torch.long, device=device)
    result = {'n_doc': n_d, 'n_query': n_q}

    for cond_name in CONDITIONS:
        cfg = CONDITION_CONFIG[cond_name]
        surr_ids = prime_map[cfg["prime"]]

        # --- Build Phase A sequence ---
        if cfg["inter"]:
            seq, prime_positions, doc_positions = make_interspersed_sequence(
                bos_id, surr_ids, doc_ids, n_chunks=3)
        else:
            seq = [bos_id] + surr_ids + doc_ids
            prime_positions = list(range(1, 1 + len(surr_ids)))
            doc_positions = list(range(1 + len(surr_ids), len(seq)))

        n_prefix = len(seq)
        prefix_input = torch.tensor([seq], dtype=torch.long, device=device)

        # Record prime token count
        result[f'n_prime_{cond_name}'] = len(prime_positions)

        # --- Build Phase A mask ---
        phase_a_mask = make_causal_mask(n_prefix)

        if cfg["boost"] > 0 and len(prime_positions) > 0:
            apply_boost(phase_a_mask, doc_positions, prime_positions, cfg["boost"])
        if cfg["bidir"] and len(prime_positions) > 0:
            apply_bidir(phase_a_mask, prime_positions, doc_positions)

        phase_a_dict = make_mask_dict(phase_a_mask.to(device))
        phase_a_pos = torch.arange(n_prefix, device=device).unsqueeze(0)

        # --- Phase A forward ---
        extract_attn = (cond_name in ATTN_CONDITIONS)
        with torch.no_grad():
            out_a = model(input_ids=prefix_input,
                          attention_mask=phase_a_dict,
                          position_ids=phase_a_pos,
                          use_cache=True,
                          output_attentions=extract_attn)
        past_kv = out_a.past_key_values

        # --- Extract attention stats if probing ---
        if extract_attn and out_a.attentions is not None:
            attentions = out_a.attentions
            for layer_idx in probe_layers:
                if layer_idx >= len(attentions):
                    continue
                attn = attentions[layer_idx][0]  # (n_heads, n_prefix, n_prefix)

                if n_d == 0:
                    continue

                # Doc-token attention patterns
                doc_attn = attn[:, doc_positions, :]  # (n_heads, n_d, n_prefix)

                frac_bos = doc_attn[:, :, 0].mean().item()

                if len(prime_positions) > 0:
                    frac_prime = doc_attn[:, :, prime_positions].sum(dim=-1).mean().item()
                else:
                    frac_prime = 0.0

                frac_doc = doc_attn[:, :, doc_positions].sum(dim=-1).mean().item()

                # Attention entropy
                eps = 1e-10
                ent = -(doc_attn * (doc_attn + eps).log()).sum(dim=-1).mean().item()

                result[f'{cond_name}_L{layer_idx}_frac_bos'] = frac_bos
                result[f'{cond_name}_L{layer_idx}_frac_prime'] = frac_prime
                result[f'{cond_name}_L{layer_idx}_frac_doc'] = frac_doc
                result[f'{cond_name}_L{layer_idx}_entropy'] = ent

            del attentions

        # --- Value injection (post Phase A, pre Phase B) ---
        if cfg["vinj"] > 0 and len(prime_positions) > 0:
            inject_values(past_kv, prime_positions, doc_positions, cfg["vinj"])

        # --- Phase B (truncation: mask prime positions) ---
        cont_tokens = query_ids + answer_ids
        n_cont = len(cont_tokens)
        cont_input = torch.tensor([cont_tokens], dtype=torch.long, device=device)

        phase_b_mask = make_phase_b_mask_general(n_prefix, prime_positions, n_q, n_a)
        phase_b_dict = make_mask_dict(phase_b_mask.to(device))
        phase_b_pos = torch.arange(n_prefix, n_prefix + n_cont,
                                    device=device).unsqueeze(0)

        with torch.no_grad():
            out_b = model(input_ids=cont_input,
                          attention_mask=phase_b_dict,
                          position_ids=phase_b_pos,
                          past_key_values=past_kv)

        # Compute answer NLL
        answer_logits = out_b.logits[0, n_q - 1 : n_q + n_a - 1, :]
        log_probs = F.log_softmax(answer_logits, dim=-1)
        token_nlls = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        result[f'nll_{cond_name}'] = token_nlls.mean().item()

        # Cleanup
        del out_a, out_b, past_kv, prefix_input, cont_input
        del phase_a_mask, phase_a_dict, phase_b_mask, phase_b_dict
        del answer_logits, log_probs, token_nlls

    return result


print(f"Scoring function defined ({len(CONDITIONS)} conditions per sample).")
print(f"Attention probing on: {sorted(ATTN_CONDITIONS)}")
print(f"Probe layers: {PROBE_LAYERS}")
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
        result = score_sample(model, tokenizer, s, DEVICE, PROBE_LAYERS)
    except Exception as e:
        print(f"ERROR at sample {i}: {e}")
        import traceback; traceback.print_exc()
        result = None

    if result is None:
        continue

    result['query'] = s['query'][:50]
    result['query_doc_overlap'] = s['query_doc_overlap']
    result['answer_wc'] = s['answer_wc']
    result['query_wc'] = _cw(s['query'])
    result['doc_wc'] = s['word_count']
    all_results.append(result)

    if (i + 1) % 25 == 0 or i == N_SAMPLES - 1:
        ckpt = {
            'model': MODEL_NAME,
            'n_total': N_SAMPLES,
            'n_conditions': len(CONDITIONS),
            'condition_names': CONDITIONS,
            'probe_layers': PROBE_LAYERS,
            'results': all_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        CKPT_PATH.write_text(json.dumps(ckpt))

    if (i + 1) % 50 == 0:
        gc.collect()
        torch.cuda.empty_cache()

elapsed = time.time() - t0
print(f"\nDone: {len(all_results)} samples in {elapsed/60:.1f} min")
print(f"\nQuick summary:")
for cn in CONDITIONS:
    vals = [r[f'nll_{cn}'] for r in all_results]
    print(f"  {cn:<20} NLL={np.mean(vals):.4f}")
""")


# ===== Cell 7: Analysis A-D =====
code(r"""# Cell 7: Analysis A-D
print("=" * 70)
print("RESULTS: VALUE STORAGE ENHANCEMENT")
print("=" * 70)

nll = {}
for cn in CONDITIONS:
    nll[cn] = np.array([r[f'nll_{cn}'] for r in all_results])

N = len(all_results)

# ============================================================
# A. Full ranking
# ============================================================
print(f"\n--- A. Full Ranking ({N} samples) ---\n")
print(f"  {'Condition':<20} {'Mean NLL':>10} {'d vs bare':>10} {'d vs rand':>10}"
      f" {'d vs orac':>10} {'p vs rand':>12} {'sig':>5}")
print(f"  {'-'*82}")

ranked = sorted(CONDITIONS, key=lambda cn: nll[cn].mean())
for cn in ranked:
    d_base = cohens_d(nll['bare'] - nll[cn]) if cn != "bare" else 0.0
    diff_rand = nll['random'] - nll[cn]
    d_rand = cohens_d(diff_rand) if cn != "random" else 0.0
    diff_orac = nll['oracle'] - nll[cn]
    d_orac = cohens_d(diff_orac) if cn != "oracle" else 0.0

    if cn == "bare":
        p_rand = 1.0
    elif cn == "random":
        p_rand = 1.0
    else:
        _, p_rand = stats.ttest_1samp(diff_rand, 0)

    sig = '***' if p_rand < 0.001 else '**' if p_rand < 0.01 else '*' if p_rand < 0.05 else 'ns'
    print(f"  {cn:<20} {nll[cn].mean():>10.4f} {d_base:>+10.3f} {d_rand:>+10.3f}"
          f" {d_orac:>+10.3f} {p_rand:>12.2e} {sig:>5}")

# ============================================================
# B. Per-approach analysis
# ============================================================

# --- B1. Boost dose-response ---
print(f"\n--- B1. Boost Dose-Response ---\n")
print(f"  {'Condition':<20} {'Mean NLL':>10} {'d vs bare':>10} {'d vs oracle':>12}"
      f" {'p vs oracle':>12} {'sig':>5}")
print(f"  {'-'*72}")

boost_conds = ["oracle", "oracle_boost2", "oracle_boost4", "oracle_boost8"]
for cn in boost_conds:
    d_base = cohens_d(nll['bare'] - nll[cn])
    diff_orac = nll['oracle'] - nll[cn]
    d_orac = cohens_d(diff_orac) if cn != "oracle" else 0.0
    _, p_orac = stats.ttest_1samp(diff_orac, 0) if cn != "oracle" else (None, 1.0)
    sig = '***' if p_orac < 0.001 else '**' if p_orac < 0.01 else '*' if p_orac < 0.05 else 'ns'
    boost_val = CONDITION_CONFIG[cn]["boost"]
    print(f"  {cn:<20} {nll[cn].mean():>10.4f} {d_base:>+10.3f} {d_orac:>+12.3f}"
          f" {p_orac:>12.2e} {sig:>5}")

# Random boost comparison
diff_rb = nll['random'] - nll['random_boost4']
d_rb = cohens_d(diff_rb)
_, p_rb = stats.ttest_1samp(diff_rb, 0)
sig_rb = '***' if p_rb < 0.001 else '**' if p_rb < 0.01 else '*' if p_rb < 0.05 else 'ns'
print(f"\n  random_boost4 vs random: d={d_rb:+.3f}, p={p_rb:.2e} {sig_rb}")
print(f"  (Does boosting help structural primes too?)")

# --- B2. Bidirectional bridge ---
print(f"\n--- B2. Bidirectional Bridge ---\n")
diff_bd = nll['oracle'] - nll['oracle_bidir']
d_bd = cohens_d(diff_bd)
_, p_bd = stats.ttest_1samp(diff_bd, 0)
sig_bd = '***' if p_bd < 0.001 else '**' if p_bd < 0.01 else '*' if p_bd < 0.05 else 'ns'
win_bd = (diff_bd > 0).mean() * 100
print(f"  oracle_bidir vs oracle: d={d_bd:+.3f}, p={p_bd:.2e} {sig_bd}, win%={win_bd:.1f}%")

# --- B3. Chat format ---
print(f"\n--- B3. Chat Format ---\n")
print(f"  {'Comparison':<40} {'d':>8} {'p':>12} {'sig':>5} {'win%':>7}")
print(f"  {'-'*75}")

for cn in ["oracle_chat", "instr_chat"]:
    diff = nll['oracle'] - nll[cn]
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    win = (diff > 0).mean() * 100
    print(f"  {cn + ' vs oracle':<40} {d:>+8.3f} {p:>12.2e} {sig:>5} {win:>6.1f}%")

# --- B4. Interspersed repetition ---
print(f"\n--- B4. Interspersed Repetition ---\n")
print(f"  {'Comparison':<40} {'d':>8} {'p':>12} {'sig':>5} {'win%':>7}")
print(f"  {'-'*75}")

for base, inter in [("oracle", "oracle_inter"), ("random", "random_inter")]:
    diff = nll[base] - nll[inter]
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    win = (diff > 0).mean() * 100
    print(f"  {inter + ' vs ' + base:<40} {d:>+8.3f} {p:>12.2e} {sig:>5} {win:>6.1f}%")

# --- B5. Value injection dose-response ---
print(f"\n--- B5. Value Injection Dose-Response ---\n")
print(f"  {'Condition':<20} {'Mean NLL':>10} {'d vs bare':>10} {'d vs oracle':>12}"
      f" {'p vs oracle':>12} {'sig':>5}")
print(f"  {'-'*72}")

vinj_conds = ["oracle", "oracle_vinj01", "oracle_vinj05", "oracle_vinj10"]
for cn in vinj_conds:
    d_base = cohens_d(nll['bare'] - nll[cn])
    diff_orac = nll['oracle'] - nll[cn]
    d_orac = cohens_d(diff_orac) if cn != "oracle" else 0.0
    _, p_orac = stats.ttest_1samp(diff_orac, 0) if cn != "oracle" else (None, 1.0)
    sig = '***' if p_orac < 0.001 else '**' if p_orac < 0.01 else '*' if p_orac < 0.05 else 'ns'
    print(f"  {cn:<20} {nll[cn].mean():>10.4f} {d_base:>+10.3f} {d_orac:>+12.3f}"
          f" {p_orac:>12.2e} {sig:>5}")

# ============================================================
# C. Best-of-each approach vs oracle
# ============================================================
print(f"\n--- C. Best-of-Each Approach vs Oracle ---\n")

APPROACH_GROUPS = {
    "boost": ["oracle_boost2", "oracle_boost4", "oracle_boost8"],
    "bidir": ["oracle_bidir"],
    "chat": ["oracle_chat", "instr_chat"],
    "interspersed": ["oracle_inter", "random_inter"],
    "vinject": ["oracle_vinj01", "oracle_vinj05", "oracle_vinj10"],
    "reference": ["pointer", "oracle_plus_vocab"],
}

print(f"  {'Approach':<15} {'Best cond':<20} {'Mean NLL':>10} {'d vs oracle':>12}"
      f" {'p vs oracle':>12} {'sig':>5}")
print(f"  {'-'*78}")

for approach, conds in APPROACH_GROUPS.items():
    best_cn = min(conds, key=lambda cn: nll[cn].mean())
    d_base = cohens_d(nll['bare'] - nll[best_cn])
    diff_orac = nll['oracle'] - nll[best_cn]
    d_orac = cohens_d(diff_orac)
    _, p_orac = stats.ttest_1samp(diff_orac, 0)
    sig = '***' if p_orac < 0.001 else '**' if p_orac < 0.01 else '*' if p_orac < 0.05 else 'ns'
    print(f"  {approach:<15} {best_cn:<20} {nll[best_cn].mean():>10.4f} {d_orac:>+12.3f}"
          f" {p_orac:>12.2e} {sig:>5}")

# ============================================================
# D. Attention verification (boost conditions only)
# ============================================================
print(f"\n--- D. Attention Verification ---\n")

# D1: frac_prime by condition and layer
attn_conds_ordered = ["bare", "random", "oracle", "oracle_boost2",
                      "oracle_boost4", "oracle_boost8", "random_boost4"]
print(f"  Doc-to-prime attention fraction (frac_prime):\n")
header = f"  {'Layer':>6}"
for cn in attn_conds_ordered:
    short = cn.replace("oracle_", "o_").replace("random_", "r_")
    header += f" {short:>10}"
print(header)
print(f"  {'-'*(8 + 11*len(attn_conds_ordered))}")

layer_attn_data = {}
for layer_idx in PROBE_LAYERS:
    row = f"  L{layer_idx:>4}"
    layer_attn_data[layer_idx] = {}
    for cn in attn_conds_ordered:
        key = f'{cn}_L{layer_idx}_frac_prime'
        if key in all_results[0]:
            vals = np.array([r[key] for r in all_results])
            layer_attn_data[layer_idx][cn] = vals
            row += f" {vals.mean():>10.4f}"
        else:
            row += f" {'N/A':>10}"
    print(row)

# D2: Entropy by condition and layer
print(f"\n  Attention entropy:\n")
header = f"  {'Layer':>6}"
for cn in attn_conds_ordered:
    short = cn.replace("oracle_", "o_").replace("random_", "r_")
    header += f" {short:>10}"
print(header)
print(f"  {'-'*(8 + 11*len(attn_conds_ordered))}")

for layer_idx in PROBE_LAYERS:
    row = f"  L{layer_idx:>4}"
    for cn in attn_conds_ordered:
        key = f'{cn}_L{layer_idx}_entropy'
        if key in all_results[0]:
            vals = np.array([r[key] for r in all_results])
            row += f" {vals.mean():>10.3f}"
        else:
            row += f" {'N/A':>10}"
    print(row)

# D3: Is frac_prime monotonic with boost?
print(f"\n  Monotonicity check (last probed layer L{PROBE_LAYERS[-1]}):")
last_l = PROBE_LAYERS[-1]
for cn in ["oracle", "oracle_boost2", "oracle_boost4", "oracle_boost8"]:
    key = f'{cn}_L{last_l}_frac_prime'
    if key in all_results[0]:
        vals = np.array([r[key] for r in all_results])
        boost_val = CONDITION_CONFIG[cn]["boost"]
        print(f"    boost={boost_val}: frac_prime={vals.mean():.4f}")

# D4: Correlation between boost-induced prime attention and NLL benefit
print(f"\n  Correlation: frac_prime x NLL benefit (vs bare):\n")
print(f"  {'Condition':<20} {'r':>8} {'p':>12} {'sig':>5}")
print(f"  {'-'*50}")

for cn in attn_conds_ordered:
    if cn == "bare":
        continue
    key = f'{cn}_L{last_l}_frac_prime'
    if key not in all_results[0]:
        continue
    frac = np.array([r[key] for r in all_results])
    benefit = nll['bare'] - nll[cn]
    r_val, p_val = stats.pearsonr(frac, benefit)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"  {cn:<20} {r_val:>+8.3f} {p_val:>12.2e} {sig:>5}")
""")


# ===== Cell 8: Analysis E-G =====
code(r"""# Cell 8: Analysis E-G

# ============================================================
# E. Structural fraction per approach
# ============================================================
print(f"\n--- E. Structural Fraction ---\n")

d_oracle_base = cohens_d(nll['bare'] - nll['oracle'])
d_random_base = cohens_d(nll['bare'] - nll['random'])
struct_base = d_random_base / d_oracle_base if d_oracle_base != 0 else float('nan')

print(f"  Baseline: d_oracle={d_oracle_base:+.3f}, d_random={d_random_base:+.3f},"
      f" structural={struct_base:.0%}")

# Approaches with both oracle and random variants
print(f"\n  {'Approach':<15} {'d_oracle_var':>13} {'d_random_var':>13} {'struct%':>8}")
print(f"  {'-'*55}")

struct_pairs = [
    ("standard", "oracle", "random"),
    ("boost4", "oracle_boost4", "random_boost4"),
    ("interspersed", "oracle_inter", "random_inter"),
]
for label, orc_cn, rnd_cn in struct_pairs:
    d_orc = cohens_d(nll['bare'] - nll[orc_cn])
    d_rnd = cohens_d(nll['bare'] - nll[rnd_cn])
    sf = d_rnd / d_orc if d_orc != 0 else float('nan')
    print(f"  {label:<15} {d_orc:>+13.3f} {d_rnd:>+13.3f} {sf:>7.0%}")

# All conditions: d vs bare
print(f"\n  All conditions d vs bare:")
print(f"  {'Condition':<20} {'d vs bare':>10}")
print(f"  {'-'*33}")
for cn in ranked:
    d = cohens_d(nll['bare'] - nll[cn]) if cn != "bare" else 0.0
    print(f"  {cn:<20} {d:>+10.3f}")

# ============================================================
# F. Per-sample heterogeneity
# ============================================================
print(f"\n--- F. Per-Sample Heterogeneity ---\n")

answer_wc = np.array([r['answer_wc'] for r in all_results])
doc_wc = np.array([r['doc_wc'] for r in all_results])
qd_overlap = np.array([r['query_doc_overlap'] for r in all_results])

# For each approach category, pick the best non-control condition
# and correlate its benefit (vs bare) with sample features
approach_best = {}
for approach, conds in APPROACH_GROUPS.items():
    best_cn = min(conds, key=lambda cn: nll[cn].mean())
    approach_best[approach] = best_cn

print(f"  Correlation of NLL benefit (vs bare) with sample features:\n")
print(f"  {'Condition':<20} {'r(ans_wc)':>10} {'r(doc_wc)':>10} {'r(overlap)':>10}")
print(f"  {'-'*55}")

for label in ["oracle", "random"] + list(approach_best.values()):
    if label in ("oracle", "random"):
        cn = label
    else:
        cn = label
    benefit = nll['bare'] - nll[cn]
    r_awc, _ = stats.pearsonr(benefit, answer_wc)
    r_dwc, _ = stats.pearsonr(benefit, doc_wc)
    r_ov, _ = stats.pearsonr(benefit, qd_overlap)
    print(f"  {cn:<20} {r_awc:>+10.3f} {r_dwc:>+10.3f} {r_ov:>+10.3f}")

# Difficulty split: short vs long answers
print(f"\n  Split by answer length:")
short = answer_wc <= 5
long_a = answer_wc > 5

print(f"  {'Condition':<20} {'Short d':>10} {'Long d':>10}")
print(f"  {'-'*43}")
for cn in ranked:
    if cn == "bare":
        continue
    ds = cohens_d((nll['bare'] - nll[cn])[short])
    dl = cohens_d((nll['bare'] - nll[cn])[long_a])
    print(f"  {cn:<20} {ds:>+10.3f} {dl:>+10.3f}")

# ============================================================
# G. 04e Replication Check
# ============================================================
print(f"\n--- G. 04e Replication Check ---\n")
print(f"  Expected from 04e: pointer d~+0.250 vs random, oracle_plus_vocab d~+0.311 vs random\n")

for cn in ["pointer", "oracle_plus_vocab"]:
    diff = nll['random'] - nll[cn]
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {cn:<20} d vs random = {d:+.3f} (p={p:.2e}) {sig}")

# Prime token counts
print(f"\n  Prime token counts (mean):")
for cn in CONDITIONS:
    if cn == "bare":
        continue
    vals = [r[f'n_prime_{cn}'] for r in all_results]
    print(f"    {cn:<20} {np.mean(vals):>6.1f} tokens")
""")


# ===== Cell 9: Save results + verdict =====
code(r"""# Cell 9: Save results + verdict
print("=" * 70)
print("VERDICT -- Prefix LM Exp 04g: Value Storage Enhancement")
print("=" * 70)

d_oracle_v_bare = cohens_d(nll['bare'] - nll['oracle'])
d_random_v_bare = cohens_d(nll['bare'] - nll['random'])
struct_frac = d_random_v_bare / d_oracle_v_bare if d_oracle_v_bare != 0 else float('nan')

print(f"\n  Baseline replication:")
print(f"    d_oracle={d_oracle_v_bare:+.3f}, d_random={d_random_v_bare:+.3f}, structural={struct_frac:.0%}")

# --- Per-approach verdicts ---
print(f"\n  Per-approach verdicts:\n")

# Boost
best_boost = min(["oracle_boost2", "oracle_boost4", "oracle_boost8"],
                 key=lambda cn: nll[cn].mean())
diff_bb = nll['oracle'] - nll[best_boost]
d_bb = cohens_d(diff_bb)
_, p_bb = stats.ttest_1samp(diff_bb, 0)
if p_bb < 0.05 and d_bb > 0:
    print(f"  BOOST: HELPS. Best={best_boost}, d vs oracle={d_bb:+.3f} (p={p_bb:.2e})")
elif p_bb < 0.05 and d_bb < 0:
    print(f"  BOOST: HURTS. Best={best_boost}, d vs oracle={d_bb:+.3f} (p={p_bb:.2e})")
else:
    print(f"  BOOST: NO EFFECT. Best={best_boost}, d vs oracle={d_bb:+.3f} (p={p_bb:.2e})")

# Random boost
diff_rb = nll['random'] - nll['random_boost4']
d_rb = cohens_d(diff_rb)
_, p_rb = stats.ttest_1samp(diff_rb, 0)
if p_rb < 0.05 and d_rb > 0:
    print(f"  RANDOM BOOST: HELPS structural too. d={d_rb:+.3f} (p={p_rb:.2e})")
else:
    print(f"  RANDOM BOOST: No effect on structural. d={d_rb:+.3f} (p={p_rb:.2e})")

# Bidir
diff_bd = nll['oracle'] - nll['oracle_bidir']
d_bd = cohens_d(diff_bd)
_, p_bd = stats.ttest_1samp(diff_bd, 0)
if p_bd < 0.05 and d_bd > 0:
    print(f"  BIDIR: HELPS. d vs oracle={d_bd:+.3f} (p={p_bd:.2e})")
elif p_bd < 0.05 and d_bd < 0:
    print(f"  BIDIR: HURTS. d vs oracle={d_bd:+.3f} (p={p_bd:.2e})")
else:
    print(f"  BIDIR: NO EFFECT. d vs oracle={d_bd:+.3f} (p={p_bd:.2e})")

# Chat
best_chat = min(["oracle_chat", "instr_chat"], key=lambda cn: nll[cn].mean())
diff_ch = nll['oracle'] - nll[best_chat]
d_ch = cohens_d(diff_ch)
_, p_ch = stats.ttest_1samp(diff_ch, 0)
if p_ch < 0.05 and d_ch > 0:
    print(f"  CHAT: HELPS. Best={best_chat}, d vs oracle={d_ch:+.3f} (p={p_ch:.2e})")
elif p_ch < 0.05 and d_ch < 0:
    print(f"  CHAT: HURTS. Best={best_chat}, d vs oracle={d_ch:+.3f} (p={p_ch:.2e})")
else:
    print(f"  CHAT: NO EFFECT. Best={best_chat}, d vs oracle={d_ch:+.3f} (p={p_ch:.2e})")

# Interspersed
diff_io = nll['oracle'] - nll['oracle_inter']
d_io = cohens_d(diff_io)
_, p_io = stats.ttest_1samp(diff_io, 0)
if p_io < 0.05 and d_io > 0:
    print(f"  INTERSPERSED: HELPS. oracle_inter vs oracle d={d_io:+.3f} (p={p_io:.2e})")
elif p_io < 0.05 and d_io < 0:
    print(f"  INTERSPERSED: HURTS. oracle_inter vs oracle d={d_io:+.3f} (p={p_io:.2e})")
else:
    print(f"  INTERSPERSED: NO EFFECT. oracle_inter vs oracle d={d_io:+.3f} (p={p_io:.2e})")

diff_ir = nll['random'] - nll['random_inter']
d_ir = cohens_d(diff_ir)
_, p_ir = stats.ttest_1samp(diff_ir, 0)
print(f"    random_inter vs random: d={d_ir:+.3f} (p={p_ir:.2e})")

# Value injection
best_vinj = min(["oracle_vinj01", "oracle_vinj05", "oracle_vinj10"],
                key=lambda cn: nll[cn].mean())
diff_vj = nll['oracle'] - nll[best_vinj]
d_vj = cohens_d(diff_vj)
_, p_vj = stats.ttest_1samp(diff_vj, 0)
if p_vj < 0.05 and d_vj > 0:
    print(f"  VINJECT: HELPS. Best={best_vinj}, d vs oracle={d_vj:+.3f} (p={p_vj:.2e})")
elif p_vj < 0.05 and d_vj < 0:
    print(f"  VINJECT: HURTS. Best={best_vinj}, d vs oracle={d_vj:+.3f} (p={p_vj:.2e})")
else:
    print(f"  VINJECT: NO EFFECT. Best={best_vinj}, d vs oracle={d_vj:+.3f} (p={p_vj:.2e})")

# --- Overall verdict ---
print(f"\n  --- OVERALL ---")

# Does ANY approach beat oracle?
any_beats_oracle = False
best_overall = None
best_d_vs_oracle = -999
for approach, conds in APPROACH_GROUPS.items():
    for cn in conds:
        diff = nll['oracle'] - nll[cn]
        d = cohens_d(diff)
        _, p = stats.ttest_1samp(diff, 0)
        if d > 0 and p < 0.05:
            any_beats_oracle = True
        if d > best_d_vs_oracle:
            best_d_vs_oracle = d
            best_overall = cn

if any_beats_oracle:
    print(f"  YES: At least one approach significantly beats oracle.")
    print(f"  Best overall: {best_overall} (d vs oracle = {best_d_vs_oracle:+.3f})")
else:
    print(f"  NO: No approach significantly beats oracle.")
    print(f"  Closest: {best_overall} (d vs oracle = {best_d_vs_oracle:+.3f})")

# Does anything beat random?
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
    print(f"  NOTHING beats random — mechanism is purely structural even with forced value storage.")
else:
    print(f"  Some conditions beat random (semantic signal can be amplified).")

# --- Save results ---
summary = {'n_samples': N, 'model': MODEL_NAME}
for cn in CONDITIONS:
    summary[f'nll_{cn}'] = float(nll[cn].mean())
    summary[f'd_vs_bare_{cn}'] = float(cohens_d(nll['bare'] - nll[cn])) if cn != "bare" else 0.0
summary['d_structural'] = float(d_random_v_bare)
summary['d_oracle'] = float(d_oracle_v_bare)
summary['structural_fraction'] = float(struct_frac)

final_results = {
    'experiment': 'prefix_lm_exp04g',
    'dataset': 'ms_marco_v1.1',
    'model': MODEL_NAME,
    'n_samples': N,
    'seed': SEED,
    'conditions': CONDITIONS,
    'condition_config': {k: {kk: vv for kk, vv in v.items()}
                        for k, v in CONDITION_CONFIG.items()},
    'probe_layers': PROBE_LAYERS,
    'attn_conditions': sorted(ATTN_CONDITIONS),
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'summary': summary,
}

with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")
print("Done!")
""")


# ===== Write notebook =====
out_path = "experiments/prefix_lm/04/04g_value_storage.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
