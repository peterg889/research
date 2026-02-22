#!/usr/bin/env python3
# Build Prefix LM Exp 04h notebook: Token Stratification & Contrastive Evaluation.
#
# 04g showed all 5 mechanistic approaches to force prime info into doc values HURT.
# Average NLL says random ~ oracle. But average NLL weights all tokens equally.
# If semantic signal is concentrated on hard tokens, NLL averaging hides it.
# Also, NLL measures absolute quality, not discrimination (correct vs wrong answer).
#
# Two reframings:
#   1. Token stratification: split tokens by difficulty, check oracle vs random on hard
#   2. Contrastive evaluation: NLL gap between correct and wrong answers per condition
#
# 5 conditions, 2 answers (correct + wrong), N=500, ~18 forwards/sample.

import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 0: Markdown =====
md(r"""# Prefix LM Exp 04h: Token Stratification & Contrastive Evaluation

## Motivation

Exp 04g showed all 5 mechanistic approaches to force prime information into document
value vectors HURT performance. Under average NLL, random ~ oracle -- the structural
interpretation holds.

But average NLL weights all answer tokens equally. In "The capital of France is
**Paris**", the word "Paris" requires document understanding while "The", "of", "is"
are syntactically predictable. If the semantic signal (oracle vs random) is concentrated
on hard/informative tokens, NLL averaging hides it.

Additionally, NLL measures absolute prediction quality, not **discrimination** -- can the
model distinguish the correct answer from a wrong one? Enrichment might improve
discrimination without reducing average NLL.

## Two Reframings

**1. Token stratification**: Split answer tokens by difficulty (how surprising they are
under the bare condition). Check if oracle beats random specifically on hard tokens.

**2. Contrastive evaluation**: For each sample, evaluate BOTH a correct and a wrong
answer. Measure the NLL gap (wrong - correct). Does enrichment increase the model's
ability to prefer the correct answer?

## Conditions (5)

| # | Condition | Description |
|---|-----------|-------------|
| 1 | `no_doc` | No document -- `[BOS, query, answer]` single-pass causal. Model's prior. |
| 2 | `bare` | `[BOS, doc]` cached, Phase B `[query, answer]` |
| 3 | `random` | `[BOS, random_8w, doc]` cached, truncated Phase B |
| 4 | `oracle` | `[BOS, query, doc]` cached, truncated Phase B |
| 5 | `oracle_plus_vocab` | `[BOS, query+answer_vocab, doc]` cached, truncated Phase B |

## Wrong Answer Assignment

For each sample i, the wrong answer is the gold answer from sample `(i + 250) % 500`.
Deterministic, well-separated pairings.

## Key Analyses

- **A**: Token stratification by difficulty (bare NLL quartiles)
- **B**: Token stratification by document dependence (no_doc - bare quartiles)
- **C**: Content word vs function word effects
- **D**: Contrastive evaluation (NLL gap, AUC, prior-controlled discrimination)
- **E**: Contrastive x difficulty interaction
- **F**: Summary + replication of prior results""")


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
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, "../../..")
from lib.analysis import cohens_d

SEED = 42
N_SAMPLES = 500

MODEL_NAME = "google/gemma-3-12b-it"

RESULTS_DIR = Path("../../../results/prefix_lm_exp04h")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONDITIONS = ["no_doc", "bare", "random", "oracle", "oracle_plus_vocab"]
TWO_PASS_CONDITIONS = ["bare", "random", "oracle", "oracle_plus_vocab"]

print(f"Prefix LM Exp 04h: Token Stratification & Contrastive Evaluation")
print(f"N: {N_SAMPLES}, Conditions: {len(CONDITIONS)}")
print(f"DEVICE: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"\nConditions: {CONDITIONS}")
""")


# ===== Cell 2: Load model + tokenizer =====
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
code(r"""# Cell 3: Mask functions + sanity check

def make_causal_mask(n, dtype=torch.bfloat16):
    # Standard causal mask: lower triangle = 0, upper triangle = min_val
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((n, n), min_val, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, n, n)


def make_phase_b_mask(n_s, n_d, n_q, n_a, dtype=torch.bfloat16):
    # Phase B: continuation [query, answer] sees [BOS, doc] but NOT prime (truncation)
    n_prefix = 1 + n_s + n_d
    n_cont = n_q + n_a
    min_val = torch.finfo(dtype).min
    mask = torch.full((n_cont, n_prefix + n_cont), min_val, dtype=dtype)
    # Allow attending to all prefix positions
    mask[:, :n_prefix] = 0.0
    # Mask out prime positions (truncation)
    if n_s > 0:
        mask[:, 1:1 + n_s] = min_val
    # Causal mask for continuation tokens among themselves
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

del out_default, out_custom
gc.collect(); torch.cuda.empty_cache()
""")


# ===== Cell 4: Data loading + wrong-answer assignment =====
code(r"""# Cell 4: Load MS MARCO + prepare per-sample fields + wrong-answer pairing
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

    # Answer-doc overlap words (for oracle_plus_vocab)
    overlap_words = sorted(a_words & d_words)
    if not overlap_words:
        overlap_words = content_words(s['answer'])[:5]
    s['answer_vocab'] = " ".join(overlap_words[:10])
    s['oracle_plus_vocab'] = s['query'] + " " + s['answer_vocab']

    s['answer_wc'] = count_words(s['answer'])

# --- Wrong-answer pairing ---
for i in range(N_SAMPLES):
    j = (i + 250) % N_SAMPLES
    samples[i]['wrong_answer'] = samples[j]['answer']

print(f"\nLoaded {len(samples)} samples")
print(f"Mean passage words: {np.mean([s['word_count'] for s in samples]):.0f}")
print(f"Mean query words: {np.mean([count_words(s['query']) for s in samples]):.0f}")
print(f"Mean answer words: {np.mean([s['answer_wc'] for s in samples]):.0f}")

# Verify wrong-answer pairing
print(f"\n--- Wrong-answer examples ---")
for j in [0, 1, 250]:
    print(f"  Sample {j}: correct='{samples[j]['answer'][:50]}...'")
    print(f"             wrong='{samples[j]['wrong_answer'][:50]}...'")
    assert samples[j]['wrong_answer'] == samples[(j + 250) % N_SAMPLES]['answer']
print(f"  Pairing verified.")
""")


# ===== Cell 5: Scoring function =====
code(r"""# Cell 5: score_sample() -- 5 conditions x 2 answers, per-token NLLs

def score_sample(model, tokenizer, sample, device):
    passage = sample['passage']
    query = sample['query']
    correct_answer = sample['answer']
    wrong_answer = sample['wrong_answer']

    bos_id = tokenizer.bos_token_id

    doc_ids = tokenizer(passage, add_special_tokens=False, truncation=True,
                        max_length=1024).input_ids
    query_ids = tokenizer(query, add_special_tokens=False, truncation=True,
                          max_length=512).input_ids
    correct_answer_ids = tokenizer(correct_answer, add_special_tokens=False,
                                   truncation=True, max_length=256).input_ids
    wrong_answer_ids = tokenizer(wrong_answer, add_special_tokens=False,
                                 truncation=True, max_length=256).input_ids

    if len(correct_answer_ids) == 0 or len(wrong_answer_ids) == 0:
        return None

    # Prime variants
    oracle_ids = tokenizer(query, add_special_tokens=False, truncation=True,
                           max_length=256).input_ids
    random_ids = tokenizer(sample['random_prefix'],
                           add_special_tokens=False).input_ids
    oracle_plus_vocab_ids = tokenizer(sample['oracle_plus_vocab'],
                                      add_special_tokens=False,
                                      truncation=True, max_length=256).input_ids

    prime_map = {
        "bare": [],
        "random": random_ids,
        "oracle": oracle_ids,
        "oracle_plus_vocab": oracle_plus_vocab_ids,
    }

    n_q = len(query_ids)
    n_d = len(doc_ids)

    result = {
        'n_doc': n_d,
        'n_query': n_q,
        'correct_answer_ids': correct_answer_ids,
        'wrong_answer_ids': wrong_answer_ids,
    }

    answer_map = {
        'correct': correct_answer_ids,
        'wrong': wrong_answer_ids,
    }

    for answer_type, answer_ids in answer_map.items():
        n_a = len(answer_ids)
        targets = torch.tensor(answer_ids, dtype=torch.long, device=device)

        # --- no_doc: single-pass [BOS, query, answer] ---
        no_doc_tokens = [bos_id] + query_ids + answer_ids
        no_doc_input = torch.tensor([no_doc_tokens], dtype=torch.long,
                                    device=device)
        n_total = len(no_doc_tokens)

        no_doc_mask = make_causal_mask(n_total)
        no_doc_dict = make_mask_dict(no_doc_mask.to(device))
        no_doc_pos = torch.arange(n_total, device=device).unsqueeze(0)

        with torch.no_grad():
            out = model(input_ids=no_doc_input, attention_mask=no_doc_dict,
                        position_ids=no_doc_pos)

        # Logit at position n_q (last query token) predicts first answer token
        # Input: [BOS, q0..q_{nq-1}, a0..a_{na-1}]
        # Positions:  0    1..n_q      n_q+1..n_q+n_a
        answer_logits = out.logits[0, n_q : n_q + n_a, :]
        log_probs = F.log_softmax(answer_logits, dim=-1)
        token_nlls = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        result[f'token_nlls_no_doc_{answer_type}'] = token_nlls.cpu().tolist()
        result[f'nll_no_doc_{answer_type}'] = token_nlls.mean().item()

        del out, no_doc_input, no_doc_mask, no_doc_dict
        del answer_logits, log_probs, token_nlls

        # --- Two-pass conditions ---
        for cond_name in TWO_PASS_CONDITIONS:
            surr_ids = prime_map[cond_name]
            n_s = len(surr_ids)
            n_prefix = 1 + n_s + n_d

            # Phase A: cache [BOS, prime, doc]
            prefix_tokens = [bos_id] + surr_ids + doc_ids
            prefix_input = torch.tensor([prefix_tokens], dtype=torch.long,
                                        device=device)

            phase_a_mask = make_causal_mask(n_prefix)
            phase_a_dict = make_mask_dict(phase_a_mask.to(device))
            phase_a_pos = torch.arange(n_prefix, device=device).unsqueeze(0)

            with torch.no_grad():
                out_a = model(input_ids=prefix_input,
                              attention_mask=phase_a_dict,
                              position_ids=phase_a_pos, use_cache=True)
            past_kv = out_a.past_key_values

            # Phase B: evaluate [query, answer] with truncation
            cont_tokens = query_ids + answer_ids
            n_cont = len(cont_tokens)
            cont_input = torch.tensor([cont_tokens], dtype=torch.long,
                                      device=device)

            phase_b_mask = make_phase_b_mask(n_s, n_d, n_q, n_a)
            phase_b_dict = make_mask_dict(phase_b_mask.to(device))
            phase_b_pos = torch.arange(n_prefix, n_prefix + n_cont,
                                       device=device).unsqueeze(0)

            with torch.no_grad():
                out_b = model(input_ids=cont_input,
                              attention_mask=phase_b_dict,
                              position_ids=phase_b_pos,
                              past_key_values=past_kv)

            # Logit at Phase B position n_q-1 predicts first answer token
            answer_logits = out_b.logits[0, n_q - 1 : n_q + n_a - 1, :]
            log_probs = F.log_softmax(answer_logits, dim=-1)
            token_nlls = -log_probs.gather(
                1, targets.unsqueeze(1)).squeeze(1)

            result[f'token_nlls_{cond_name}_{answer_type}'] = \
                token_nlls.cpu().tolist()
            result[f'nll_{cond_name}_{answer_type}'] = \
                token_nlls.mean().item()

            del out_a, out_b, past_kv, prefix_input, cont_input
            del phase_a_mask, phase_a_dict, phase_b_mask, phase_b_dict
            del answer_logits, log_probs, token_nlls

    return result


# Forward count: no_doc x 2 = 2, two-pass x 4 x 2 = 16, total = 18
print(f"Scoring function defined.")
print(f"  {len(CONDITIONS)} conditions x 2 answers = {len(CONDITIONS)*2} NLL arrays/sample")
print(f"  18 forwards/sample")
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
    print(f"Starting fresh: {N_SAMPLES} samples x {len(CONDITIONS)} conds x 2 answers")

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
              desc="Scoring"):
    s = samples[i]
    try:
        result = score_sample(model, tokenizer, s, DEVICE)
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
            'results': all_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        CKPT_PATH.write_text(json.dumps(ckpt))

    if (i + 1) % 50 == 0:
        gc.collect()
        torch.cuda.empty_cache()

elapsed = time.time() - t0
print(f"\nDone: {len(all_results)} samples in {elapsed/60:.1f} min")

print(f"\nQuick summary (correct answer, mean NLL):")
for cn in CONDITIONS:
    vals = [r[f'nll_{cn}_correct'] for r in all_results]
    print(f"  {cn:<20} NLL={np.mean(vals):.4f}")

print(f"\nQuick summary (wrong answer, mean NLL):")
for cn in CONDITIONS:
    vals = [r[f'nll_{cn}_wrong'] for r in all_results]
    print(f"  {cn:<20} NLL={np.mean(vals):.4f}")
""")


# ===== Cell 7: Analysis A-C (token stratification) =====
code(r"""# Cell 7: Analysis A-C -- Token Stratification
print("=" * 70)
print("ANALYSIS A-C: TOKEN STRATIFICATION")
print("=" * 70)

N = len(all_results)

# --- Gather per-token NLLs for correct answers into flat arrays ---
tok_data = {cn: [] for cn in CONDITIONS}
tok_ids_flat = []
tok_sample_idx = []

for i, r in enumerate(all_results):
    n_a = len(r['correct_answer_ids'])
    for cn in CONDITIONS:
        tok_data[cn].extend(r[f'token_nlls_{cn}_correct'])
    tok_ids_flat.extend(r['correct_answer_ids'])
    tok_sample_idx.extend([i] * n_a)

for cn in CONDITIONS:
    tok_data[cn] = np.array(tok_data[cn])
tok_ids_flat = np.array(tok_ids_flat)
tok_sample_idx = np.array(tok_sample_idx)

n_tokens_total = len(tok_ids_flat)
print(f"Total correct-answer tokens pooled: {n_tokens_total}")
print(f"Mean tokens per sample: {n_tokens_total / N:.1f}")

# ============================================================
# A. Token Stratification by Difficulty
# ============================================================
print(f"\n--- A. Token Stratification by Difficulty (bare NLL quartiles) ---\n")

# Difficulty = bare_correct per-token NLL
difficulty = tok_data['bare']
q25, q50, q75 = np.percentile(difficulty, [25, 50, 75])
quartile = np.digitize(difficulty, bins=[q25, q50, q75])  # 0=easy, 3=hard

print(f"  Quartile thresholds: Q1={q25:.3f}, Q2={q50:.3f}, Q3={q75:.3f}")
print(f"  Quartile sizes: {[(quartile == q).sum() for q in range(4)]}")

q_labels = ["Q1 (easy)", "Q2", "Q3", "Q4 (hard)"]
print(f"\n  {'Quartile':<12} {'N':>6} {'bare':>8} {'no_doc':>8} {'random':>8}"
      f" {'oracle':>8} {'opv':>8} {'d(o-r)':>8} {'d(o-b)':>8} {'d(opv-r)':>9}")
print(f"  {'-'*96}")

for q in range(4):
    mask = quartile == q
    n_tok = mask.sum()
    bare_m = tok_data['bare'][mask].mean()
    nodoc_m = tok_data['no_doc'][mask].mean()
    rand_m = tok_data['random'][mask].mean()
    orac_m = tok_data['oracle'][mask].mean()
    opv_m = tok_data['oracle_plus_vocab'][mask].mean()
    d_or = cohens_d(tok_data['random'][mask] - tok_data['oracle'][mask])
    d_ob = cohens_d(tok_data['bare'][mask] - tok_data['oracle'][mask])
    d_opvr = cohens_d(tok_data['random'][mask] -
                      tok_data['oracle_plus_vocab'][mask])
    print(f"  {q_labels[q]:<12} {n_tok:>6} {bare_m:>8.3f} {nodoc_m:>8.3f}"
          f" {rand_m:>8.3f} {orac_m:>8.3f} {opv_m:>8.3f}"
          f" {d_or:>+8.3f} {d_ob:>+8.3f} {d_opvr:>+9.3f}")

d_or_q1 = cohens_d(tok_data['random'][quartile == 0] -
                    tok_data['oracle'][quartile == 0])
d_or_q4 = cohens_d(tok_data['random'][quartile == 3] -
                    tok_data['oracle'][quartile == 3])
print(f"\n  d(oracle-random): Q1(easy)={d_or_q1:+.3f}, Q4(hard)={d_or_q4:+.3f}")
print(f"  Gradient (Q4-Q1): {d_or_q4 - d_or_q1:+.3f}")
if d_or_q4 > d_or_q1 + 0.05:
    print(f"  => Semantic signal IS concentrated on hard tokens!")
else:
    print(f"  => Semantic signal NOT concentrated on hard tokens.")

# ============================================================
# B. Token Stratification by Document Dependence
# ============================================================
print(f"\n--- B. Token Stratification by Document Dependence ---\n")

# Doc dependence = no_doc - bare per token (positive = doc helps)
doc_dep = tok_data['no_doc'] - tok_data['bare']
dq25, dq50, dq75 = np.percentile(doc_dep, [25, 50, 75])
dep_quartile = np.digitize(doc_dep, bins=[dq25, dq50, dq75])

print(f"  Doc-dependence thresholds: Q1={dq25:.3f}, Q2={dq50:.3f}, Q3={dq75:.3f}")
print(f"  (positive = document helps reduce NLL)")

dep_labels = ["Q1 (doc-indep)", "Q2", "Q3", "Q4 (doc-dep)"]
print(f"\n  {'Quartile':<16} {'N':>6} {'dep':>8} {'d(o-r)':>8}"
      f" {'d(o-b)':>8} {'d(opv-r)':>9}")
print(f"  {'-'*62}")

for q in range(4):
    mask = dep_quartile == q
    n_tok = mask.sum()
    dep_m = doc_dep[mask].mean()
    d_or = cohens_d(tok_data['random'][mask] - tok_data['oracle'][mask])
    d_ob = cohens_d(tok_data['bare'][mask] - tok_data['oracle'][mask])
    d_opvr = cohens_d(tok_data['random'][mask] -
                      tok_data['oracle_plus_vocab'][mask])
    print(f"  {dep_labels[q]:<16} {n_tok:>6} {dep_m:>+8.3f} {d_or:>+8.3f}"
          f" {d_ob:>+8.3f} {d_opvr:>+9.3f}")

d_or_dep1 = cohens_d(tok_data['random'][dep_quartile == 0] -
                      tok_data['oracle'][dep_quartile == 0])
d_or_dep4 = cohens_d(tok_data['random'][dep_quartile == 3] -
                      tok_data['oracle'][dep_quartile == 3])
print(f"\n  d(oracle-random): doc-indep={d_or_dep1:+.3f}, doc-dep={d_or_dep4:+.3f}")
if d_or_dep4 > d_or_dep1 + 0.05:
    print(f"  => Semantic signal appears on document-dependent tokens!")
else:
    print(f"  => Semantic signal does NOT concentrate on doc-dependent tokens.")

# ============================================================
# C. Content Word vs Function Word
# ============================================================
print(f"\n--- C. Content Word vs Function Word ---\n")

is_content = np.zeros(n_tokens_total, dtype=bool)
for idx in range(n_tokens_total):
    word = tokenizer.decode([int(tok_ids_flat[idx])]).strip().lower()
    word_clean = re.sub(r'[^\w]', '', word)
    if word_clean and word_clean not in STOP_WORDS and len(word_clean) > 2:
        is_content[idx] = True

n_content = is_content.sum()
n_function = (~is_content).sum()
print(f"  Content tokens: {n_content} ({100*n_content/n_tokens_total:.1f}%)")
print(f"  Function tokens: {n_function} ({100*n_function/n_tokens_total:.1f}%)")

print(f"\n  {'Type':<12} {'N':>6} {'bare':>8} {'random':>8} {'oracle':>8}"
      f" {'d(o-r)':>8} {'d(o-b)':>8}")
print(f"  {'-'*58}")

for label, mask in [("Content", is_content), ("Function", ~is_content)]:
    bare_m = tok_data['bare'][mask].mean()
    rand_m = tok_data['random'][mask].mean()
    orac_m = tok_data['oracle'][mask].mean()
    d_or = cohens_d(tok_data['random'][mask] - tok_data['oracle'][mask])
    d_ob = cohens_d(tok_data['bare'][mask] - tok_data['oracle'][mask])
    print(f"  {label:<12} {mask.sum():>6} {bare_m:>8.3f} {rand_m:>8.3f}"
          f" {orac_m:>8.3f} {d_or:>+8.3f} {d_ob:>+8.3f}")

# Cross-tabulate: content x difficulty
print(f"\n  Content x Difficulty (d oracle-random):")
print(f"  {'':>16} {'Content':>10} {'Function':>10}")
print(f"  {'-'*38}")
for q in range(4):
    q_mask = quartile == q
    d_content = cohens_d(tok_data['random'][q_mask & is_content] -
                         tok_data['oracle'][q_mask & is_content])
    d_function = cohens_d(tok_data['random'][q_mask & ~is_content] -
                          tok_data['oracle'][q_mask & ~is_content])
    print(f"  {q_labels[q]:<16} {d_content:>+10.3f} {d_function:>+10.3f}")
""")


# ===== Cell 8: Analysis D-E (contrastive) =====
code(r"""# Cell 8: Analysis D-E -- Contrastive Evaluation
print("=" * 70)
print("ANALYSIS D-E: CONTRASTIVE EVALUATION")
print("=" * 70)

# --- Gather per-sample mean NLLs ---
nll_correct = {}
nll_wrong = {}
for cn in CONDITIONS:
    nll_correct[cn] = np.array([r[f'nll_{cn}_correct'] for r in all_results])
    nll_wrong[cn] = np.array([r[f'nll_{cn}_wrong'] for r in all_results])

# ============================================================
# D. Contrastive Evaluation
# ============================================================
print(f"\n--- D. Contrastive Evaluation ({N} samples) ---\n")

# Gap = wrong_NLL - correct_NLL (positive = model prefers correct)
gap = {}
for cn in CONDITIONS:
    gap[cn] = nll_wrong[cn] - nll_correct[cn]

print(f"  {'Condition':<20} {'mean_gap':>10} {'std_gap':>10} {'AUC':>8}"
      f" {'d(gap)':>8} {'p(gap>0)':>12} {'sig':>5}")
print(f"  {'-'*78}")

for cn in CONDITIONS:
    g = gap[cn]
    auc = (g > 0).mean()
    d_gap = cohens_d(g)
    _, p_gap = stats.ttest_1samp(g, 0)
    sig = ('***' if p_gap < 0.001 else '**' if p_gap < 0.01
           else '*' if p_gap < 0.05 else 'ns')
    print(f"  {cn:<20} {g.mean():>+10.3f} {g.std():>10.3f} {auc:>8.1%}"
          f" {d_gap:>+8.3f} {p_gap:>12.2e} {sig:>5}")

# Prior-controlled discrimination
print(f"\n  Prior-controlled discrimination (gap - gap_no_doc):\n")
print(f"  {'Condition':<20} {'mean':>10} {'d':>8} {'p':>12} {'sig':>5}")
print(f"  {'-'*58}")

for cn in CONDITIONS:
    if cn == "no_doc":
        continue
    doc_discrim = gap[cn] - gap['no_doc']
    d_dd = cohens_d(doc_discrim)
    _, p_dd = stats.ttest_1samp(doc_discrim, 0)
    sig = ('***' if p_dd < 0.001 else '**' if p_dd < 0.01
           else '*' if p_dd < 0.05 else 'ns')
    print(f"  {cn:<20} {doc_discrim.mean():>+10.3f} {d_dd:>+8.3f}"
          f" {p_dd:>12.2e} {sig:>5}")

# Oracle vs random discrimination
diff_discrim = gap['oracle'] - gap['random']
d_discrim = cohens_d(diff_discrim)
_, p_discrim = stats.ttest_1samp(diff_discrim, 0)
sig_d = ('***' if p_discrim < 0.001 else '**' if p_discrim < 0.01
         else '*' if p_discrim < 0.05 else 'ns')
print(f"\n  Oracle vs random discrimination:")
print(f"    gap(oracle) - gap(random): mean={diff_discrim.mean():+.3f},"
      f" d={d_discrim:+.3f}, p={p_discrim:.2e} {sig_d}")
if p_discrim < 0.05 and d_discrim > 0:
    print(f"    => Oracle enrichment DOES increase discrimination over random!")
else:
    print(f"    => Oracle enrichment does NOT increase discrimination over random.")

# ============================================================
# E. Contrastive x Difficulty Interaction
# ============================================================
print(f"\n--- E. Contrastive x Difficulty Interaction ---\n")

# Split samples by difficulty (mean bare_correct NLL)
sample_difficulty = nll_correct['bare']
sq25, sq50, sq75 = np.percentile(sample_difficulty, [25, 50, 75])
sample_quartile = np.digitize(sample_difficulty, bins=[sq25, sq50, sq75])

print(f"  Sample difficulty quartiles (bare correct NLL):")
print(f"    Q1: <{sq25:.3f}, Q2: {sq25:.3f}-{sq50:.3f},"
      f" Q3: {sq50:.3f}-{sq75:.3f}, Q4: >{sq75:.3f}")

s_labels = ["Q1 (easy)", "Q2", "Q3", "Q4 (hard)"]
print(f"\n  {'Quartile':<12} {'N':>5} {'AUC_bare':>10} {'AUC_rand':>10}"
      f" {'AUC_orac':>10} {'gap_orac':>10} {'gap_rand':>10} {'d(o-r)':>8}")
print(f"  {'-'*80}")

for q in range(4):
    mask = sample_quartile == q
    n_sq = mask.sum()
    auc_bare = (gap['bare'][mask] > 0).mean()
    auc_rand = (gap['random'][mask] > 0).mean()
    auc_orac = (gap['oracle'][mask] > 0).mean()
    gap_orac = gap['oracle'][mask].mean()
    gap_rand = gap['random'][mask].mean()
    diff = gap['oracle'][mask] - gap['random'][mask]
    d_or = cohens_d(diff) if len(diff) > 1 else 0.0
    print(f"  {s_labels[q]:<12} {n_sq:>5} {auc_bare:>10.1%} {auc_rand:>10.1%}"
          f" {auc_orac:>10.1%} {gap_orac:>+10.3f} {gap_rand:>+10.3f}"
          f" {d_or:>+8.3f}")

# Per-quartile significance
print(f"\n  Per-quartile: oracle vs random discrimination (gap difference):")
for q in range(4):
    mask = sample_quartile == q
    diff = gap['oracle'][mask] - gap['random'][mask]
    d_or = cohens_d(diff) if len(diff) > 1 else 0.0
    _, p = stats.ttest_1samp(diff, 0) if len(diff) > 1 else (None, 1.0)
    sig = ('***' if p < 0.001 else '**' if p < 0.01
           else '*' if p < 0.05 else 'ns')
    print(f"    {s_labels[q]}: d={d_or:+.3f}, p={p:.2e} {sig}")

# Also check: correct-only NLL by difficulty quartile
print(f"\n  Correct-answer NLL effect by difficulty quartile (d vs bare):")
print(f"  {'Quartile':<12} {'d_oracle':>10} {'d_random':>10} {'d(o-r)':>10}")
print(f"  {'-'*46}")
for q in range(4):
    mask = sample_quartile == q
    d_orc = cohens_d((nll_correct['bare'] - nll_correct['oracle'])[mask])
    d_rnd = cohens_d((nll_correct['bare'] - nll_correct['random'])[mask])
    d_or = cohens_d((nll_correct['random'] - nll_correct['oracle'])[mask])
    print(f"  {s_labels[q]:<12} {d_orc:>+10.3f} {d_rnd:>+10.3f} {d_or:>+10.3f}")
""")


# ===== Cell 9: Analysis F (summary, replication, save) =====
code(r"""# Cell 9: Analysis F -- Summary, Replication, Save
print("=" * 70)
print("ANALYSIS F: SUMMARY & REPLICATION")
print("=" * 70)

nll_c = nll_correct  # shorthand

# --- Replication of prior results using correct-answer NLLs ---
d_oracle = cohens_d(nll_c['bare'] - nll_c['oracle'])
d_random = cohens_d(nll_c['bare'] - nll_c['random'])
struct_frac = d_random / d_oracle if d_oracle != 0 else float('nan')

print(f"\n  Replication (correct answer NLLs, d vs bare):")
print(f"    d_oracle = {d_oracle:+.3f} (expected ~+0.452)")
print(f"    d_random = {d_random:+.3f} (expected ~+0.475)")
print(f"    structural fraction = {struct_frac:.0%} (expected ~105%)")

# Oracle vs random
diff_or = nll_c['random'] - nll_c['oracle']
d_or = cohens_d(diff_or)
_, p_or = stats.ttest_1samp(diff_or, 0)
sig_or = ('***' if p_or < 0.001 else '**' if p_or < 0.01
          else '*' if p_or < 0.05 else 'ns')
print(f"    d(oracle vs random) = {d_or:+.3f} (p={p_or:.2e}) {sig_or}")

# oracle_plus_vocab vs random
diff_opv = nll_c['random'] - nll_c['oracle_plus_vocab']
d_opv = cohens_d(diff_opv)
_, p_opv = stats.ttest_1samp(diff_opv, 0)
sig_opv = ('***' if p_opv < 0.001 else '**' if p_opv < 0.01
           else '*' if p_opv < 0.05 else 'ns')
print(f"    d(oracle_plus_vocab vs random) = {d_opv:+.3f}"
      f" (expected ~+0.311, p={p_opv:.2e}) {sig_opv}")

# oracle_plus_vocab vs oracle
diff_opvo = nll_c['oracle'] - nll_c['oracle_plus_vocab']
d_opvo = cohens_d(diff_opvo)
_, p_opvo = stats.ttest_1samp(diff_opvo, 0)
sig_opvo = ('***' if p_opvo < 0.001 else '**' if p_opvo < 0.01
            else '*' if p_opvo < 0.05 else 'ns')
print(f"    d(oracle_plus_vocab vs oracle) = {d_opvo:+.3f}"
      f" (p={p_opvo:.2e}) {sig_opvo}")

# --- Sanity checks ---
print(f"\n  Sanity checks:")
print(f"    Contrastive: AUC(bare) = {(gap['bare'] > 0).mean():.1%}"
      f" (should be > 50%)")
print(f"    Prior: AUC(no_doc) = {(gap['no_doc'] > 0).mean():.1%}")

# Token count check
tok_ok = True
for cn in CONDITIONS:
    for at in ['correct', 'wrong']:
        lengths = [len(r[f'token_nlls_{cn}_{at}']) for r in all_results]
        answer_key = f'{at}_answer_ids'
        expected = [len(r[answer_key]) for r in all_results]
        if not all(l == e for l, e in zip(lengths, expected)):
            print(f"    WARNING: token count mismatch for {cn}_{at}!")
            tok_ok = False
if tok_ok:
    print(f"    Token counts: all verified OK")

# --- Overall verdict ---
print(f"\n  --- VERDICT ---")

# Token stratification
d_or_easy = cohens_d(tok_data['random'][quartile == 0] -
                     tok_data['oracle'][quartile == 0])
d_or_hard = cohens_d(tok_data['random'][quartile == 3] -
                     tok_data['oracle'][quartile == 3])
if d_or_hard > d_or_easy + 0.05:
    print(f"  TOKEN STRATIFICATION: Semantic signal concentrated on hard tokens")
    print(f"    d(oracle-random): easy={d_or_easy:+.3f}, hard={d_or_hard:+.3f}")
else:
    print(f"  TOKEN STRATIFICATION: No concentration on hard tokens")
    print(f"    d(oracle-random): easy={d_or_easy:+.3f}, hard={d_or_hard:+.3f}")

# Contrastive
if p_discrim < 0.05 and d_discrim > 0:
    print(f"  CONTRASTIVE: Oracle enrichment increases discrimination"
          f" (d={d_discrim:+.3f})")
else:
    print(f"  CONTRASTIVE: Oracle enrichment does NOT increase discrimination"
          f" (d={d_discrim:+.3f})")

# --- Save results ---
summary = {
    'n_samples': N,
    'model': MODEL_NAME,
    'd_oracle_vs_bare': float(d_oracle),
    'd_random_vs_bare': float(d_random),
    'structural_fraction': float(struct_frac),
    'd_opv_vs_random': float(d_opv),
    'd_oracle_vs_random_correct': float(d_or),
    'd_discrimination_oracle_vs_random': float(d_discrim),
}
for cn in CONDITIONS:
    summary[f'nll_{cn}_correct'] = float(nll_c[cn].mean())
    summary[f'nll_{cn}_wrong'] = float(nll_wrong[cn].mean())
    summary[f'auc_{cn}'] = float((gap[cn] > 0).mean())

final_results = {
    'experiment': 'prefix_lm_exp04h',
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
out_path = "experiments/prefix_lm/04/04h_token_contrastive.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
