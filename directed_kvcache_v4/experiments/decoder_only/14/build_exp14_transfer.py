#!/usr/bin/env python3
"""Build Exp 14 transfer evaluation notebook.

Generates 14_transfer_eval.ipynb — loads model, runs 8x7 cross-dataset evaluation
(7 per-dataset rand soft prompts + 1 universal rand), saves transfer_matrix.json.

Usage:
    cd /home/jupyter/research/directed_kvcache_v4
    python3 experiments/decoder_only/14/build_exp14_transfer.py
    cd experiments/decoder_only/14
    papermill 14_transfer_eval.ipynb 14_transfer_eval_executed.ipynb --no-progress-bar
"""

import os
import ast
import nbformat as nbf

os.makedirs("experiments/decoder_only/14", exist_ok=True)

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3',
}


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise SyntaxError(f"Cell has syntax error at line {e.lineno}: {e.msg}\n"
                          f"  {e.text}") from e
    nb.cells.append(nbf.v4.new_code_cell(source))


# =====================================================================
# Cell 0: Title
# =====================================================================
md(r"""# Experiment 14: Cross-Dataset Transfer Evaluation

**Question**: How well do soft prompts trained on one dataset generalize to others?

## Design

- 7 per-dataset `rand` soft prompts (best init from Exp 14)
- 1 universal `rand` soft prompt
- Each evaluated on all 7 datasets' 160 hard samples
- Produces 8x7 = 56 evaluation cells
- Compare each (source, target) pair vs bare using Cohen's d

## Output

`results/decoder_only/exp14/transfer_matrix.json` — full transfer results.""")


# =====================================================================
# Cell 1: Setup + Model + Functions
# =====================================================================
code(r"""# Cell 1: Setup, model loading, and evaluation functions
import os
os.umask(0o000)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys, json, time, gc, copy
import random as pyrandom
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, "../../..")
from lib.analysis import cohens_d, win_rate, paired_ttest
from lib.data import count_words
from lib.cache import deep_copy_cache, make_prefix, scramble_prefix
from lib.rope import build_layer_inv_freqs, get_layer_types, select_kv_cache, reposition_kv_cache
from lib.quantization import norm_roundtrip_kv_cache

SEED = 42
N_SAMPLES = 400
HARD_FRAC = 0.40
N_HARD = int(N_SAMPLES * HARD_FRAC)  # 160
PREFIX_L = 64
MODEL_NAME = "google/gemma-3-12b-it"

RESULTS_DIR = Path("../../../results/decoder_only/exp14")
EXP02_DIR = Path("../../../results/decoder_only/exp02")
EXP03_DIR = Path("../../../results/decoder_only/exp03")
EXP05_DIR = Path("../../../results/decoder_only/exp05")
EXP06_DIR = Path("../../../results/decoder_only/exp06")

DATASETS = ['ms_marco', 'squad_v2', 'triviaqa', 'hotpotqa', 'drop', 'boolq', 'gsm8k']

DS_SEEDS = {
    'ms_marco': SEED,
    'squad_v2': SEED + 100,
    'triviaqa': SEED + 200,
    'hotpotqa': SEED + 300,
    'drop': SEED + 400,
    'boolq': SEED + 500,
    'gsm8k': SEED + 700,
}

SCORING_KEY = 'bos_retained_exp14_transfer'

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", dtype=torch.bfloat16, token=HF_TOKEN,
)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

DEVICE = next(model.parameters()).device
BOS_ID = tokenizer.bos_token_id
NEWLINE_IDS = tokenizer("\n", add_special_tokens=False).input_ids
text_cfg = getattr(model.config, 'text_config', model.config)
VOCAB_SIZE = model.get_input_embeddings().num_embeddings
N_LAYERS = len(get_layer_types(model))
SLIDING_WINDOW = getattr(text_cfg, 'sliding_window', 4096)
SLIDING_CACHE_LIMIT = SLIDING_WINDOW - 1
HIDDEN_SIZE = getattr(text_cfg, 'hidden_size', 3840)

NL = len(NEWLINE_IDS)
COMMON_MAX_DOC = SLIDING_CACHE_LIMIT - 1 - 256 - NL

LAYER_INV_FREQS = build_layer_inv_freqs(model, DEVICE)
LAYER_TYPES = get_layer_types(model)

embed_fn = model.get_input_embeddings()

print(f"Model loaded: {MODEL_NAME}, DEVICE: {DEVICE}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"N_LAYERS: {N_LAYERS}, SLIDING_CACHE_LIMIT: {SLIDING_CACHE_LIMIT}")
print(f"COMMON_MAX_DOC: {COMMON_MAX_DOC}")


# ===================================================================
# EVALUATION FUNCTIONS (same as build_exp14.py)
# ===================================================================

def encode_phase_a_soft(doc_text, soft_prompt_embeddings, normalize=True):
    # Phase A with learned soft prompt embeddings.
    # Always with torch.no_grad() since this is eval-only.
    doc_ids = tokenizer(doc_text, add_special_tokens=False,
                        truncation=True, max_length=1024).input_ids
    if len(doc_ids) > COMMON_MAX_DOC:
        doc_ids = doc_ids[:COMMON_MAX_DOC]

    P = soft_prompt_embeddings.shape[0]
    _NL = len(NEWLINE_IDS)
    max_doc = SLIDING_CACHE_LIMIT - 1 - P - _NL
    if len(doc_ids) > max_doc:
        doc_ids = doc_ids[:max_doc]
    D = len(doc_ids)

    bos_emb = embed_fn(torch.tensor([[BOS_ID]], device=DEVICE))
    nl_emb = embed_fn(torch.tensor([NEWLINE_IDS], device=DEVICE))
    doc_emb = embed_fn(torch.tensor([doc_ids], device=DEVICE))

    inputs_embeds = torch.cat([
        bos_emb,
        soft_prompt_embeddings.unsqueeze(0),
        nl_emb,
        doc_emb,
    ], dim=1)

    pa = model(inputs_embeds=inputs_embeds, use_cache=True, output_attentions=False)
    cache = pa.past_key_values
    del pa

    keep_indices = [0] + list(range(1 + P + _NL, 1 + P + _NL + D))
    cache = select_kv_cache(cache, keep_indices)

    old_pos = torch.arange(1 + P + _NL, 1 + P + _NL + D, device=DEVICE)
    new_pos = torch.arange(1, D + 1, device=DEVICE)
    cache = reposition_kv_cache(cache, old_pos, new_pos,
                                LAYER_INV_FREQS, LAYER_TYPES, bos_start=0)

    if normalize:
        cache = deep_copy_cache(cache)
        norm_roundtrip_kv_cache(cache)

    return cache, D, doc_ids


def score_phase_b_eval(cache, D_effective, query_text, answer_text):
    # Phase B returning scalar NLL (no gradients).
    phase_b_start = D_effective + 1
    query_ids = tokenizer("\n" + query_text + "\n",
                          add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False,
                           truncation=True, max_length=256).input_ids
    if not answer_ids:
        return 0.0

    pb_ids = query_ids + answer_ids
    pos = torch.arange(phase_b_start, phase_b_start + len(pb_ids), device=DEVICE)

    with torch.no_grad():
        pb = model(
            input_ids=torch.tensor([pb_ids], device=DEVICE),
            past_key_values=cache,
            position_ids=pos.unsqueeze(0),
            use_cache=False,
        )

    n_q = len(query_ids)
    logits = pb.logits[0, n_q - 1:n_q - 1 + len(answer_ids), :].float()
    targets = torch.tensor(answer_ids, device=DEVICE)
    nll = -F.log_softmax(logits, dim=-1).gather(
        1, targets.unsqueeze(1)).squeeze(1).mean().item()
    del pb
    return nll


print("Functions defined: encode_phase_a_soft, score_phase_b_eval")
""")


# =====================================================================
# Cell 2: Dataset Loading (hard samples only — same as build_exp14.py)
# =====================================================================
code(r"""# Cell 2: Load 7 datasets, identify hard samples (eval only, no train/val split)
from datasets import load_dataset

print("=" * 70)
print("LOADING 7 DATASETS — HARD SAMPLES ONLY")
print("=" * 70)

hard_samples = {}
all_samples = {}

# ================================================================
# MS MARCO
# ================================================================
print("\n--- MS MARCO ---")
exp02_ckpt = json.loads((EXP02_DIR / "checkpoint.json").read_text())
exp02_results = exp02_ckpt['results']
assert len(exp02_results) == N_SAMPLES

msmarco_bare = np.array([r['nll_bare'] for r in exp02_results])
sorted_idx = np.argsort(msmarco_bare)[::-1]
msmarco_hard_idx = set(np.sort(sorted_idx[:N_HARD]).tolist())

ds_msmarco = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
msmarco_candidates = []
for item in ds_msmarco:
    if len(msmarco_candidates) >= 3 * N_SAMPLES:
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
            msmarco_candidates.append({
                'passage': pt, 'query': query, 'answer': answer,
                'word_count': wc,
            })
            break

np.random.seed(SEED)
indices = np.random.permutation(len(msmarco_candidates))[:N_SAMPLES]
msmarco_all = [msmarco_candidates[i] for i in indices]
del ds_msmarco, msmarco_candidates
gc.collect()

for i in range(min(20, N_SAMPLES)):
    assert msmarco_all[i]['query'][:50] == exp02_results[i]['query'][:50], \
        f"MS MARCO query mismatch at sample {i}"

hs_msmarco = []
for idx in range(N_SAMPLES):
    if idx in msmarco_hard_idx:
        s = dict(msmarco_all[idx])
        s['original_idx'] = idx
        s['nll_bare_ref'] = float(msmarco_bare[idx])
        hs_msmarco.append(s)
hard_samples['ms_marco'] = hs_msmarco
all_samples['ms_marco'] = msmarco_all
print(f"  MS MARCO: {len(hs_msmarco)} hard samples")
del exp02_ckpt, exp02_results
gc.collect()

# ================================================================
# SQuAD 2.0
# ================================================================
print("\n--- SQuAD 2.0 ---")
ds_squad = load_dataset("rajpurkar/squad_v2", split="validation")
squad_candidates = []
for item in ds_squad:
    answers = item.get('answers', {})
    answer_texts = answers.get('text', [])
    if not answer_texts:
        continue
    passage = item['context']
    query = item['question']
    answer = answer_texts[0]
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        squad_candidates.append({
            'passage': passage, 'query': query, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['squad_v2'])
sq_indices = np.random.permutation(len(squad_candidates))[:N_SAMPLES]
all_samples['squad_v2'] = [squad_candidates[i] for i in sq_indices]
del ds_squad, squad_candidates
gc.collect()

# ================================================================
# TriviaQA
# ================================================================
print("\n--- TriviaQA ---")
ds_trivia = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation")
trivia_candidates = []
for item in ds_trivia:
    entity_pages = item.get('entity_pages', {})
    wiki_contexts = entity_pages.get('wiki_context', [])
    if not wiki_contexts or not wiki_contexts[0]:
        continue
    words = wiki_contexts[0].split()[:500]
    passage = ' '.join(words)
    query = item['question']
    answer_val = item['answer']['value']
    aliases = item['answer'].get('aliases', [])
    passage_lower = passage.lower()
    found = answer_val.lower() in passage_lower
    if not found:
        for alias in aliases:
            if alias.lower() in passage_lower:
                found = True
                break
    if not found:
        continue
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer_val) >= 1:
        trivia_candidates.append({
            'passage': passage, 'query': query, 'answer': answer_val,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['triviaqa'])
tr_indices = np.random.permutation(len(trivia_candidates))[:N_SAMPLES]
all_samples['triviaqa'] = [trivia_candidates[i] for i in tr_indices]
del ds_trivia, trivia_candidates
gc.collect()

# ================================================================
# HotpotQA
# ================================================================
print("\n--- HotpotQA ---")
ds_hotpot = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
hotpot_candidates = []
for item in ds_hotpot:
    context = item.get('context', {})
    sf = item.get('supporting_facts', {})
    ctx_titles = context.get('title', [])
    ctx_sentences = context.get('sentences', [])
    sf_titles = sf.get('title', [])
    sf_sent_ids = sf.get('sent_id', [])
    title_to_sents = {}
    for title, sents in zip(ctx_titles, ctx_sentences):
        title_to_sents[title] = sents
    passage_parts = []
    for title, sid in zip(sf_titles, sf_sent_ids):
        if title in title_to_sents and sid < len(title_to_sents[title]):
            passage_parts.append(title_to_sents[title][sid])
    if not passage_parts:
        continue
    passage = ' '.join(passage_parts)
    query = item['question']
    answer = item['answer']
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        hotpot_candidates.append({
            'passage': passage, 'query': query, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['hotpotqa'])
hp_indices = np.random.permutation(len(hotpot_candidates))[:N_SAMPLES]
all_samples['hotpotqa'] = [hotpot_candidates[i] for i in hp_indices]
del ds_hotpot, hotpot_candidates
gc.collect()

# ================================================================
# DROP
# ================================================================
print("\n--- DROP ---")
ds_drop = load_dataset("ucinlp/drop", split="validation")
drop_candidates = []
for item in ds_drop:
    passage = item['passage']
    question = item['question']
    answers_spans = item.get('answers_spans', {})
    spans = answers_spans.get('spans', [])
    if not spans or not spans[0]:
        continue
    answer = spans[0]
    wc = count_words(passage)
    if 30 <= wc <= 500 and count_words(answer) >= 1:
        drop_candidates.append({
            'passage': passage, 'query': question, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['drop'])
drop_indices = np.random.permutation(len(drop_candidates))[:N_SAMPLES]
all_samples['drop'] = [drop_candidates[i] for i in drop_indices]
del ds_drop, drop_candidates
gc.collect()

# ================================================================
# BoolQ
# ================================================================
print("\n--- BoolQ ---")
ds_boolq = load_dataset("google/boolq", split="validation")
boolq_candidates = []
for item in ds_boolq:
    passage = item['passage']
    question = item['question']
    answer = "Yes" if item['answer'] else "No"
    wc = count_words(passage)
    if 30 <= wc <= 500:
        boolq_candidates.append({
            'passage': passage, 'query': question, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['boolq'])
boolq_indices = np.random.permutation(len(boolq_candidates))[:N_SAMPLES]
all_samples['boolq'] = [boolq_candidates[i] for i in boolq_indices]
del ds_boolq, boolq_candidates
gc.collect()

# ================================================================
# GSM8K
# ================================================================
print("\n--- GSM8K ---")
ds_gsm8k = load_dataset("openai/gsm8k", "main", split="test")
gsm8k_candidates = []
for item in ds_gsm8k:
    passage = item['question']
    raw_answer = item['answer']
    if '####' not in raw_answer:
        continue
    answer = raw_answer.split('####')[-1].strip()
    if not answer:
        continue
    query = "What is the answer?"
    wc = count_words(passage)
    if 30 <= wc <= 500:
        gsm8k_candidates.append({
            'passage': passage, 'query': query, 'answer': answer,
            'word_count': wc,
        })
np.random.seed(DS_SEEDS['gsm8k'])
gsm8k_indices = np.random.permutation(len(gsm8k_candidates))[:N_SAMPLES]
all_samples['gsm8k'] = [gsm8k_candidates[i] for i in gsm8k_indices]
del ds_gsm8k, gsm8k_candidates
gc.collect()

# ================================================================
# Load bare NLLs for hard selection (non-MS-MARCO datasets)
# ================================================================
print("\n--- Loading bare NLLs for hard selection ---")
BARE_NLL_SOURCES = {
    'squad_v2': EXP03_DIR / "bare_squad_v2.json",
    'triviaqa': EXP03_DIR / "bare_triviaqa.json",
    'hotpotqa': EXP03_DIR / "bare_hotpotqa.json",
    'drop': EXP05_DIR / "bare_drop.json",
    'boolq': EXP05_DIR / "bare_boolq.json",
    'gsm8k': EXP06_DIR / "bare_gsm8k.json",
}

for ds_name, bare_path in BARE_NLL_SOURCES.items():
    samples_ds = all_samples[ds_name]
    bare_ckpt = json.loads(bare_path.read_text())
    bare_nlls_all = bare_ckpt['bare_nlls'][:N_SAMPLES]

    saved_queries = bare_ckpt.get('queries_first50', [])
    n_check = min(len(saved_queries), len(samples_ds))
    current_queries = [s['query'][:50] for s in samples_ds[:n_check]]
    assert saved_queries[:n_check] == current_queries, f"{ds_name}: query alignment mismatch"

    bare_arr = np.array(bare_nlls_all)
    sorted_idx = np.argsort(bare_arr)[::-1]
    h_idx = set(np.sort(sorted_idx[:N_HARD]).tolist())

    hs = []
    for idx in range(N_SAMPLES):
        if idx in h_idx:
            s = dict(samples_ds[idx])
            s['original_idx'] = idx
            s['nll_bare_ref'] = float(bare_arr[idx])
            hs.append(s)
    hard_samples[ds_name] = hs
    print(f"  {ds_name}: {len(hs)} hard samples")

del bare_ckpt
gc.collect()

# Summary
print("\n" + "=" * 70)
for ds_name in DATASETS:
    n_h = len(hard_samples[ds_name])
    mean_bare = np.mean([s['nll_bare_ref'] for s in hard_samples[ds_name]])
    print(f"  {ds_name:<12}: {n_h} hard samples, mean bare NLL: {mean_bare:.3f}")
""")


# =====================================================================
# Cell 3: Load soft prompts + bare NLLs from Exp 14 checkpoints
# =====================================================================
code(r"""# Cell 3: Load all rand soft prompts and bare NLLs from Exp 14 checkpoints
print("=" * 70)
print("LOADING SOFT PROMPTS AND BARE NLLs")
print("=" * 70)

# Load per-dataset rand soft prompts
soft_prompts = {}  # ds_name -> soft_prompt tensor
for ds_name in DATASETS:
    pt_path = RESULTS_DIR / f"soft_prompt_{ds_name}_rand.pt"
    assert pt_path.exists(), f"Missing soft prompt: {pt_path}"
    sp = torch.load(pt_path, map_location=DEVICE, weights_only=True)
    soft_prompts[ds_name] = nn.Parameter(sp, requires_grad=False)
    print(f"  Loaded {ds_name} rand soft prompt: shape={sp.shape}")

# Load universal rand soft prompt
univ_pt_path = RESULTS_DIR / "soft_prompt_universal_rand.pt"
assert univ_pt_path.exists(), f"Missing universal soft prompt: {univ_pt_path}"
sp_univ = torch.load(univ_pt_path, map_location=DEVICE, weights_only=True)
soft_prompts['universal'] = nn.Parameter(sp_univ, requires_grad=False)
print(f"  Loaded universal rand soft prompt: shape={sp_univ.shape}")

# Load bare NLLs from Exp 14 checkpoints (already computed in main experiment)
bare_nlls = {}  # ds_name -> np.array of NLLs for 160 hard samples
for ds_name in DATASETS:
    ckpt_path = RESULTS_DIR / f"checkpoint_{ds_name}.json"
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
    ckpt = json.loads(ckpt_path.read_text())
    results = ckpt['results']
    assert len(results) == N_HARD, f"{ds_name}: expected {N_HARD} results, got {len(results)}"
    bare_nlls[ds_name] = np.array([r['nll_bare'] for r in results])
    print(f"  {ds_name}: {len(results)} bare NLLs, mean={bare_nlls[ds_name].mean():.4f}")
    del ckpt

print(f"\nSources: {len(soft_prompts)} soft prompts (7 per-dataset + 1 universal)")
print(f"Targets: {len(DATASETS)} datasets x {N_HARD} hard samples = {len(DATASETS) * N_HARD} evals per source")
""")


# =====================================================================
# Cell 4: Transfer evaluation loop
# =====================================================================
code(r"""# Cell 4: Cross-dataset transfer evaluation (8 sources x 7 targets)
print("=" * 70)
print("CROSS-DATASET TRANSFER EVALUATION")
print("=" * 70)

SOURCE_NAMES = DATASETS + ['universal']
transfer_nlls = {}  # (source, target) -> list of NLLs

ckpt_path = RESULTS_DIR / "transfer_checkpoint.json"

# Resume from checkpoint
completed_pairs = set()
if ckpt_path.exists():
    ckpt = json.loads(ckpt_path.read_text())
    if ckpt.get('scoring') == SCORING_KEY:
        for key, nlls in ckpt.get('results', {}).items():
            transfer_nlls[key] = nlls
            completed_pairs.add(key)
        print(f"  Resumed {len(completed_pairs)} completed source-target pairs")
    del ckpt

t0_all = time.time()
total_pairs = len(SOURCE_NAMES) * len(DATASETS)
pair_count = 0

for source_name in SOURCE_NAMES:
    sp = soft_prompts[source_name]

    for target_ds in DATASETS:
        pair_key = f"{source_name}_to_{target_ds}"
        pair_count += 1

        if pair_key in completed_pairs:
            print(f"  [{pair_count}/{total_pairs}] {pair_key}: cached ({len(transfer_nlls[pair_key])} samples)")
            continue

        hs = hard_samples[target_ds]
        print(f"\n  [{pair_count}/{total_pairs}] {source_name} -> {target_ds} ({len(hs)} samples)")
        t0 = time.time()
        nlls = []

        with torch.no_grad():
            for i, s in enumerate(tqdm(hs, desc=f"{source_name}->{target_ds}", leave=False)):
                cache, D, _ = encode_phase_a_soft(s['passage'], sp, normalize=True)
                nll = score_phase_b_eval(cache, D, s['query'], s['answer'])
                nlls.append(nll)
                del cache
                gc.collect()
                torch.cuda.empty_cache()

        transfer_nlls[pair_key] = nlls
        completed_pairs.add(pair_key)
        elapsed = time.time() - t0
        print(f"    mean NLL={np.mean(nlls):.4f}, {elapsed/60:.1f} min")

        # Checkpoint after each pair
        ckpt = {
            'scoring': SCORING_KEY,
            'results': transfer_nlls,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        ckpt_path.write_text(json.dumps(ckpt))

total_elapsed = time.time() - t0_all
print(f"\nTransfer evaluation complete: {pair_count} pairs in {total_elapsed/3600:.1f} hours")
""")


# =====================================================================
# Cell 5: Compute statistics and save transfer_matrix.json
# =====================================================================
code(r"""# Cell 5: Compute Cohen's d, win rate, p-value for each (source, target) vs bare
print("=" * 70)
print("COMPUTING TRANSFER STATISTICS")
print("=" * 70)

transfer_matrix = {
    'experiment': 'exp14_transfer_evaluation',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'scoring_key': SCORING_KEY,
    'source_prompts': SOURCE_NAMES,
    'target_datasets': DATASETS,
    'n_hard': N_HARD,
    'per_pair': {},
    'matrix_d': {},      # source -> {target -> Cohen's d}
    'matrix_win': {},    # source -> {target -> win rate}
    'matrix_p': {},      # source -> {target -> p-value}
}

print(f"\n{'Source':<12} {'Target':<12} {'d':>8} {'Win%':>7} {'p':>10} {'mean NLL':>10}")
print("-" * 65)

for source_name in SOURCE_NAMES:
    transfer_matrix['matrix_d'][source_name] = {}
    transfer_matrix['matrix_win'][source_name] = {}
    transfer_matrix['matrix_p'][source_name] = {}

    for target_ds in DATASETS:
        pair_key = f"{source_name}_to_{target_ds}"
        nlls = np.array(transfer_nlls[pair_key])
        bare = bare_nlls[target_ds]

        assert len(nlls) == len(bare) == N_HARD, \
            f"{pair_key}: length mismatch ({len(nlls)} vs {len(bare)})"

        diff = bare - nlls  # positive = soft prompt helps
        d = cohens_d(diff)
        w = win_rate(diff)
        _, p = paired_ttest(diff)

        transfer_matrix['per_pair'][pair_key] = {
            'source': source_name,
            'target': target_ds,
            'd': float(d),
            'win': float(w),
            'p': float(p),
            'mean_nll': float(np.mean(nlls)),
            'mean_bare': float(np.mean(bare)),
            'mean_diff': float(np.mean(diff)),
            'nlls': [float(x) for x in nlls],
        }
        transfer_matrix['matrix_d'][source_name][target_ds] = float(d)
        transfer_matrix['matrix_win'][source_name][target_ds] = float(w)
        transfer_matrix['matrix_p'][source_name][target_ds] = float(p)

        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"{source_name:<12} {target_ds:<12} {d:>+8.3f} {w:>6.1%} {p:>10.2e} {np.mean(nlls):>10.4f} {sig}")

# Save
out_path = RESULTS_DIR / "transfer_matrix.json"
with open(out_path, 'w') as f:
    json.dump(transfer_matrix, f, indent=2)
print(f"\nSaved transfer matrix: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

# Clean up checkpoint
if ckpt_path.exists():
    ckpt_path.unlink()
    print("Cleaned up transfer checkpoint")
""")


# =====================================================================
# Cell 6: Summary analysis
# =====================================================================
code(r"""# Cell 6: Summary — diagonal vs off-diagonal, transfer patterns
print("=" * 70)
print("TRANSFER ANALYSIS SUMMARY")
print("=" * 70)

# Diagonal (same-dataset) vs off-diagonal (cross-dataset)
diag_ds = []
off_diag_ds = []
for source_name in DATASETS:
    for target_ds in DATASETS:
        d = transfer_matrix['matrix_d'][source_name][target_ds]
        if source_name == target_ds:
            diag_ds.append(d)
        else:
            off_diag_ds.append(d)

print(f"\nPer-dataset soft prompts:")
print(f"  Diagonal (same-dataset):  mean d = {np.mean(diag_ds):+.3f} (n={len(diag_ds)})")
print(f"  Off-diagonal (transfer):  mean d = {np.mean(off_diag_ds):+.3f} (n={len(off_diag_ds)})")
print(f"  Specialization gap:       {np.mean(diag_ds) - np.mean(off_diag_ds):+.3f}")

# Universal performance
univ_ds = [transfer_matrix['matrix_d']['universal'][ds] for ds in DATASETS]
print(f"\nUniversal soft prompt:")
print(f"  Mean d across datasets:   {np.mean(univ_ds):+.3f}")
print(f"  vs diagonal (per-ds):     {np.mean(univ_ds) - np.mean(diag_ds):+.3f}")
print(f"  vs off-diagonal (xfer):   {np.mean(univ_ds) - np.mean(off_diag_ds):+.3f}")

# Best source for each target
print(f"\nBest source prompt for each target dataset:")
for target_ds in DATASETS:
    best_source = None
    best_d = -999
    for source_name in SOURCE_NAMES:
        d = transfer_matrix['matrix_d'][source_name][target_ds]
        if d > best_d:
            best_d = d
            best_source = source_name
    is_self = ' (self)' if best_source == target_ds else ''
    is_univ = ' (universal)' if best_source == 'universal' else ''
    print(f"  {target_ds:<12}: {best_source:<12} d={best_d:+.3f}{is_self}{is_univ}")

# Which per-dataset prompts transfer well?
print(f"\nTransfer quality by source (mean off-diagonal d):")
for source_name in DATASETS:
    off_diag = [transfer_matrix['matrix_d'][source_name][t]
                for t in DATASETS if t != source_name]
    print(f"  {source_name:<12}: mean d = {np.mean(off_diag):+.3f}, "
          f"range [{min(off_diag):+.3f}, {max(off_diag):+.3f}]")

print("\nDone.")
""")


# =====================================================================
# Write notebook
# =====================================================================
out_path = "experiments/decoder_only/14/14_transfer_eval.ipynb"
nbf.write(nb, out_path)

n_md = sum(1 for c in nb.cells if c.cell_type == 'markdown')
n_code = sum(1 for c in nb.cells if c.cell_type == 'code')
print(f"Notebook written to {out_path}")
print(f"Cells: {len(nb.cells)} ({n_md} markdown, {n_code} code)")
