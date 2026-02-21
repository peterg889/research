# Cell 1: Setup & Imports
import os
os.umask(0o000)  # File permission safety (two-user environment)

import json
import time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from tqdm.auto import tqdm
from datasets import load_dataset
import matplotlib.pyplot as plt

from lib.config import ExperimentConfig
from lib.kv_cache import (
    build_kv_cache,
    build_beacon_cache_sequential,
    build_beacon_cache_batch,
    extract_cache_at_indices,
    correct_rope_positions_chunked,
    extract_and_truncate_cache_with_bos,
    correct_rope_positions_with_bos,
    score_answer_with_cache,
    deepcopy_cache,
    _get_cache_keys,
    _get_cache_values,
    _ensure_dynamic_cache,
)
from lib.analysis import cohens_d
from lib.surrogate import STATIC_SURROGATE_QUERIES

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Paths
RESULTS_DIR = Path('results/exp18')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Model
MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.2'
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map='auto'
)
config = ExperimentConfig(device='cuda')

print(f'Model loaded: {MODEL_NAME}')
print(f'Device: {config.device}')
print(f'Results dir: {RESULTS_DIR}')

# Cell 2: Configuration

# Beacon text (static_factual — best performer from Exp 07)
BEACON_TEXT = STATIC_SURROGATE_QUERIES['static_factual']['query']
BEACON_IDS = tokenizer.encode(BEACON_TEXT, add_special_tokens=False)
BEACON_LEN = len(BEACON_IDS)

# Random beacon (same length, random tokens)
torch.manual_seed(SEED)
RANDOM_BEACON_IDS = torch.randint(100, tokenizer.vocab_size - 100, (BEACON_LEN,)).tolist()

# Chunk sizes
CHUNK_SIZES = [256, 512]

# MS MARCO
N_MSMARCO = 300
MSMARCO_CHECKPOINT_PATH = RESULTS_DIR / 'msmarco_checkpoint.json'
MSMARCO_RESULTS_PATH = RESULTS_DIR / 'msmarco_results.json'
CHECKPOINT_EVERY = 25

# NQ — stratified sampling
N_NQ = 240
NQ_CHECKPOINT_PATH = RESULTS_DIR / 'nq_checkpoint.json'
NQ_RESULTS_PATH = RESULTS_DIR / 'nq_results.json'
NQ_SAMPLES_CACHE_PATH = RESULTS_DIR / 'nq_samples.json'

LENGTH_BINS = [
    ('short',     100,  300),
    ('medium',    300,  800),
    ('long',      800,  2000),
    ('very_long', 2000, 4000),
]
SAMPLES_PER_BIN = {'short': 15, 'medium': 75, 'long': 75, 'very_long': 75}
MAX_DOC_WORDS = 4000

# Templates (matched to Exp 07/17)
QUERY_TEMPLATE = '\nQuery: {query}\nAnswer:'
ANSWER_TEMPLATE = ' {answer}'

# Condition names
MSMARCO_CONDITIONS = ['bare', 'single_prefix', 'beacon_seq_256', 'beacon_seq_512']
NQ_CONDITIONS = [
    'bare', 'single_prefix',
    'beacon_seq_256', 'beacon_seq_512',
    'beacon_trunc_256', 'random_beacon_256', 'beacon_batch_256',
]

print(f'Beacon text: "{BEACON_TEXT}"')
print(f'Beacon token length: {BEACON_LEN}')
print(f'Random beacon token length: {len(RANDOM_BEACON_IDS)}')
print(f'MS MARCO: N={N_MSMARCO}, conditions={MSMARCO_CONDITIONS}')
print(f'NQ: N={N_NQ}, conditions={NQ_CONDITIONS}')
print(f'Samples per NQ bin: {SAMPLES_PER_BIN}')

# Cell 3: Explain Experimental Conditions
print('=' * 70)
print('EXPERIMENTAL CONDITIONS EXPLAINED')
print('=' * 70)

# Show concrete example
example_doc = 'The quick brown fox jumps over the lazy dog. ' * 20  # ~180 words
example_doc_ids = tokenizer.encode(example_doc, add_special_tokens=False)
n_doc_tokens = len(example_doc_ids)
bos_id = tokenizer.bos_token_id

print(f'\nExample document: {n_doc_tokens} tokens')
print(f'Beacon: {BEACON_LEN} tokens ("{BEACON_TEXT}")')
print()

print('### 1. bare ###')
print(f'  Cache: [BOS][doc({n_doc_tokens} tokens)]')
print(f'  Total: {1 + n_doc_tokens} tokens')
print(f'  Baseline — document cached in isolation.')
print()

print('### 2. single_prefix ###')
prefix_ids = tokenizer.encode(BEACON_TEXT + '\n', add_special_tokens=False)
print(f'  Build: [BOS][prefix({len(prefix_ids)} tokens)][doc({n_doc_tokens} tokens)]')
print(f'  → Truncate prefix → RoPE correct')
print(f'  Query-time cache: [BOS][doc\'({n_doc_tokens} tokens)]  (contaminated values, bare positions)')
print(f'  Replicates Exp 07 static_fact_trunc.')
print()

n_chunks_256 = -(-n_doc_tokens // 256)
seq_256 = 1 + n_chunks_256 * (BEACON_LEN + min(256, n_doc_tokens))
# More accurate for variable last chunk
chunk_sizes_actual = [min(256, n_doc_tokens - i*256) for i in range(n_chunks_256)]
seq_256 = 1 + sum(BEACON_LEN + cs for cs in chunk_sizes_actual)
print(f'### 3. beacon_seq_256 ###')
print(f'  Build: [BOS][B({BEACON_LEN})][C1({chunk_sizes_actual[0]})][B({BEACON_LEN})][C2({chunk_sizes_actual[1] if len(chunk_sizes_actual) > 1 else "..."})]..')
print(f'  {n_chunks_256} chunks of ≤256 tokens, {n_chunks_256} beacons')
print(f'  Total: {seq_256} tokens (single forward pass, full causal attention)')
print(f'  Query-time: full cache (beacons present). Query sees beacon+doc interleaved.')
print(f'  MAIN TEST: periodic contamination sources throughout the document.')
print()

n_chunks_512 = -(-n_doc_tokens // 512)
chunk_sizes_512 = [min(512, n_doc_tokens - i*512) for i in range(n_chunks_512)]
seq_512 = 1 + sum(BEACON_LEN + cs for cs in chunk_sizes_512)
print(f'### 4. beacon_seq_512 ###')
print(f'  Same as #3 but stride=512. {n_chunks_512} chunks, total: {seq_512} tokens.')
print(f'  Tests whether closer beacons (256 vs 512) matter.')
print()

print(f'### 5. beacon_trunc_256 ###')
print(f'  Build: Same as #3 (beacon_seq_256)')
print(f'  → Extract doc positions only + per-chunk RoPE correction')
print(f'  Query-time cache: [BOS][doc\'({n_doc_tokens} tokens)]  (beacons removed)')
print(f'  Tests VALUE CONTAMINATION only (beacon attention removed at query time).')
print(f'  If #5 ≈ #3: contamination is the mechanism.')
print(f'  If #5 << #3: attention routing to beacons also matters.')
print()

print(f'### 6. random_beacon_256 ###')
print(f'  Build: [BOS][R({BEACON_LEN})][C1][R({BEACON_LEN})][C2]... (random tokens as beacons)')
print(f'  Same structure as #3 but beacons are random tokens, not semantic.')
print(f'  If #6 ≈ #3: structural effect (any periodic injection helps).')
print(f'  If #6 << #3: semantic content of beacons matters.')
print()

print(f'### 7. beacon_batch_256 ###')
print(f'  Build: Each [BOS][beacon][chunk_i] processed independently')
print(f'  → Flattened into single cache (BOS stripped from non-first chunks)')
print(f'  Same STRUCTURE as #3 but no cross-chunk attention during build.')
print(f'  If #7 ≈ #3: cross-chunk attention is irrelevant.')
print(f'  If #7 << #3: cross-chunk attention matters.')

# Cell 4: Helper Functions

def build_matched_beacon_caches(
    passage, query, answer,
    conditions, model, tokenizer, config,
    beacon_ids=BEACON_IDS, random_beacon_ids=RANDOM_BEACON_IDS,
    chunk_sizes=CHUNK_SIZES,
):
    """Build all condition caches for one sample with matched tokenization.

    Tokenizes the document once and reuses the same doc_ids across all conditions.

    Args:
        passage: Document text.
        query: Query text.
        answer: Answer text.
        conditions: List of condition names to evaluate.
        model, tokenizer, config: Standard model env.
        beacon_ids: Token IDs for the semantic beacon.
        random_beacon_ids: Token IDs for the random beacon.
        chunk_sizes: List of chunk sizes [256, 512].

    Returns:
        dict mapping condition_name -> NLL, plus 'doc_len_tokens'.
    """
    query_prompt = QUERY_TEMPLATE.format(query=query)
    answer_text = ANSWER_TEMPLATE.format(answer=answer)

    # Tokenize document once (no BOS, no framing)
    doc_ids = tokenizer.encode(passage, add_special_tokens=False)
    doc_len = len(doc_ids)
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
    bos_tensor = torch.tensor([[bos_id]], device=config.device)
    doc_tensor = torch.tensor([doc_ids], device=config.device)

    beacon_len = len(beacon_ids)

    def score(cache, seq_len, position_offset=0):
        return score_answer_with_cache(
            deepcopy_cache(cache), seq_len, query_prompt, answer_text,
            model, tokenizer, config, position_offset=position_offset
        )

    results = {'doc_len_tokens': doc_len}

    # --- bare ---
    if 'bare' in conditions:
        bare_input = torch.cat([bos_tensor, doc_tensor], dim=1)
        with torch.no_grad():
            out = model(
                input_ids=bare_input,
                attention_mask=torch.ones_like(bare_input),
                use_cache=True, return_dict=True,
            )
        results['bare'] = score(out.past_key_values, bare_input.shape[1])
        del out

    # --- single_prefix ---
    if 'single_prefix' in conditions:
        prefix_text = BEACON_TEXT + '\n'
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        prefix_tensor = torch.tensor([prefix_ids], device=config.device)
        full_input = torch.cat([bos_tensor, prefix_tensor, doc_tensor], dim=1)
        prefix_token_len = 1 + len(prefix_ids)  # BOS + prefix

        with torch.no_grad():
            out = model(
                input_ids=full_input,
                attention_mask=torch.ones_like(full_input),
                use_cache=True, return_dict=True,
            )
        truncated = extract_and_truncate_cache_with_bos(out.past_key_values, doc_len)
        keep_len = 1 + doc_len
        surrogate_offset = prefix_token_len - 1
        correct_rope_positions_with_bos(truncated, surrogate_offset, model)
        results['single_prefix'] = score(truncated, keep_len)
        del out, truncated

    # --- beacon_seq_256 and beacon_seq_512 ---
    for cs in chunk_sizes:
        cond_name = f'beacon_seq_{cs}'
        if cond_name not in conditions:
            continue
        cache, seq_len, _, bpos, dpos = build_beacon_cache_sequential(
            passage, beacon_ids, cs, model, tokenizer, config
        )
        results[cond_name] = score(cache, seq_len)

        # Save 256 cache artifacts for beacon_trunc_256
        if cs == 256:
            cache_256 = cache
            bpos_256 = bpos
            dpos_256 = dpos
            seq_len_256 = seq_len
        else:
            del cache

    # --- beacon_trunc_256 ---
    if 'beacon_trunc_256' in conditions:
        # Build beacon_seq_256 if not already done
        if 'cache_256' not in dir():
            cache_256, seq_len_256, _, bpos_256, dpos_256 = \
                build_beacon_cache_sequential(
                    passage, beacon_ids, 256, model, tokenizer, config
                )

        extract_indices = [0] + dpos_256  # BOS + doc positions
        extracted = extract_cache_at_indices(cache_256, extract_indices)

        # Compute per-chunk RoPE correction
        n_chunks = -(-doc_len // 256)
        boundaries = []
        offsets_list = []
        pos = 1  # after BOS
        for k in range(n_chunks):
            boundaries.append(pos)
            offsets_list.append((k + 1) * beacon_len)
            actual_chunk = min(256, doc_len - k * 256)
            pos += actual_chunk

        correct_rope_positions_chunked(extracted, boundaries, offsets_list, model)
        keep_len = 1 + doc_len
        results['beacon_trunc_256'] = score(extracted, keep_len)
        del extracted

    # Clean up 256 cache if still around
    if 'cache_256' in dir():
        del cache_256

    # --- random_beacon_256 ---
    if 'random_beacon_256' in conditions:
        cache, seq_len, _, _, _ = build_beacon_cache_sequential(
            passage, random_beacon_ids, 256, model, tokenizer, config
        )
        results['random_beacon_256'] = score(cache, seq_len)
        del cache

    # --- beacon_batch_256 ---
    if 'beacon_batch_256' in conditions:
        cache, seq_len, _ = build_beacon_cache_batch(
            passage, beacon_ids, 256, model, tokenizer, config
        )
        results['beacon_batch_256'] = score(cache, seq_len)
        del cache

    torch.cuda.empty_cache()
    return results


def run_analysis(results, condition_names, dataset_label):
    """Run statistical analysis on experiment results."""
    cond_arrays = {cn: np.array([r[cn] for r in results]) for cn in condition_names}

    # Filter failed samples
    valid = np.ones(len(results), dtype=bool)
    for cn in condition_names:
        valid &= np.isfinite(cond_arrays[cn])
        valid &= (cond_arrays[cn] != 0)
    n_valid = int(np.sum(valid))
    n_excluded = len(results) - n_valid

    c = {cn: cond_arrays[cn][valid] for cn in condition_names}

    print(f'\n{"=" * 70}')
    print(f'{dataset_label} ANALYSIS (n_valid={n_valid}, excluded={n_excluded})')
    print('=' * 70)

    # All conditions vs bare
    print(f'\n{"Condition":<25} {"Mean NLL":>10} {"Mean Δ":>10} {"d":>8} {"Win%":>7} {"p":>12} {"sig":>5}')
    print('-' * 80)

    all_vs_bare = {}
    for cn in condition_names:
        if cn == 'bare':
            print(f'{cn:<25} {np.mean(c[cn]):>10.4f} {"---":>10} {"---":>8} {"---":>7}')
            continue
        delta = c['bare'] - c[cn]  # positive = condition is better
        d = cohens_d(delta)
        win = np.mean(delta > 0) * 100
        t_stat, p_val = stats.ttest_1samp(delta, 0)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f'{cn:<25} {np.mean(c[cn]):>10.4f} {np.mean(delta):>+10.4f} {d:>+8.3f} {win:>6.1f}% {p_val:>12.2e} {sig:>5}')
        all_vs_bare[cn] = {
            'mean_nll': float(np.mean(c[cn])),
            'mean_delta': float(np.mean(delta)),
            'cohens_d': float(d),
            'win_pct': float(win),
            't_stat': float(t_stat),
            'p_value': float(p_val),
        }

    return {'n_valid': n_valid, 'n_excluded': n_excluded, 'all_vs_bare': all_vs_bare, 'conditions': c}


print('Helper functions defined.')

# Cell 5: MS MARCO Evaluation

from lib.data import load_ms_marco, load_evaluation_samples

# Load data
msmarco_dataset = load_ms_marco(config)
all_msmarco_samples = load_evaluation_samples(msmarco_dataset, config, require_answer=True)
msmarco_samples = all_msmarco_samples[:N_MSMARCO]
print(f'MS MARCO samples: {len(msmarco_samples)}')

# Evaluate conditions 1-4 (bare, single_prefix, beacon_seq_256, beacon_seq_512)
msmarco_results = []
start_idx = 0

if MSMARCO_CHECKPOINT_PATH.exists():
    with open(MSMARCO_CHECKPOINT_PATH, 'r') as f:
        ckpt = json.load(f)
    ckpt_queries = ckpt.get('sample_queries', [])
    current_queries = [s['query'] for s in msmarco_samples]
    if ckpt_queries == current_queries:
        msmarco_results = ckpt['results']
        start_idx = len(msmarco_results)
        print(f'Resuming from checkpoint: {start_idx}/{N_MSMARCO}')
    else:
        print('Checkpoint sample mismatch. Starting fresh.')

for idx in tqdm(range(start_idx, N_MSMARCO), initial=start_idx, total=N_MSMARCO,
                desc='MS MARCO eval'):
    sample = msmarco_samples[idx]

    result = build_matched_beacon_caches(
        sample['passage'], sample['query'], sample['answer'],
        MSMARCO_CONDITIONS, model, tokenizer, config,
    )
    result['idx'] = idx
    result['word_count'] = len(sample['passage'].split())
    msmarco_results.append(result)

    if (idx + 1) % CHECKPOINT_EVERY == 0 or idx == N_MSMARCO - 1:
        ckpt_data = {
            'results': msmarco_results,
            'sample_queries': [s['query'] for s in msmarco_samples],
            'completed': len(msmarco_results),
            'total': N_MSMARCO,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(MSMARCO_CHECKPOINT_PATH, 'w') as f:
            json.dump(ckpt_data, f)

print(f'MS MARCO evaluation complete: {len(msmarco_results)} samples')

# Cell 6: MS MARCO Analysis

msmarco_analysis = run_analysis(msmarco_results, MSMARCO_CONDITIONS, 'MS MARCO (Short Docs)')

# Head-to-head comparisons
c = msmarco_analysis['conditions']
print(f'\n--- Key Comparisons (MS MARCO) ---')
print(f'{"Comparison":<45} {"Mean Δ":>10} {"d":>8} {"p":>12}')
print('-' * 80)

h2h = {}
comparisons = [
    ('beacon_seq_256 vs bare', 'beacon_seq_256', 'bare'),
    ('single_prefix vs bare', 'single_prefix', 'bare'),
    ('beacon_seq_256 vs single_prefix', 'beacon_seq_256', 'single_prefix'),
    ('beacon_seq_256 vs beacon_seq_512', 'beacon_seq_256', 'beacon_seq_512'),
]
for label, cond_a, cond_b in comparisons:
    delta = c[cond_b] - c[cond_a]  # positive = cond_a is better (lower NLL)
    d = cohens_d(delta)
    _, p_val = stats.ttest_1samp(delta, 0)
    print(f'{label:<45} {np.mean(delta):>+10.4f} {d:>+8.3f} {p_val:>12.2e}')
    h2h[label] = {'d': float(d), 'p': float(p_val)}

# Save
msmarco_final = {
    'experiment': 'exp18_periodic_beacon_msmarco',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'config': {
        'model_name': MODEL_NAME, 'seed': SEED, 'n_eval': N_MSMARCO,
        'dataset': 'MS MARCO v1.1', 'beacon_text': BEACON_TEXT,
        'beacon_len': BEACON_LEN, 'chunk_sizes': CHUNK_SIZES,
    },
    'condition_names': MSMARCO_CONDITIONS,
    'analysis': {k: v for k, v in msmarco_analysis.items() if k != 'conditions'},
    'head_to_head': h2h,
    'per_sample_results': msmarco_results,
}
with open(MSMARCO_RESULTS_PATH, 'w') as f:
    json.dump(msmarco_final, f, indent=2)
print(f'\nSaved to {MSMARCO_RESULTS_PATH}')

# Cell 7: NQ Evaluation (Long Docs)

# --- Load NQ samples (stratified by length) ---
if NQ_SAMPLES_CACHE_PATH.exists():
    with open(NQ_SAMPLES_CACHE_PATH, 'r') as f:
        nq_cache = json.load(f)
    nq_samples = nq_cache['samples']
    print(f'Loaded {len(nq_samples)} NQ samples from cache')
else:
    print('Loading NQ dataset (streaming)...')
    nq = load_dataset(
        'google-research-datasets/natural_questions',
        split='validation',
        streaming=True,
    )

    bin_samples = {name: [] for name, _, _ in LENGTH_BINS}
    n_processed = 0

    for example in tqdm(nq, desc='Processing NQ'):
        n_processed += 1

        # Extract clean document text
        doc_tokens = example['document']['tokens']
        if isinstance(doc_tokens, dict):
            token_strs = doc_tokens['token']
            is_html_flags = doc_tokens['is_html']
            clean_tokens = [t for t, h in zip(token_strs, is_html_flags) if not h]
        else:
            clean_tokens = [t['token'] for t in doc_tokens if not t['is_html']]

        doc_text = ' '.join(clean_tokens)
        word_count = len(doc_text.split())

        if word_count < LENGTH_BINS[0][1]:
            continue
        if word_count > MAX_DOC_WORDS:
            words = doc_text.split()
            doc_text = ' '.join(words[:MAX_DOC_WORDS])
            word_count = MAX_DOC_WORDS

        # Extract short answer
        annotations = example['annotations']
        short_answers_list = annotations['short_answers']

        answer_text = None
        for annotator_sa in short_answers_list:
            if not annotator_sa:
                continue
            texts = annotator_sa.get('text', [])
            if texts:
                answer_text = texts[0]
                break
            starts = annotator_sa.get('start_token', [])
            ends = annotator_sa.get('end_token', [])
            if not starts or not ends:
                continue
            start_tok = starts[0] if isinstance(starts, list) else starts
            end_tok = ends[0] if isinstance(ends, list) else ends
            if start_tok >= 0 and end_tok > start_tok:
                if isinstance(doc_tokens, dict):
                    ans_tokens = [
                        doc_tokens['token'][i]
                        for i in range(start_tok, min(end_tok, len(doc_tokens['token'])))
                        if not doc_tokens['is_html'][i]
                    ]
                else:
                    ans_tokens = [
                        doc_tokens[i]['token']
                        for i in range(start_tok, min(end_tok, len(doc_tokens)))
                        if not doc_tokens[i]['is_html']
                    ]
                if ans_tokens:
                    answer_text = ' '.join(ans_tokens)
                    break

        if not answer_text or len(answer_text.strip()) == 0:
            continue
        if len(answer_text.split()) > 20:
            continue

        question = example['question']
        query = question.get('text', '') if isinstance(question, dict) else str(question)

        # Assign to length bin
        for bin_name, bin_min, bin_max in LENGTH_BINS:
            if bin_min <= word_count < bin_max:
                if len(bin_samples[bin_name]) < SAMPLES_PER_BIN[bin_name]:
                    bin_samples[bin_name].append({
                        'passage': doc_text,
                        'query': query,
                        'answer': answer_text,
                        'word_count': word_count,
                        'length_bin': bin_name,
                    })
                break

        if all(len(bin_samples[name]) >= SAMPLES_PER_BIN[name] for name, _, _ in LENGTH_BINS):
            break

    # Combine
    nq_samples = []
    for bin_name, _, _ in LENGTH_BINS:
        bs = bin_samples[bin_name]
        np.random.seed(SEED)
        np.random.shuffle(bs)
        nq_samples.extend(bs)
        print(f'  {bin_name}: {len(bs)} samples')

    with open(NQ_SAMPLES_CACHE_PATH, 'w') as f:
        json.dump({'samples': nq_samples, 'n_processed': n_processed}, f)
    print(f'Saved {len(nq_samples)} NQ samples')

N_NQ = min(N_NQ, len(nq_samples))
print(f'NQ samples to evaluate: {N_NQ}')

# Evaluate all 7 conditions
nq_results = []
start_idx = 0

if NQ_CHECKPOINT_PATH.exists():
    with open(NQ_CHECKPOINT_PATH, 'r') as f:
        ckpt = json.load(f)
    ckpt_queries = ckpt.get('sample_queries', [])
    current_queries = [s['query'] for s in nq_samples[:N_NQ]]
    if ckpt_queries == current_queries:
        nq_results = ckpt['results']
        start_idx = len(nq_results)
        print(f'Resuming from checkpoint: {start_idx}/{N_NQ}')
    else:
        print('Checkpoint sample mismatch. Starting fresh.')

for idx in tqdm(range(start_idx, N_NQ), initial=start_idx, total=N_NQ,
                desc='NQ eval'):
    sample = nq_samples[idx]

    result = build_matched_beacon_caches(
        sample['passage'], sample['query'], sample['answer'],
        NQ_CONDITIONS, model, tokenizer, config,
    )
    result['idx'] = idx
    result['word_count'] = sample['word_count']
    result['length_bin'] = sample['length_bin']
    nq_results.append(result)

    if (idx + 1) % CHECKPOINT_EVERY == 0 or idx == N_NQ - 1:
        ckpt_data = {
            'results': nq_results,
            'sample_queries': [s['query'] for s in nq_samples[:N_NQ]],
            'completed': len(nq_results),
            'total': N_NQ,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(NQ_CHECKPOINT_PATH, 'w') as f:
            json.dump(ckpt_data, f)

print(f'NQ evaluation complete: {len(nq_results)} samples')

# Cell 8: NQ Analysis

nq_analysis = run_analysis(nq_results, NQ_CONDITIONS, 'NQ (Long Docs)')
c = nq_analysis['conditions']

# --- Key Comparisons ---
print(f'\n--- Key Comparisons (NQ) ---')
print(f'{"Question":<55} {"Comparison":<35} {"d":>8} {"p":>12}')
print('-' * 115)

nq_h2h = {}
key_comparisons = [
    ('Do periodic beacons help long docs?', 'beacon_seq_256', 'bare'),
    ('Does stride matter?', 'beacon_seq_256', 'beacon_seq_512'),
    ('Contamination or attention routing?', 'beacon_trunc_256', 'beacon_seq_256'),
    ('Semantic or structural?', 'random_beacon_256', 'beacon_seq_256'),
    ('Cross-chunk attention needed?', 'beacon_batch_256', 'beacon_seq_256'),
    ('Better than single prefix?', 'beacon_seq_256', 'single_prefix'),
]
for question, cond_a, cond_b in key_comparisons:
    delta = c[cond_b] - c[cond_a]  # positive = cond_a is better
    d = cohens_d(delta)
    _, p_val = stats.ttest_1samp(delta, 0)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    label = f'{cond_a} vs {cond_b}'
    print(f'{question:<55} {label:<35} {d:>+8.3f} {p_val:>12.2e} {sig}')
    nq_h2h[question] = {'cond_a': cond_a, 'cond_b': cond_b, 'd': float(d), 'p': float(p_val)}

# --- Length-Stratified Analysis ---
print(f'\n--- Length-Stratified Analysis ---')
length_bins_arr = np.array([r['length_bin'] for r in nq_results])
valid_mask = np.ones(len(nq_results), dtype=bool)
for cn in NQ_CONDITIONS:
    arr = np.array([r[cn] for r in nq_results])
    valid_mask &= np.isfinite(arr) & (arr != 0)
length_bins_valid = length_bins_arr[valid_mask]
word_counts_valid = np.array([r['word_count'] for r in nq_results])[valid_mask]

c_valid = {cn: np.array([r[cn] for r in nq_results])[valid_mask] for cn in NQ_CONDITIONS}

bin_names_ordered = [name for name, _, _ in LENGTH_BINS]
per_bin = {}

header = f'{"Condition":<25}'
for bn in bin_names_ordered:
    header += f'{bn:>15}'
print(header)
print('-' * (25 + 15 * len(bin_names_ordered)))

for cn in NQ_CONDITIONS:
    if cn == 'bare':
        continue
    row = f'{cn:<25}'
    cn_bins = {}
    for bn in bin_names_ordered:
        mask = length_bins_valid == bn
        n_bin = int(np.sum(mask))
        if n_bin < 5:
            row += f'{"n/a":>15}'
            continue
        delta = c_valid['bare'][mask] - c_valid[cn][mask]
        d = cohens_d(delta)
        win = np.mean(delta > 0) * 100
        row += f'{d:>+7.3f} ({win:4.0f}%)'
        cn_bins[bn] = {'d': float(d), 'win': float(win), 'n': n_bin}
    print(row)
    per_bin[cn] = cn_bins

# --- Mechanism Decomposition ---
print(f'\n--- Mechanism Decomposition ---')
print('beacon_seq_256 = value contamination + attention routing to beacons')
print('beacon_trunc_256 = value contamination only (beacons removed at query time)')
print()
for bn in bin_names_ordered:
    mask = length_bins_valid == bn
    n_bin = int(np.sum(mask))
    if n_bin < 5:
        continue
    d_seq = cohens_d(c_valid['bare'][mask] - c_valid['beacon_seq_256'][mask])
    d_trunc = cohens_d(c_valid['bare'][mask] - c_valid['beacon_trunc_256'][mask])
    d_diff = d_seq - d_trunc
    print(f'  {bn}: seq d={d_seq:+.3f}, trunc d={d_trunc:+.3f}, attention contribution={d_diff:+.3f}')

# --- Length correlation ---
print(f'\n--- Length × Effect Size Correlation ---')
interaction = {}
for cn in NQ_CONDITIONS:
    if cn == 'bare':
        continue
    delta = c_valid['bare'] - c_valid[cn]
    r_s, p_s = spearmanr(word_counts_valid, delta)
    print(f'  {cn}: Spearman r={r_s:+.3f} (p={p_s:.3f})')
    interaction[cn] = {'spearman_r': float(r_s), 'spearman_p': float(p_s)}

nq_analysis_clean = {k: v for k, v in nq_analysis.items() if k != 'conditions'}
nq_analysis_clean['head_to_head'] = nq_h2h
nq_analysis_clean['per_bin'] = per_bin
nq_analysis_clean['length_interaction'] = interaction

# Save
nq_final = {
    'experiment': 'exp18_periodic_beacon_nq',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'config': {
        'model_name': MODEL_NAME, 'seed': SEED, 'n_eval': N_NQ,
        'dataset': 'google-research-datasets/natural_questions',
        'beacon_text': BEACON_TEXT, 'beacon_len': BEACON_LEN,
        'chunk_sizes': CHUNK_SIZES, 'length_bins': LENGTH_BINS,
        'samples_per_bin': SAMPLES_PER_BIN,
    },
    'condition_names': NQ_CONDITIONS,
    'analysis': nq_analysis_clean,
    'per_sample_results': nq_results,
}
with open(NQ_RESULTS_PATH, 'w') as f:
    json.dump(nq_final, f, indent=2)
print(f'\nSaved to {NQ_RESULTS_PATH}')

# Cell 9: Summary & Figures

# Reload results for analysis (in case cells are run out of order)
nq_c = {}
nq_arr = {cn: np.array([r[cn] for r in nq_results]) for cn in NQ_CONDITIONS}
nq_valid = np.ones(len(nq_results), dtype=bool)
for cn in NQ_CONDITIONS:
    nq_valid &= np.isfinite(nq_arr[cn]) & (nq_arr[cn] != 0)
nq_c = {cn: nq_arr[cn][nq_valid] for cn in NQ_CONDITIONS}
nq_bins_arr = np.array([r['length_bin'] for r in nq_results])[nq_valid]
nq_words_arr = np.array([r['word_count'] for r in nq_results])[nq_valid]

# --- Figure 1: Effect Size Comparison (all conditions vs bare) ---
fig, ax = plt.subplots(figsize=(10, 6))
conds_to_plot = [cn for cn in NQ_CONDITIONS if cn != 'bare']
ds = [cohens_d(nq_c['bare'] - nq_c[cn]) for cn in conds_to_plot]
colors = ['#2196F3', '#4CAF50', '#8BC34A', '#FF9800', '#F44336', '#9C27B0']
x_pos = np.arange(len(conds_to_plot))

bars = ax.bar(x_pos, ds, color=colors[:len(conds_to_plot)], alpha=0.85, edgecolor='black', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(conds_to_plot, rotation=25, ha='right', fontsize=9)
ax.set_ylabel("Cohen's d vs bare (positive = better)", fontsize=11)
ax.set_title('Exp 18: Effect Sizes on NQ (All Conditions vs Bare)', fontsize=13)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
for i, (d_val, bar) in enumerate(zip(ds, bars)):
    ax.text(bar.get_x() + bar.get_width() / 2, d_val + 0.01 * np.sign(d_val),
            f'{d_val:+.3f}', ha='center', va='bottom' if d_val >= 0 else 'top', fontsize=8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'effect_sizes_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {FIGURES_DIR / "effect_sizes_comparison.png"}')

# --- Figure 2: Length × Method Interaction ---
fig, ax = plt.subplots(figsize=(10, 6))

plot_conds = [
    ('single_prefix', '#2196F3', 'o', '--'),
    ('beacon_seq_256', '#4CAF50', 's', '-'),
    ('beacon_trunc_256', '#FF9800', '^', '-'),
    ('random_beacon_256', '#F44336', 'D', ':'),
    ('beacon_batch_256', '#9C27B0', 'v', ':'),
]

bin_names_ordered = [name for name, _, _ in LENGTH_BINS]
for cn, color, marker, ls in plot_conds:
    plot_ds = []
    bin_centers = []
    for bn, bmin, bmax in LENGTH_BINS:
        mask = nq_bins_arr == bn
        n_bin = int(np.sum(mask))
        if n_bin < 5:
            continue
        d = cohens_d(nq_c['bare'][mask] - nq_c[cn][mask])
        plot_ds.append(d)
        bin_centers.append((bmin + bmax) / 2)
    ax.plot(bin_centers, plot_ds, marker=marker, label=cn, color=color,
            linewidth=2, markersize=8, linestyle=ls)

ax.set_xlabel('Document Length (words, bin center)', fontsize=11)
ax.set_ylabel("Cohen's d vs bare", fontsize=11)
ax.set_title('Exp 18: Length × Method Interaction (NQ)', fontsize=13)
ax.legend(fontsize=9)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xscale('log')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'length_interaction.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {FIGURES_DIR / "length_interaction.png"}')

# --- Figure 3: Mechanism Decomposition ---
fig, ax = plt.subplots(figsize=(8, 5))

decomp_conds = ['beacon_seq_256', 'beacon_trunc_256']
width = 0.35
x = np.arange(len(bin_names_ordered))

for i, (cn, color, label) in enumerate([
    ('beacon_seq_256', '#4CAF50', 'Full (contamination + attention)'),
    ('beacon_trunc_256', '#FF9800', 'Truncated (contamination only)'),
]):
    bin_ds = []
    for bn in bin_names_ordered:
        mask = nq_bins_arr == bn
        if np.sum(mask) < 5:
            bin_ds.append(0)
        else:
            bin_ds.append(cohens_d(nq_c['bare'][mask] - nq_c[cn][mask]))
    ax.bar(x + i * width - width / 2, bin_ds, width, label=label,
           color=color, alpha=0.85, edgecolor='black', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(bin_names_ordered)
ax.set_ylabel("Cohen's d vs bare")
ax.set_title('Mechanism Decomposition: Contamination vs Attention')
ax.legend()
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'mechanism_decomposition.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {FIGURES_DIR / "mechanism_decomposition.png"}')

# --- Summary ---
print('\n' + '=' * 70)
print('EXPERIMENT 18 SUMMARY')
print('=' * 70)
print()
print('Key question: Do periodic beacon injections recover priming benefit')
print('on long documents where single-prefix priming fails?')
print()
print('NQ Results (all conditions vs bare):')
for cn in NQ_CONDITIONS:
    if cn == 'bare':
        continue
    delta = nq_c['bare'] - nq_c[cn]
    d = cohens_d(delta)
    win = np.mean(delta > 0) * 100
    _, p = stats.ttest_1samp(delta, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    verdict = 'HELPS' if d > 0 and p < 0.05 else 'HURTS' if d < 0 and p < 0.05 else 'NEUTRAL'
    print(f'  {cn:<25} d={d:+.3f}  win={win:.0f}%  p={p:.2e}  {sig}  → {verdict}')

# Save combined analysis
combined = {
    'experiment': 'exp18_periodic_beacon',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'msmarco': {k: v for k, v in msmarco_analysis.items() if k != 'conditions'},
    'nq': nq_analysis_clean,
}
with open(RESULTS_DIR / 'analysis_summary.json', 'w') as f:
    json.dump(combined, f, indent=2)
print(f'\nAll results saved to {RESULTS_DIR}/')
