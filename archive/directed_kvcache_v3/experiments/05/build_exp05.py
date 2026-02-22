#!/usr/bin/env python3
# Build Exp 05 notebook: LLM-Generated Surrogate Queries.
#
# Two-phase design:
#   Phase 1: Load Gemma 2 9B-IT, generate surrogates (3 prompts x 500), save JSON, free VRAM
#   Phase 2: Load T5Gemma, score 14 conditions x 500 samples, save results
#
# 14 conditions, 500 samples = 7,000 scoring passes.
# Total runtime: ~75 min generation + ~23 min scoring + overhead ~ 105 min.

import json
import nbformat as nbf

nb = nbf.v4.new_notebook()


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ===== Cell 1: Markdown =====
md(r"""# Experiment 05: LLM-Generated Surrogate Queries

## Motivation

Exps 2B through 3F established the mechanism:
- **~85% structural** (attention sink redistribution, Exp 3E)
- **~15% vocabulary** at higher repetitions (Exp 3F)
- **~10% word-order semantics**, flat regardless of repetition (Exp 3F)
- **Stop words help** — stripping them halves the semantic ratio (Exp 3F)
- **Complementary > overlapping** — surr_lead (echoes doc) was worst at 40% (Exp 02)
- **Real queries cause interference** on neural-bridge (all surrogates > oracle, Exp 3D)

Can an LLM generate surrogates that beat simple heuristics like "What is [keyword]?"?

## Prompt Design (Informed by Prior Findings)

### Prompt A — "need" (hypothesis for best)
> Write a short web search (5-8 words) someone would type to find this
> document. Focus on what the searcher NEEDS, not what the document says.
> Use simple everyday language.

Rationale: complementary vocabulary (avoids surr_lead overlap), natural stop-word
scaffolding (Exp 3F showed stop words help), broad not specific (avoids semantic
interference from Exp 3D).

### Prompt B — "question" (traditional approach)
> Write a short question (5-8 words) that this document answers.

Rationale: traditional query generation. May cause semantic interference.

### Prompt C — "keywords" (stop-word-free control)
> List 3-5 search keywords for this document, separated by spaces.
> Only content words, no filler words.

Rationale: tests stop-word scaffolding hypothesis. If keywords underperform
need-focused, confirms stop words matter.

## Conditions (14)

| # | Condition | Prefix | Group |
|---|-----------|--------|-------|
| 1 | `bare` | (none) | control |
| 2 | `oracle_x1_trunc` | query x 1 | upper bound |
| 3 | `oracle_x4_trunc` | query x 4 | upper bound + rep |
| 4 | `llm_need_x1_trunc` | Prompt A output x 1 | LLM (main) |
| 5 | `llm_need_x4_trunc` | Prompt A output x 4 | LLM + rep |
| 6 | `llm_question_x1_trunc` | Prompt B output x 1 | LLM (question) |
| 7 | `llm_question_x4_trunc` | Prompt B output x 4 | LLM + rep |
| 8 | `llm_keywords_x1_trunc` | Prompt C output x 1 | LLM (keywords) |
| 9 | `llm_keywords_x4_trunc` | Prompt C output x 4 | LLM + rep |
| 10 | `surr_template_x1_trunc` | "What is [kw]?" x 1 | heuristic |
| 11 | `surr_template_x4_trunc` | "What is [kw]?" x 4 | heuristic + rep |
| 12 | `random_x1_trunc` | random_matched x 1 | structural ctrl |
| 13 | `random_x4_trunc` | random_matched x 4 | structural ctrl + rep |
| 14 | `scrambled_llm_need_x4_trunc` | shuffled Prompt A x 4 | decomposition |

## Analysis

- **Part 1**: LLM vs heuristic (key question)
- **Part 2**: Prompt variant comparison (need vs question vs keywords)
- **Part 3**: Repetition + vocabulary/semantics decomposition
- **Part 4**: Hardness interaction""")


# ===== Cell 2: Setup =====
code(r"""# Cell 2: Setup
import os
os.umask(0o000)

import sys, json, time, re, gc, random as pyrandom
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from scipy import stats
from tqdm.auto import tqdm

sys.path.insert(0, "../..")
from lib.analysis import cohens_d

SEED = 42
N_SAMPLES = 500
T5GEMMA_NAME = "google/t5gemma-2-4b-4b"
GEMMA_IT_NAME = "google/gemma-2-9b-it"

RESULTS_DIR = Path("../../results/exp05")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SURROGATES_PATH = RESULTS_DIR / "surrogates.json"
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"
EXP02_CHECKPOINT = Path("../../results/exp02/checkpoint.json")

np.random.seed(SEED)
torch.manual_seed(SEED)
pyrandom.seed(SEED)

# Prompt templates for Gemma 2 9B-IT
PROMPT_NEED = (
    "Write a short web search (5-8 words) someone would type to find this "
    "document. Focus on what the searcher NEEDS, not what the document says. "
    "Use simple everyday language."
)
PROMPT_QUESTION = (
    "Write a short question (5-8 words) that this document answers."
)
PROMPT_KEYWORDS = (
    "List 3-5 search keywords for this document, separated by spaces. "
    "Only content words, no filler words."
)

PROMPT_VARIANTS = {
    'need': PROMPT_NEED,
    'question': PROMPT_QUESTION,
    'keywords': PROMPT_KEYWORDS,
}

print("Exp 05: LLM-Generated Surrogate Queries")
print(f"N: {N_SAMPLES}")
print(f"Generation model: {GEMMA_IT_NAME}")
print(f"Scoring model: {T5GEMMA_NAME}")
print(f"Prompt variants: {list(PROMPT_VARIANTS.keys())}")
""")


# ===== Cell 3: Load MS MARCO + reconstruct samples =====
code(r"""# Cell 3: Load MS MARCO and reconstruct same 500 samples as Exp 02
from lib.data import count_words
from datasets import load_dataset

# Load Exp 02 checkpoint for sample alignment verification
print("Loading Exp 02 checkpoint...")
exp02_ckpt = json.loads(EXP02_CHECKPOINT.read_text())
exp02_results = exp02_ckpt['results']
assert len(exp02_results) == N_SAMPLES, f"Expected {N_SAMPLES}, got {len(exp02_results)}"

# Reconstruct dataset samples
print("Loading MS MARCO to reconstruct samples...")
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

# Verify samples match Exp 02
for i in range(min(20, N_SAMPLES)):
    assert samples[i]['query'] == exp02_results[i]['query'], \
        f"Sample {i} query mismatch: {samples[i]['query'][:40]} != {exp02_results[i]['query'][:40]}"
passage_words = np.array([s['word_count'] for s in samples])
print(f"Verified: {N_SAMPLES} samples match Exp 02")
print(f"Document lengths: {passage_words.min()}-{passage_words.max()} words, "
      f"mean={passage_words.mean():.0f}")
""")


# ===== Cell 4: Phase 1 — LLM surrogate generation =====
code(r"""# Cell 4: Phase 1 — Generate surrogates with Gemma 2 9B-IT
# Skip if surrogates already cached

if SURROGATES_PATH.exists():
    print("Loading cached surrogates...")
    surrogates = json.loads(SURROGATES_PATH.read_text())
    # Verify alignment
    assert len(surrogates) == N_SAMPLES, f"Expected {N_SAMPLES}, got {len(surrogates)}"
    for i in range(min(10, N_SAMPLES)):
        assert surrogates[i]['query'][:50] == samples[i]['query'][:50], \
            f"Sample {i} query mismatch"
    print(f"Loaded {len(surrogates)} cached surrogates")
    print(f"Keys per sample: {list(surrogates[0].keys())}")
else:
    print(f"Loading {GEMMA_IT_NAME} for surrogate generation...")
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    HF_TOKEN = os.environ.get("HF_TOKEN")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    gen_tokenizer = AutoTokenizer.from_pretrained(GEMMA_IT_NAME, token=HF_TOKEN)
    gen_model = AutoModelForCausalLM.from_pretrained(
        GEMMA_IT_NAME, device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN,
    )
    gen_model.eval()
    GEN_DEVICE = next(gen_model.parameters()).device
    print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    def generate_surrogate(passage_text, prompt_text):
        # Truncate passage to first 150 words
        words = passage_text.split()[:150]
        truncated = " ".join(words)

        # Build chat message for Gemma IT
        messages = [
            {"role": "user",
             "content": f"{prompt_text}\n\nDocument:\n{truncated}"}
        ]
        chat_text = gen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = gen_tokenizer(chat_text, return_tensors="pt",
                               truncation=True, max_length=1024).to(GEN_DEVICE)

        with torch.no_grad():
            output_ids = gen_model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        # Decode only the new tokens
        new_tokens = output_ids[0, inputs['input_ids'].shape[1]:]
        raw_text = gen_tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Post-process: strip, take first line, remove quotes, truncate to 15 words
        cleaned = raw_text.strip().split("\n")[0].strip()
        cleaned = cleaned.strip('"').strip("'").strip()
        cleaned = " ".join(cleaned.split()[:15])
        return cleaned

    # Generate with checkpointing
    surrogates = []
    gen_ckpt_path = RESULTS_DIR / "gen_checkpoint.json"

    if gen_ckpt_path.exists():
        gen_ckpt = json.loads(gen_ckpt_path.read_text())
        if gen_ckpt.get('n_total') == N_SAMPLES:
            surrogates = gen_ckpt['surrogates']
            print(f"Resuming generation from {len(surrogates)}/{N_SAMPLES}")

    start_gen = len(surrogates)
    t0 = time.time()

    for i in tqdm(range(start_gen, N_SAMPLES), initial=start_gen, total=N_SAMPLES,
                  desc="Generating"):
        s = samples[i]
        entry = {'query': s['query']}

        for variant_name, prompt_text in PROMPT_VARIANTS.items():
            torch.manual_seed(SEED + i * 10 + hash(variant_name) % 100)
            text = generate_surrogate(s['passage'], prompt_text)
            entry[f'llm_{variant_name}'] = text

        surrogates.append(entry)

        if (i + 1) % 50 == 0 or i == N_SAMPLES - 1:
            gen_ckpt = {'n_total': N_SAMPLES, 'surrogates': surrogates,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
            gen_ckpt_path.write_text(json.dumps(gen_ckpt))
            elapsed = time.time() - t0
            done = i - start_gen + 1
            eta = (N_SAMPLES - i - 1) * elapsed / done if done > 0 else 0
            tqdm.write(f"  Gen checkpoint {i+1}/{N_SAMPLES} | {elapsed/60:.1f}m | ETA {eta/60:.1f}m")

    elapsed = time.time() - t0
    print(f"\nGeneration complete: {len(surrogates)} samples in {elapsed/60:.1f} min")

    # Save final surrogates
    SURROGATES_PATH.write_text(json.dumps(surrogates, indent=2))
    print(f"Saved surrogates to {SURROGATES_PATH}")

    # Free VRAM
    print("Freeing generation model VRAM...")
    mem_before = torch.cuda.memory_allocated() / 1e9
    del gen_model, gen_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    mem_after = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory: {mem_before:.2f} GB -> {mem_after:.2f} GB")
""")


# ===== Cell 5: Inspect surrogates =====
code(r"""# Cell 5: Inspect surrogates — examples, word counts, vocabulary overlap

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

print("=" * 70)
print("SURROGATE INSPECTION")
print("=" * 70)

# Show 5 examples
for idx in range(5):
    s = samples[idx]
    surr = surrogates[idx]
    print(f"\n--- Sample {idx} ---")
    print(f"  Passage: {s['passage'][:100]}...")
    print(f"  Query:   {s['query']}")
    for variant in PROMPT_VARIANTS:
        print(f"  llm_{variant}: {surr[f'llm_{variant}']}")

# Word count distributions
print(f"\n--- Word count distributions ---")
for variant in PROMPT_VARIANTS:
    wc = [len(surr[f'llm_{variant}'].split()) for surr in surrogates]
    wc = np.array(wc)
    print(f"  llm_{variant}: mean={wc.mean():.1f}, median={np.median(wc):.0f}, "
          f"range=[{wc.min()}, {wc.max()}], std={wc.std():.1f}")

# Oracle query word counts for comparison
oracle_wc = np.array([len(s['query'].split()) for s in samples])
print(f"  oracle query: mean={oracle_wc.mean():.1f}, median={np.median(oracle_wc):.0f}, "
      f"range=[{oracle_wc.min()}, {oracle_wc.max()}]")

# Vocabulary overlap with document (content words only)
print(f"\n--- Vocabulary overlap with document (content words) ---")
for variant in PROMPT_VARIANTS:
    overlaps = []
    for i in range(N_SAMPLES):
        doc_words = set(re.sub(r'[^\w\s]', '', samples[i]['passage'].lower()).split())
        doc_content = doc_words - STOP_WORDS
        surr_words = set(re.sub(r'[^\w\s]', '', surrogates[i][f'llm_{variant}'].lower()).split())
        surr_content = surr_words - STOP_WORDS
        if len(surr_content) > 0:
            overlap = len(surr_content & doc_content) / len(surr_content)
        else:
            overlap = 0.0
        overlaps.append(overlap)
    overlaps = np.array(overlaps)
    print(f"  llm_{variant}: mean={overlaps.mean():.3f}, median={np.median(overlaps):.3f}")

# Oracle overlap for comparison
oracle_overlaps = []
for i in range(N_SAMPLES):
    doc_words = set(re.sub(r'[^\w\s]', '', samples[i]['passage'].lower()).split())
    doc_content = doc_words - STOP_WORDS
    q_words = set(re.sub(r'[^\w\s]', '', samples[i]['query'].lower()).split())
    q_content = q_words - STOP_WORDS
    if len(q_content) > 0:
        overlap = len(q_content & doc_content) / len(q_content)
    else:
        overlap = 0.0
    oracle_overlaps.append(overlap)
print(f"  oracle query: mean={np.mean(oracle_overlaps):.3f}, "
      f"median={np.median(oracle_overlaps):.3f}")

# Stop word fraction in each variant
print(f"\n--- Stop word fraction ---")
for variant in PROMPT_VARIANTS:
    fracs = []
    for surr in surrogates:
        words = surr[f'llm_{variant}'].lower().split()
        if len(words) > 0:
            frac = sum(1 for w in words if w in STOP_WORDS) / len(words)
        else:
            frac = 0.0
        fracs.append(frac)
    fracs = np.array(fracs)
    print(f"  llm_{variant}: mean={fracs.mean():.3f}")

oracle_fracs = []
for s in samples:
    words = s['query'].lower().split()
    if len(words) > 0:
        frac = sum(1 for w in words if w in STOP_WORDS) / len(words)
    else:
        frac = 0.0
    oracle_fracs.append(frac)
print(f"  oracle query: mean={np.mean(oracle_fracs):.3f}")
""")


# ===== Cell 6: Phase 2 — Load T5Gemma + define helpers =====
code(r"""# Cell 6: Phase 2 — Load T5Gemma and define scoring helpers
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

from transformers import AutoProcessor, AutoModelForSeq2SeqLM

print(f"Loading {T5GEMMA_NAME}...")
processor = AutoProcessor.from_pretrained(T5GEMMA_NAME, token=HF_TOKEN)
tokenizer = processor.tokenizer

model = AutoModelForSeq2SeqLM.from_pretrained(
    T5GEMMA_NAME, device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN,
)
model.eval()

DEVICE = next(model.parameters()).device
print(f"Model loaded. dtype={next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def score_nll(encoder_text, answer_text, prefix_token_count=0, truncate=False):
    # Score NLL of answer given encoder text, with optional prefix truncation.
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
    # Count how many tokens the prefix occupies in [prefix + newline + document].
    full_text = prefix_text + "\n" + document_text
    full_ids = tokenizer(full_text, add_special_tokens=True, truncation=True,
                         max_length=2048).input_ids
    doc_ids = tokenizer(document_text, add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids
    return len(full_ids) - len(doc_ids)


print("Helpers defined.")
""")


# ===== Cell 7: Generate all 14 scoring conditions per sample =====
code(r"""# Cell 7: Generate all 14 scoring conditions per sample

# Pre-compute per-sample data
for i, s in enumerate(samples):
    surr = surrogates[i]
    query = s['query']
    passage = s['passage']

    # Random text from unrelated passage (same as Exp 02/2B/3F)
    other_idx = (i + N_SAMPLES // 2) % len(samples)
    other_words = samples[other_idx]['passage'].split()
    query_words = query.split()
    random_matched = " ".join(other_words[:len(query_words)])

    # Template surrogate: "What is [keyword]?"
    doc_words_clean = re.sub(r'[^\w\s]', '', passage.lower()).split()
    content = [w for w in doc_words_clean if w not in STOP_WORDS and len(w) > 2]
    if content:
        kw = Counter(content).most_common(1)[0][0]
    else:
        kw = "information"

    # Scrambled LLM need output (for decomposition)
    need_words = surr['llm_need'].split()
    rng = np.random.RandomState(SEED + i)
    shuffled_need = list(need_words)
    rng.shuffle(shuffled_need)

    # Store all prefix texts
    s['oracle_x1'] = query
    s['oracle_x4'] = " ".join([query] * 4)
    s['llm_need_x1'] = surr['llm_need']
    s['llm_need_x4'] = " ".join([surr['llm_need']] * 4)
    s['llm_question_x1'] = surr['llm_question']
    s['llm_question_x4'] = " ".join([surr['llm_question']] * 4)
    s['llm_keywords_x1'] = surr['llm_keywords']
    s['llm_keywords_x4'] = " ".join([surr['llm_keywords']] * 4)
    s['surr_template_x1'] = f"What is {kw}?"
    s['surr_template_x4'] = " ".join([f"What is {kw}?"] * 4)
    s['random_x1'] = random_matched
    s['random_x4'] = " ".join([random_matched] * 4)
    s['scrambled_llm_need_x4'] = " ".join([" ".join(shuffled_need)] * 4)

# Define all scoring conditions
COND_NAMES = [
    'bare',
    'oracle_x1_trunc',
    'oracle_x4_trunc',
    'llm_need_x1_trunc',
    'llm_need_x4_trunc',
    'llm_question_x1_trunc',
    'llm_question_x4_trunc',
    'llm_keywords_x1_trunc',
    'llm_keywords_x4_trunc',
    'surr_template_x1_trunc',
    'surr_template_x4_trunc',
    'random_x1_trunc',
    'random_x4_trunc',
    'scrambled_llm_need_x4_trunc',
]

print(f"Conditions ({len(COND_NAMES)}):")
for c in COND_NAMES:
    print(f"  {c}")

# Show example
ex = samples[0]
print(f"\nExample (sample 0):")
print(f"  Query:   {ex['query'][:80]}")
print(f"  Answer:  {ex['answer'][:80]}")
print(f"  Passage: {ex['passage'][:80]}...")
print()
for c in COND_NAMES:
    if c == 'bare':
        print(f"  {c:<35}: [document only]")
    else:
        key = c.replace('_trunc', '')
        text = ex[key]
        ptoks = count_prefix_tokens(text, ex['passage'])
        print(f"  {c:<35} ({ptoks:>3} toks): {str(text)[:55]}")

# Token count stats across first 50 samples
print(f"\nPrefix token counts (first 50 samples):")
for c in COND_NAMES:
    if c == 'bare':
        continue
    key = c.replace('_trunc', '')
    toks = [count_prefix_tokens(s[key], s['passage']) for s in samples[:50]]
    print(f"  {c:<35} mean={np.mean(toks):.1f}, range=[{min(toks)}, {max(toks)}]")
""")


# ===== Cell 8: Scoring loop with checkpointing =====
code(r"""# Cell 8: Scoring loop with checkpointing

print("=" * 70)
print("SCORING ALL CONDITIONS")
print("=" * 70)

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
          f"= {len(COND_NAMES) * N_SAMPLES} scorings")

t0 = time.time()

for i in tqdm(range(start_idx, N_SAMPLES), initial=start_idx, total=N_SAMPLES,
              desc="Scoring"):
    s = samples[i]
    result = {
        'query': s['query'],
        'answer': s['answer'],
        'passage_words': s['word_count'],
    }

    for cond in COND_NAMES:
        if cond == 'bare':
            nll = score_nll(s['passage'], s['answer'])
            result['nll_bare'] = nll
        else:
            key = cond.replace('_trunc', '')
            prefix = s[key]
            enc_text = prefix + "\n" + s['passage']
            ptoks = count_prefix_tokens(prefix, s['passage'])
            nll = score_nll(enc_text, s['answer'], ptoks, truncate=True)
            result[f'nll_{cond}'] = nll
            result[f'ptoks_{cond}'] = ptoks

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


# ===== Cell 9: Part 1 — LLM vs Heuristic =====
code(r"""# Cell 9: Part 1 — LLM vs Heuristic (the key question)

print("=" * 70)
print("PART 1: LLM vs HEURISTIC")
print("=" * 70)
print("Key question: does LLM-generated surrogate beat 'What is [keyword]?'\n")

# Extract NLLs
bare_nlls = np.array([r['nll_bare'] for r in results])
oracle_x1_nlls = np.array([r['nll_oracle_x1_trunc'] for r in results])
oracle_x4_nlls = np.array([r['nll_oracle_x4_trunc'] for r in results])

# Verify against Exp 02
exp02_bare = np.array([r['nll_bare'] for r in exp02_results])
exp02_oracle = np.array([r['nll_oracle_trunc'] for r in exp02_results])
bare_diff = np.abs(bare_nlls - exp02_bare)
oracle_diff = np.abs(oracle_x1_nlls - exp02_oracle)
print(f"--- Verification against Exp 02 ---")
print(f"  bare NLL max diff: {bare_diff.max():.6f} (should be ~0)")
print(f"  oracle_x1 vs Exp02 oracle max diff: {oracle_diff.max():.6f} (should be ~0)")

# All conditions table
oracle_x1_d = cohens_d(bare_nlls - oracle_x1_nlls)

all_conds = [
    ('oracle_x1_trunc', 'Real query x1 (upper bound)'),
    ('oracle_x4_trunc', 'Real query x4'),
    ('llm_need_x1_trunc', 'LLM need x1'),
    ('llm_need_x4_trunc', 'LLM need x4'),
    ('llm_question_x1_trunc', 'LLM question x1'),
    ('llm_question_x4_trunc', 'LLM question x4'),
    ('llm_keywords_x1_trunc', 'LLM keywords x1'),
    ('llm_keywords_x4_trunc', 'LLM keywords x4'),
    ('surr_template_x1_trunc', '"What is [kw]?" x1'),
    ('surr_template_x4_trunc', '"What is [kw]?" x4'),
    ('random_x1_trunc', 'Random matched x1'),
    ('random_x4_trunc', 'Random matched x4'),
    ('scrambled_llm_need_x4_trunc', 'Scrambled LLM need x4'),
]

# Bonferroni threshold
alpha_bonf = 0.05 / len(all_conds)

print(f"\n{'Condition':<40} {'NLL':>8} {'Delta':>8} {'d':>8} "
      f"{'Win%':>7} {'%Orc':>6} {'p':>12} {'sig':>5}")
print("-" * 100)

for cond, desc in all_conds:
    nlls = np.array([r[f'nll_{cond}'] for r in results])
    benefit = bare_nlls - nlls
    d = cohens_d(benefit)
    delta = benefit.mean()
    win = 100 * np.mean(benefit > 0)
    pct = d / oracle_x1_d * 100 if oracle_x1_d > 0 else 0
    _, p = stats.ttest_1samp(benefit, 0)
    sig = '***' if p < alpha_bonf / 10 else '**' if p < alpha_bonf else '*' if p < 0.05 else 'ns'
    print(f"  {desc:<38} {nlls.mean():>8.4f} {delta:>+8.4f} {d:>+8.3f} "
          f"{win:>6.1f}% {pct:>5.0f}% {p:>12.2e} {sig}")

print(f"\n  bare (lower bound): {bare_nlls.mean():.4f}")
print(f"  Bonferroni threshold: alpha={alpha_bonf:.4f}")

# Head-to-head: LLM need vs surr_template
print(f"\n--- Head-to-head: LLM need vs surr_template ---")
for rep in ['x1', 'x4']:
    need_nlls = np.array([r[f'nll_llm_need_{rep}_trunc'] for r in results])
    tmpl_nlls = np.array([r[f'nll_surr_template_{rep}_trunc'] for r in results])
    diff = tmpl_nlls - need_nlls  # positive = need is better
    d = cohens_d(diff)
    win = 100 * np.mean(diff > 0)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    winner = 'llm_need' if d > 0 else 'surr_template'
    print(f"  {rep}: llm_need vs surr_template: d={d:+.3f}, win={win:.1f}%, "
          f"p={p:.2e} {sig} [{winner}]")

# Head-to-head: LLM need vs random (is there semantic uplift?)
print(f"\n--- Head-to-head: LLM need vs random ---")
for rep in ['x1', 'x4']:
    need_nlls = np.array([r[f'nll_llm_need_{rep}_trunc'] for r in results])
    rand_nlls = np.array([r[f'nll_random_{rep}_trunc'] for r in results])
    diff = rand_nlls - need_nlls  # positive = need is better
    d = cohens_d(diff)
    win = 100 * np.mean(diff > 0)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    winner = 'llm_need' if d > 0 else 'random'
    print(f"  {rep}: llm_need vs random: d={d:+.3f}, win={win:.1f}%, "
          f"p={p:.2e} {sig} [{winner}]")
""")


# ===== Cell 10: Part 2 — Prompt Variant Comparison =====
code(r"""# Cell 10: Part 2 — Prompt Variant Comparison

print("=" * 70)
print("PART 2: PROMPT VARIANT COMPARISON")
print("=" * 70)
print("Which prompt style generates the best surrogates?\n")

for rep in ['x1', 'x4']:
    print(f"--- {rep} ---")
    variants = ['llm_need', 'llm_question', 'llm_keywords']

    # Pairwise comparisons
    for j in range(len(variants)):
        for k in range(j + 1, len(variants)):
            a_name = variants[j]
            b_name = variants[k]
            a_nlls = np.array([r[f'nll_{a_name}_{rep}_trunc'] for r in results])
            b_nlls = np.array([r[f'nll_{b_name}_{rep}_trunc'] for r in results])
            diff = b_nlls - a_nlls  # positive = A is better
            d = cohens_d(diff)
            win = 100 * np.mean(diff > 0)
            _, p = stats.ttest_1samp(diff, 0)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            winner = a_name if d > 0 else b_name
            print(f"  {a_name} vs {b_name}: d={d:+.3f}, win={win:.1f}%, "
                  f"p={p:.2e} {sig} [{winner}]")
    print()

# Do keywords underperform need? (tests stop-word hypothesis)
print("--- Stop-word hypothesis test ---")
for rep in ['x1', 'x4']:
    need_nlls = np.array([r[f'nll_llm_need_{rep}_trunc'] for r in results])
    kw_nlls = np.array([r[f'nll_llm_keywords_{rep}_trunc'] for r in results])
    diff = kw_nlls - need_nlls  # positive = need is better
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    if d > 0 and p < 0.05:
        verdict = "CONFIRMED: need > keywords (stop words help)"
    elif d < 0 and p < 0.05:
        verdict = "REFUTED: keywords > need (stop words hurt)"
    else:
        verdict = "INCONCLUSIVE: no significant difference"
    print(f"  {rep}: need vs keywords: d={d:+.3f}, p={p:.2e} {sig}")
    print(f"    {verdict}")

# Does question framing cause semantic interference?
print(f"\n--- Semantic interference test ---")
for rep in ['x1', 'x4']:
    need_nlls = np.array([r[f'nll_llm_need_{rep}_trunc'] for r in results])
    q_nlls = np.array([r[f'nll_llm_question_{rep}_trunc'] for r in results])
    diff = q_nlls - need_nlls  # positive = need is better
    d = cohens_d(diff)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    if d > 0 and p < 0.05:
        verdict = "CONFIRMED: need > question (question causes interference)"
    elif d < 0 and p < 0.05:
        verdict = "REFUTED: question > need"
    else:
        verdict = "INCONCLUSIVE: no significant difference"
    print(f"  {rep}: need vs question: d={d:+.3f}, p={p:.2e} {sig}")
    print(f"    {verdict}")
""")


# ===== Cell 11: Part 3 — Repetition + Decomposition =====
code(r"""# Cell 11: Part 3 — Repetition + Vocabulary/Semantics Decomposition

print("=" * 70)
print("PART 3: REPETITION + DECOMPOSITION")
print("=" * 70)

# x1 vs x4 for each condition
print("--- x1 vs x4 ---")
print(f"  {'Condition':<25} {'x1 d':>8} {'x4 d':>8} {'Uplift':>8} {'p':>12} {'sig':>5}")
print(f"  {'-'*65}")

paired_conds = [
    ('oracle', 'oracle_x1_trunc', 'oracle_x4_trunc'),
    ('llm_need', 'llm_need_x1_trunc', 'llm_need_x4_trunc'),
    ('llm_question', 'llm_question_x1_trunc', 'llm_question_x4_trunc'),
    ('llm_keywords', 'llm_keywords_x1_trunc', 'llm_keywords_x4_trunc'),
    ('surr_template', 'surr_template_x1_trunc', 'surr_template_x4_trunc'),
    ('random', 'random_x1_trunc', 'random_x4_trunc'),
]

for name, x1_cond, x4_cond in paired_conds:
    x1_nlls = np.array([r[f'nll_{x1_cond}'] for r in results])
    x4_nlls = np.array([r[f'nll_{x4_cond}'] for r in results])
    d_x1 = cohens_d(bare_nlls - x1_nlls)
    d_x4 = cohens_d(bare_nlls - x4_nlls)
    uplift = d_x4 - d_x1
    diff = x1_nlls - x4_nlls  # positive = x4 is better
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {name:<25} {d_x1:>+8.3f} {d_x4:>+8.3f} {uplift:>+8.3f} {p:>12.2e} {sig}")

# 3-Way decomposition for llm_need_x4
print(f"\n--- 3-Way Decomposition: llm_need_x4 ---")
print(f"  bare -> random_x4 -> scrambled_llm_need_x4 -> llm_need_x4")
print(f"    Structure:  bare -> random_x4")
print(f"    Vocabulary: random_x4 -> scrambled_llm_need_x4")
print(f"    Semantics:  scrambled_llm_need_x4 -> llm_need_x4\n")

random_x4_nlls = np.array([r['nll_random_x4_trunc'] for r in results])
scrambled_x4_nlls = np.array([r['nll_scrambled_llm_need_x4_trunc'] for r in results])
llm_need_x4_nlls = np.array([r['nll_llm_need_x4_trunc'] for r in results])

struct_comp = bare_nlls - random_x4_nlls
vocab_comp = random_x4_nlls - scrambled_x4_nlls
sem_comp = scrambled_x4_nlls - llm_need_x4_nlls
total_comp = bare_nlls - llm_need_x4_nlls

total_mean = total_comp.mean()

print(f"  {'Component':<20} {'Delta':>10} {'%total':>8} {'d':>8} {'p':>12} {'sig':>5}")
print(f"  {'-'*65}")

for label, comp in [('Structure', struct_comp), ('Vocabulary', vocab_comp),
                    ('Semantics', sem_comp)]:
    mu = comp.mean()
    pct = mu / total_mean * 100 if total_mean != 0 else 0
    d = cohens_d(comp)
    _, p = stats.ttest_1samp(comp, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {label:<20} {mu:>+10.4f} {pct:>7.1f}% {d:>+8.3f} {p:>12.2e} {sig}")

print(f"  {'TOTAL':<20} {total_mean:>+10.4f} {'100.0%':>8}")
residual = total_mean - (struct_comp.mean() + vocab_comp.mean() + sem_comp.mean())
print(f"\n  Decomposition residual: {residual:.6f} (should be ~0)")

# Compare with oracle decomposition from Exp 2B
print(f"\n--- Comparison with Exp 2B oracle decomposition (N=1) ---")
print(f"  Exp 2B (oracle, N=1): Structure=84.7%, Vocabulary=5.5%, Semantics=9.7%")
struct_pct = struct_comp.mean() / total_mean * 100 if total_mean != 0 else 0
vocab_pct = vocab_comp.mean() / total_mean * 100 if total_mean != 0 else 0
sem_pct = sem_comp.mean() / total_mean * 100 if total_mean != 0 else 0
print(f"  LLM need (x4):       Structure={struct_pct:.1f}%, Vocabulary={vocab_pct:.1f}%, "
      f"Semantics={sem_pct:.1f}%")
""")


# ===== Cell 12: Part 4 — Hardness Interaction =====
code(r"""# Cell 12: Part 4 — Hardness Interaction

print("=" * 70)
print("PART 4: HARDNESS INTERACTION")
print("=" * 70)
print("Does LLM surrogate help more for harder samples?\n")

quintile_bounds = np.percentile(bare_nlls, [20, 40, 60, 80])
quintiles = np.digitize(bare_nlls, quintile_bounds)
q_labels = ['Q1 easy', 'Q2', 'Q3', 'Q4', 'Q5 hard']

# Table: key conditions by quintile
key_conds = [
    ('oracle_x1_trunc', 'Oracle x1'),
    ('llm_need_x1_trunc', 'LLM need x1'),
    ('surr_template_x1_trunc', 'Template x1'),
    ('random_x1_trunc', 'Random x1'),
    ('llm_need_x4_trunc', 'LLM need x4'),
    ('surr_template_x4_trunc', 'Template x4'),
    ('random_x4_trunc', 'Random x4'),
]

print(f"  {'Quintile':<12} {'Bare NLL':>10}", end="")
for _, desc in key_conds:
    print(f"  {desc:>12}", end="")
print()
print(f"  {'-'*(14 + 14 * len(key_conds))}")

for q in range(5):
    mask = quintiles == q
    n = mask.sum()
    row = f"  {q_labels[q]:<12} {bare_nlls[mask].mean():>10.3f}"
    for cond, _ in key_conds:
        nlls_c = np.array([r[f'nll_{cond}'] for r in results])[mask]
        d = cohens_d(bare_nlls[mask] - nlls_c)
        row += f"  {d:>+12.3f}"
    print(row)

# Semantic gap by quintile: LLM need vs random
print(f"\n--- Semantic gap (LLM need - random) by quintile ---")
print(f"  {'Quintile':<12} {'x1 gap':>10} {'x4 gap':>10} {'x4 sem frac':>12}")
print(f"  {'-'*50}")

for q in range(5):
    mask = quintiles == q
    for rep in ['x1', 'x4']:
        need_nlls_q = np.array([r[f'nll_llm_need_{rep}_trunc'] for r in results])[mask]
        rand_nlls_q = np.array([r[f'nll_random_{rep}_trunc'] for r in results])[mask]
        if rep == 'x1':
            gap_x1 = (rand_nlls_q - need_nlls_q).mean()
        else:
            gap_x4 = (rand_nlls_q - need_nlls_q).mean()
            total_q = (bare_nlls[mask] - need_nlls_q).mean()
            sem_frac = gap_x4 / total_q * 100 if total_q > 0 else 0
    print(f"  {q_labels[q]:<12} {gap_x1:>+10.4f} {gap_x4:>+10.4f} {sem_frac:>11.1f}%")

# Head-to-head: LLM need vs surr_template by quintile
print(f"\n--- LLM need vs surr_template by quintile (x1) ---")
for q in range(5):
    mask = quintiles == q
    need_nlls_q = np.array([r['nll_llm_need_x1_trunc'] for r in results])[mask]
    tmpl_nlls_q = np.array([r['nll_surr_template_x1_trunc'] for r in results])[mask]
    diff = tmpl_nlls_q - need_nlls_q  # positive = need is better
    d = cohens_d(diff)
    win = 100 * np.mean(diff > 0)
    _, p = stats.ttest_1samp(diff, 0)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    winner = 'need' if d > 0 else 'template'
    print(f"  {q_labels[q]}: d={d:+.3f}, win={win:.1f}%, p={p:.2e} {sig} [{winner}]")

# Correlation: hardness vs LLM need semantic gap
print(f"\n--- Correlations ---")
for rep in ['x1', 'x4']:
    need_nlls_all = np.array([r[f'nll_llm_need_{rep}_trunc'] for r in results])
    rand_nlls_all = np.array([r[f'nll_random_{rep}_trunc'] for r in results])
    sem_gap = rand_nlls_all - need_nlls_all
    r_val, p_val = stats.pearsonr(bare_nlls, sem_gap)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"  {rep}: hardness vs LLM_need semantic gap: r={r_val:+.3f} (p={p_val:.2e}) {sig}")
""")


# ===== Cell 13: Synthesis + Save =====
code(r"""# Cell 13: Synthesis + Save

print("=" * 70)
print("SYNTHESIS: LLM-GENERATED SURROGATE RESULTS")
print("=" * 70)

oracle_x1_d = cohens_d(bare_nlls - oracle_x1_nlls)

# 1. All conditions summary
print(f"\n1. ALL CONDITIONS (d, % oracle):")
all_summary = []
for cond, desc in all_conds:
    nlls = np.array([r[f'nll_{cond}'] for r in results])
    d = cohens_d(bare_nlls - nlls)
    pct = d / oracle_x1_d * 100 if oracle_x1_d > 0 else 0
    all_summary.append((desc, d, pct))
    print(f"   {desc:<38} d={d:>+.3f} ({pct:>5.1f}% oracle)")

# 2. Key comparisons
print(f"\n2. KEY COMPARISONS:")

need_x1_d = cohens_d(bare_nlls - np.array([r['nll_llm_need_x1_trunc'] for r in results]))
tmpl_x1_d = cohens_d(bare_nlls - np.array([r['nll_surr_template_x1_trunc'] for r in results]))
rand_x1_d = cohens_d(bare_nlls - np.array([r['nll_random_x1_trunc'] for r in results]))

need_x4_d = cohens_d(bare_nlls - np.array([r['nll_llm_need_x4_trunc'] for r in results]))
tmpl_x4_d = cohens_d(bare_nlls - np.array([r['nll_surr_template_x4_trunc'] for r in results]))
rand_x4_d = cohens_d(bare_nlls - np.array([r['nll_random_x4_trunc'] for r in results]))

print(f"   LLM need x1 vs template x1:  {need_x1_d:+.3f} vs {tmpl_x1_d:+.3f} "
      f"(uplift: {need_x1_d - tmpl_x1_d:+.3f})")
print(f"   LLM need x4 vs template x4:  {need_x4_d:+.3f} vs {tmpl_x4_d:+.3f} "
      f"(uplift: {need_x4_d - tmpl_x4_d:+.3f})")
print(f"   LLM need x1 vs random x1:    {need_x1_d:+.3f} vs {rand_x1_d:+.3f} "
      f"(semantic uplift: {need_x1_d - rand_x1_d:+.3f})")

# 3. Decomposition summary
print(f"\n3. DECOMPOSITION (llm_need x4):")
print(f"   Structure:  {struct_pct:.1f}%")
print(f"   Vocabulary: {vocab_pct:.1f}%")
print(f"   Semantics:  {sem_pct:.1f}%")

# 4. Practical conclusion
print(f"\n{'='*70}")
print("CONCLUSIONS:")

# Is LLM surrogate worth it?
need_tmpl_x1_diff = need_x1_d - tmpl_x1_d
need_tmpl_x4_diff = need_x4_d - tmpl_x4_d

if need_tmpl_x1_diff > 0.05:
    print(f"  1. LLM surrogates provide MEANINGFUL uplift over template heuristic")
    print(f"     (d uplift: {need_tmpl_x1_diff:+.3f} at x1, {need_tmpl_x4_diff:+.3f} at x4)")
    practical = "POSITIVE_ROI"
elif need_tmpl_x1_diff > 0.02:
    print(f"  1. LLM surrogates provide SMALL uplift over template heuristic")
    print(f"     (d uplift: {need_tmpl_x1_diff:+.3f} at x1, {need_tmpl_x4_diff:+.3f} at x4)")
    practical = "MARGINAL_ROI"
else:
    print(f"  1. LLM surrogates provide NO meaningful uplift over template heuristic")
    print(f"     (d uplift: {need_tmpl_x1_diff:+.3f} at x1, {need_tmpl_x4_diff:+.3f} at x4)")
    print(f"     This confirms that surrogate generation has negative ROI.")
    practical = "NEGATIVE_ROI"

# Which prompt is best?
q_x1_d = cohens_d(bare_nlls - np.array([r['nll_llm_question_x1_trunc'] for r in results]))
kw_x1_d = cohens_d(bare_nlls - np.array([r['nll_llm_keywords_x1_trunc'] for r in results]))
best_variant = 'need' if need_x1_d >= max(q_x1_d, kw_x1_d) else \
               'question' if q_x1_d >= kw_x1_d else 'keywords'
print(f"  2. Best prompt variant: {best_variant}")
print(f"     need={need_x1_d:+.3f}, question={q_x1_d:+.3f}, keywords={kw_x1_d:+.3f}")

# Stop-word hypothesis
if need_x1_d > kw_x1_d + 0.02:
    print(f"  3. Stop-word hypothesis CONFIRMED: need > keywords by {need_x1_d - kw_x1_d:+.3f}")
elif kw_x1_d > need_x1_d + 0.02:
    print(f"  3. Stop-word hypothesis REFUTED: keywords > need by {kw_x1_d - need_x1_d:+.3f}")
else:
    print(f"  3. Stop-word hypothesis INCONCLUSIVE: need ~ keywords")

# Overall
print(f"\n  Practical recommendation: ", end="")
if practical == "POSITIVE_ROI":
    print(f"Use LLM-generated surrogates ('{best_variant}' prompt).")
elif practical == "MARGINAL_ROI":
    print(f"LLM surrogates offer marginal improvement. Template heuristic is simpler.")
else:
    print(f"Use 'What is [keyword]?' heuristic. LLM generation is not worth the cost.")
    print(f"  The mechanism is ~{struct_pct:.0f}% structural — any short prefix works.")

print(f"{'='*70}")

# Save results
final_results = {
    'experiment': 'exp05_llm_surrogates',
    'generation_model': GEMMA_IT_NAME,
    'scoring_model': T5GEMMA_NAME,
    'dataset': 'ms_marco_v1.1',
    'n_samples': N_SAMPLES,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'conditions': {},
    'decomposition': {
        'structure_pct': float(struct_pct),
        'vocabulary_pct': float(vocab_pct),
        'semantics_pct': float(sem_pct),
        'structure_d': float(cohens_d(struct_comp)),
        'vocabulary_d': float(cohens_d(vocab_comp)),
        'semantics_d': float(cohens_d(sem_comp)),
    },
    'key_comparisons': {
        'need_x1_vs_template_x1_uplift': float(need_tmpl_x1_diff),
        'need_x4_vs_template_x4_uplift': float(need_tmpl_x4_diff),
        'need_x1_vs_random_x1_uplift': float(need_x1_d - rand_x1_d),
    },
    'conclusion': {
        'practical': practical,
        'best_variant': best_variant,
    },
}

# Add per-condition results
for cond, desc in all_conds:
    nlls = np.array([r[f'nll_{cond}'] for r in results])
    benefit = bare_nlls - nlls
    d = cohens_d(benefit)
    _, p = stats.ttest_1samp(benefit, 0)
    final_results['conditions'][cond] = {
        'description': desc,
        'd': float(d),
        'mean_nll': float(nlls.mean()),
        'mean_delta': float(benefit.mean()),
        'pct_oracle': float(d / oracle_x1_d * 100) if oracle_x1_d > 0 else 0,
        'p': float(p),
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
out_path = "experiments/05/05_llm_surrogates.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(nb.cells)} cells)")
