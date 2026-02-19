import os, sys, random, torch
import torch.nn.functional as F
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

# Load neural-bridge/rag-dataset-12000
print("Loading neural-bridge/rag-dataset-12000...")
ds = load_dataset("neural-bridge/rag-dataset-12000", split="train")
print(f"Total samples: {len(ds)}")
print(f"Fields: {list(ds.features.keys())}")

# Show a sample to understand structure
print(f"\nSample 0: {ds[0]}")

# Filter to queries >= 15 words
samples = []
for row in ds:
    # Try common field names
    q = row.get("question", row.get("query", row.get("instruction", "")))
    doc = row.get("context", row.get("document", row.get("passage", "")))
    answer = row.get("answer", row.get("response", row.get("output", "")))
    
    if not q or not doc or not answer:
        continue
    if len(q.split()) >= 15 and len(answer.split()) >= 5:
        samples.append({"query": q, "document": doc, "answer": answer})

print(f"\nFiltered samples (q>=15w, a>=5w): {len(samples)}")
if samples:
    mean_q = sum(len(s['query'].split()) for s in samples) / len(samples)
    mean_d = sum(len(s['document'].split()) for s in samples) / len(samples)
    mean_a = sum(len(s['answer'].split()) for s in samples) / len(samples)
    print(f"Query lengths: mean={mean_q:.1f}w")
    print(f"Doc lengths: mean={mean_d:.1f}w")
    print(f"Answer lengths: mean={mean_a:.1f}w")

    # Sample 3 examples
    for i, s in enumerate(random.sample(samples, min(3, len(samples)))):
        qw = len(s['query'].split())
        dw = len(s['document'].split())
        aw = len(s['answer'].split())
        print(f"\nExample {i+1}:")
        print(f"  Q ({qw}w): {s['query'][:200]}")
        print(f"  D ({dw}w): {s['document'][:150]}...")
        print(f"  A ({aw}w): {s['answer'][:150]}...")

# Load model
print("\nLoading T5Gemma...")
processor = AutoProcessor.from_pretrained("google/t5gemma-2-4b-4b", token=HF_TOKEN)
tokenizer = processor.tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/t5gemma-2-4b-4b", device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN)
model.eval()
DEVICE = next(model.parameters()).device

def score_nll(encoder_text, answer_text):
    enc_ids = tokenizer(encoder_text, return_tensors="pt",
                        add_special_tokens=True, truncation=True,
                        max_length=2048).input_ids.to(DEVICE)
    enc_mask = torch.ones(1, enc_ids.shape[1], device=DEVICE, dtype=torch.long)
    with torch.no_grad():
        encoder_outputs = model.get_encoder()(input_ids=enc_ids, attention_mask=enc_mask)
    ans_ids = tokenizer(answer_text, return_tensors="pt",
                        add_special_tokens=False, truncation=True,
                        max_length=256).input_ids.to(DEVICE)
    if ans_ids.shape[1] == 0:
        return 0.0
    with torch.no_grad():
        outputs = model(encoder_outputs=encoder_outputs, attention_mask=enc_mask, labels=ans_ids)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[0].gather(1, ans_ids[0].unsqueeze(1)).squeeze(1)
    mean_nll = -token_log_probs.mean().item()
    del encoder_outputs, outputs, logits, log_probs
    return mean_nll

# Pre-screen: 20 random samples, bare NLL
if samples:
    print("\n=== CEILING PRE-SCREEN (20 samples) ===")
    screen_samples = random.sample(samples, min(20, len(samples)))
    bare_nlls = []
    for i, s in enumerate(screen_samples):
        nll = score_nll(s["document"], s["answer"])
        bare_nlls.append(nll)
        qw = len(s['query'].split())
        dw = len(s['document'].split())
        aw = len(s['answer'].split())
        print(f"  [{i+1}/20] bare NLL = {nll:.4f} (q={qw}w, d={dw}w, a={aw}w)")

    pct_floor = sum(1 for n in bare_nlls if n < 0.05) / len(bare_nlls)
    mean_nll = sum(bare_nlls) / len(bare_nlls)
    print(f"\nMean bare NLL: {mean_nll:.4f}")
    print(f"% at floor (<0.05): {pct_floor:.0%}")
    print(f"Min: {min(bare_nlls):.4f}, Max: {max(bare_nlls):.4f}")
    if pct_floor > 0.3:
        print("WARNING: CEILING EFFECT — skip this dataset")
    else:
        print("PASS — no ceiling effect, proceed with experiment")

    # Also score 5 samples with oracle to check headroom
    print("\n=== HEADROOM CHECK (5 samples) ===")
    for i, s in enumerate(screen_samples[:5]):
        bare = score_nll(s["document"], s["answer"])
        oracle = score_nll(s["query"] + " " + s["document"], s["answer"])
        delta = bare - oracle
        print(f"  [{i+1}/5] bare={bare:.4f}, oracle={oracle:.4f}, delta={delta:+.4f}")

print("\nDone.")
