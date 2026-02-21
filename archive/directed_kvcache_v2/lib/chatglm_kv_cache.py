"""
ChatGLM-6B specific KV cache utilities.

ChatGLM-6B is a prefix LM with bidirectional attention on prefix tokens
and causal attention on generation tokens. Key differences from Mistral:

1. KV cache layout: tuple of tuples, shape [seq_len, batch, num_heads, head_dim] (seq-first)
2. 2D RoPE: first 64 dims = absolute position, second 64 dims = block position
3. Inverted attention mask convention (True = masked)
4. Custom tokenizer with [gMASK] + BOS for generation
5. BOS token delimits prefix/generation boundary
"""

from typing import Tuple, Any
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from .config import ExperimentConfig


def _resolve_model_path(config: ExperimentConfig) -> str:
    """Resolve the model path, preferring a local patched copy."""
    import os
    local_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "chatglm-6b")
    if os.path.isdir(local_dir) and os.path.exists(os.path.join(local_dir, "config.json")):
        return local_dir
    return config.model_name


def load_chatglm(config: ExperimentConfig):
    """Load ChatGLM-6B model and tokenizer with optional 4-bit quantization.

    Args:
        config: Experiment configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    model_path = _resolve_model_path(config)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModel.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True,
        )
    else:
        model = AutoModel.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True,
        )

    model.eval()
    return model, tokenizer


def _tokenize_for_prefix_lm(text: str, tokenizer) -> torch.LongTensor:
    """Tokenize text for ChatGLM prefix LM mode.

    ChatGLM's tokenizer appends [gMASK] and <sop> (BOS) at the END of the
    sequence, not the beginning. The resulting layout is:
        [content_token_1, ..., content_token_N, gMASK, <sop>]
    The BOS position tells the model where the prefix ends and generation begins.

    Returns:
        input_ids tensor of shape [1, seq_len]
    """
    # ChatGLM tokenizer.encode adds [gMASK] and <sop> (BOS) automatically
    input_ids = tokenizer.encode(text, return_tensors="pt")
    return input_ids


def build_kv_cache_chatglm(
    context: str,
    model,
    tokenizer,
    config: ExperimentConfig,
) -> Tuple[int, Any]:
    """Build a KV cache from context text using ChatGLM.

    The entire context is placed in the prefix region (bidirectional attention).
    ChatGLM's get_masks() uses the BOS token position to construct the prefix LM mask.

    Args:
        context: Text to cache (all in prefix region with bidirectional attention)
        model: ChatGLM model
        tokenizer: ChatGLM tokenizer
        config: Experiment configuration

    Returns:
        Tuple of (context_length, past_key_values)
        past_key_values is a tuple of tuples with shape [seq, batch, heads, head_dim]
    """
    input_ids = _tokenize_for_prefix_lm(context, tokenizer).to(config.device)
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )

    return seq_len, outputs.past_key_values


def score_answer_with_cache_chatglm(
    past_key_values: Any,
    context_len: int,
    query_prompt: str,
    answer: str,
    model,
    tokenizer,
    config: ExperimentConfig,
) -> float:
    """Score an answer using a pre-built ChatGLM KV cache.

    Extends the cache with query+answer tokens in the generation region
    (causal attention). Computes mean NLL of answer tokens.

    Args:
        past_key_values: Pre-built KV cache (tuple of tuples, seq-first)
        context_len: Length of the cached context
        query_prompt: Query prompt to append
        answer: Answer text to score
        model: ChatGLM model
        tokenizer: ChatGLM tokenizer
        config: Experiment configuration

    Returns:
        Mean negative log-likelihood (lower is better)
    """
    # Tokenize query (no special tokens — these go in generation region)
    query_ids = tokenizer.encode(query_prompt, add_special_tokens=False, return_tensors="pt")
    if isinstance(query_ids, list):
        query_ids = torch.tensor([query_ids])
    query_ids = query_ids.to(config.device)
    query_len = query_ids.shape[1]

    # Tokenize answer
    answer_ids = tokenizer.encode(answer, add_special_tokens=False, return_tensors="pt")
    if isinstance(answer_ids, list):
        answer_ids = torch.tensor([answer_ids])
    answer_ids = answer_ids.to(config.device)
    answer_len = answer_ids.shape[1]

    if answer_len < 1:
        return 0.0

    # Combine query + answer for a single forward pass
    combined_ids = torch.cat([query_ids, answer_ids], dim=1)
    combined_len = combined_ids.shape[1]

    # Build position_ids for generation region
    # ChatGLM uses 2D position_ids: [batch, 2, seq_len]
    # Stream 0: absolute position = mask_position (index of [gMASK] = context_len - 2)
    # Stream 1: block positions (continue from 2, since <sop> already used block_pos=1)
    position_ids = _build_generation_position_ids(
        context_len, combined_len, config.device
    )

    with torch.no_grad():
        outputs = model(
            input_ids=combined_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )

    # Compute NLL over answer tokens only
    # The logits cover [query + answer] tokens
    # We want P(answer_token_i | everything before it)
    logits = outputs.logits  # [1, query_len + answer_len, vocab_size]

    # Answer tokens start at index query_len in the combined sequence
    # For token at position query_len + i, the prediction is at logits[:, query_len + i - 1, :]
    # So we need logits[:, query_len-1 : query_len+answer_len-1, :] to predict answer tokens
    answer_logits = logits[:, query_len - 1 : query_len + answer_len - 1, :].contiguous()
    answer_labels = answer_ids.contiguous()

    # Flatten for cross-entropy
    answer_logits_flat = answer_logits.view(-1, answer_logits.size(-1))
    answer_labels_flat = answer_labels.view(-1)

    loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
    nll = loss_fct(answer_logits_flat, answer_labels_flat).item()

    # Note: divides by answer_len (all N answer tokens), not answer_len-1.
    # The Mistral scorer divides by N-1 (skipping the first answer token).
    # This difference doesn't affect relative comparisons within this experiment.
    return nll / answer_len


def _build_generation_position_ids(
    context_len: int, gen_len: int, device: torch.device
) -> torch.LongTensor:
    """Build 2D position_ids for ChatGLM generation tokens.

    ChatGLM uses 2D positional encoding with shape [batch, 2, seq_len]:
    - Stream 0 (absolute position): all generation tokens get position = mask_position,
      which is the index of [gMASK] = context_len - 2 (since the layout is
      [content..., gMASK, <sop>] and context_len counts all tokens).
    - Stream 1 (block position): increments from 2, 3, ..., gen_len+1. Starts at 2
      because <sop> (the BOS token at end of prefix) already uses block_position=1
      during the initial forward pass.

    Args:
        context_len: Total number of prefix tokens already cached (including gMASK and <sop>)
        gen_len: Number of new generation tokens
        device: Target device

    Returns:
        position_ids of shape [1, 2, gen_len] (batch=1)
    """
    # Absolute positions: all at mask_position = context_len - 2 (index of [gMASK])
    mask_position = context_len - 2
    abs_positions = torch.full((1, gen_len), mask_position, dtype=torch.long, device=device)
    # Block positions: 2, 3, ..., gen_len+1 (continuing after <sop>'s block_pos=1)
    block_positions = torch.arange(2, gen_len + 2, dtype=torch.long, device=device).unsqueeze(0)
    # Stack into [1, 2, gen_len] — batch-first, matching ChatGLM's expected format
    position_ids = torch.stack([abs_positions, block_positions], dim=1)
    return position_ids


def build_suffix_kv_cache_chatglm(
    passage: str,
    suffix_text: str,
    model,
    tokenizer,
    config: ExperimentConfig,
    separator: str = "\n\n",
) -> Tuple[int, Any]:
    """Build a KV cache with passage + suffix, all in prefix region.

    With ChatGLM's prefix LM architecture, ALL tokens in the prefix region
    attend bidirectionally to each other. This means passage tokens CAN and DO
    attend to suffix tokens — the key difference from causal Mistral.

    Args:
        passage: Document text
        suffix_text: Suffix to append (e.g., surrogate query)
        model: ChatGLM model
        tokenizer: ChatGLM tokenizer
        config: Experiment configuration
        separator: Text between passage and suffix

    Returns:
        Tuple of (context_length, past_key_values)
    """
    full_context = passage + separator + suffix_text
    return build_kv_cache_chatglm(full_context, model, tokenizer, config)


def build_prefix_kv_cache_chatglm(
    surrogate: str,
    passage: str,
    model,
    tokenizer,
    config: ExperimentConfig,
    surrogate_template: str = "Related query: {surrogate}\n\n",
) -> Tuple[int, Any]:
    """Build a KV cache with surrogate prefix + passage, all in prefix region.

    Both surrogate and passage are in the bidirectional prefix region.

    Args:
        surrogate: Surrogate query text
        passage: Document text
        model: ChatGLM model
        tokenizer: ChatGLM tokenizer
        config: Experiment configuration
        surrogate_template: Template for surrogate prefix

    Returns:
        Tuple of (context_length, past_key_values)
    """
    prefix = surrogate_template.format(surrogate=surrogate)
    full_context = prefix + passage
    return build_kv_cache_chatglm(full_context, model, tokenizer, config)


def _get_chatglm_cache_seq_len(past_key_values) -> int:
    """Get sequence length from ChatGLM's tuple-of-tuples cache.

    ChatGLM cache shape: tuple of (key, value) per layer.
    Key shape: [seq_len, batch, num_heads, head_dim]
    """
    return past_key_values[0][0].shape[0]


def build_truncated_kv_cache_chatglm(
    surrogate: str,
    passage: str,
    model,
    tokenizer,
    config: ExperimentConfig,
    surrogate_template: str = "Related query: {surrogate}\n\n",
) -> Tuple[int, Any]:
    """Build truncated KV cache with 2D RoPE correction for ChatGLM.

    Process:
    1. Tokenize [surrogate_prefix + passage] → full prefix region
    2. Build full KV cache (bidirectional attention on everything)
    3. Truncate surrogate portion from KV cache
    4. Apply 2D RoPE correction to shift positions back

    Args:
        surrogate: Surrogate query text
        passage: Document text
        model: ChatGLM model
        tokenizer: ChatGLM tokenizer
        config: Experiment configuration
        surrogate_template: Template for surrogate prefix

    Returns:
        Tuple of (passage_len, corrected_cache)
    """
    prefix_text = surrogate_template.format(surrogate=surrogate)

    # Build full cache with surrogate + passage
    full_context = prefix_text + passage
    full_len, full_cache = build_kv_cache_chatglm(full_context, model, tokenizer, config)

    # The full tokenization includes special tokens ([gMASK], BOS)
    # added by _tokenize_for_prefix_lm. We need to find where the passage
    # starts in the full sequence.
    full_ids = _tokenize_for_prefix_lm(full_context, tokenizer).to(config.device)
    full_seq_len = full_ids.shape[1]

    # Tokenize just the passage to get its bare token count
    passage_ids = _tokenize_for_prefix_lm(passage, tokenizer).to(config.device)
    passage_seq_len = passage_ids.shape[1]

    # The surrogate offset is the number of tokens to remove.
    # full_seq_len = prefix_content + passage_content + special_tokens
    # passage_seq_len = passage_content + special_tokens
    # So surrogate_offset ≈ prefix_content token count.
    # Note: this assumes independent tokenization doesn't change token boundaries
    # at the prefix/passage junction. In practice, BPE merges across boundaries
    # could cause ±1 token differences, but this is rare and acceptable.
    surrogate_offset = full_seq_len - passage_seq_len

    if surrogate_offset <= 0:
        # No surrogate tokens to remove
        return full_len, full_cache

    # Truncate: keep only the passage portion (last passage_seq_len tokens)
    # But we also need to keep the special tokens at the start
    # ChatGLM special tokens: [gMASK] at some position, BOS (150004) at end of prefix
    # Actually, the tokenizer puts content tokens first, then [gMASK] and <sop> at the end
    # So the layout is: [prefix_content_tokens, passage_content_tokens, gMASK, BOS]

    # For ChatGLM, the special tokens [gMASK] and <sop> are at the END.
    # So truncation means removing the first surrogate_offset tokens.
    truncated_cache = _truncate_chatglm_cache(full_cache, surrogate_offset)

    # Apply 2D RoPE correction
    corrected_cache = correct_2d_rope_positions(truncated_cache, surrogate_offset, model)

    doc_len = full_seq_len - surrogate_offset
    return doc_len, corrected_cache


def _truncate_chatglm_cache(
    past_key_values: tuple,
    remove_first_n: int,
) -> tuple:
    """Remove the first N positions from a ChatGLM cache.

    ChatGLM cache: tuple of (key, value) per layer
    Key/Value shape: [seq_len, batch, num_heads, head_dim]

    Args:
        past_key_values: Original cache
        remove_first_n: Number of positions to remove from the start

    Returns:
        New cache tuple with first N positions removed
    """
    return tuple(
        (k[remove_first_n:].contiguous(), v[remove_first_n:].contiguous())
        for k, v in past_key_values
    )


def correct_2d_rope_positions(
    cache: tuple,
    offset: int,
    model,
) -> tuple:
    """Apply inverse 2D RoPE rotation to correct positions after truncation.

    ChatGLM uses 2D RoPE with rotary_dim=64 per stream on head_dim=128:
    - First 64 dims: absolute position encoding
    - Second 64 dims: block position encoding

    For prefix tokens (all in context), block positions are 0, so the second
    64 dims have no position-dependent rotation to correct.

    Only the first 64 dims (absolute positions) need correction: shift back by offset.

    Args:
        cache: ChatGLM cache tuple
        offset: Number of positions to shift back
        model: ChatGLM model (for RoPE config)

    Returns:
        Corrected cache tuple (new tensors)
    """
    if offset == 0:
        return cache

    # ChatGLM RoPE parameters
    config = model.config if hasattr(model, 'config') else model
    hidden_size = getattr(config, 'hidden_size', 4096)
    num_heads = getattr(config, 'num_attention_heads', 32)
    head_dim = hidden_size // num_heads  # 128 for ChatGLM-6B
    rotary_dim = head_dim // 2  # 64 — each stream uses half of head_dim

    rope_theta = 10000.0

    # Build correction for absolute position stream (first rotary_dim dims)
    inv_freq = 1.0 / (rope_theta ** (
        torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim
    ))
    angles = -offset * inv_freq  # (rotary_dim // 2,)
    # Duplicate for rotate_half convention
    emb = torch.cat((angles, angles), dim=-1)  # (rotary_dim,)
    cos_correction = emb.cos()
    sin_correction = emb.sin()

    corrected_layers = []
    for key, value in cache:
        # key shape: [seq_len, batch, num_heads, head_dim]
        device = key.device
        dtype = key.dtype
        c = cos_correction.to(device=device, dtype=dtype)
        s = sin_correction.to(device=device, dtype=dtype)

        # Only correct the first rotary_dim dimensions (absolute position stream)
        key_rot = key[..., :rotary_dim]  # [seq, batch, heads, rotary_dim]
        key_pass = key[..., rotary_dim:]  # [seq, batch, heads, rotary_dim] — block positions, no correction

        # Apply RoPE(-offset) to absolute position stream
        corrected_rot = key_rot * c + _rotate_half(key_rot) * s

        corrected_key = torch.cat([corrected_rot, key_pass], dim=-1)
        corrected_layers.append((corrected_key, value))

    return tuple(corrected_layers)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Identical to HuggingFace's rotate_half: split first/second half."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
