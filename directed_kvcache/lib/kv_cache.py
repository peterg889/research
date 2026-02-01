"""
KV Cache utilities for building, manipulating, and scoring with KV caches.
"""

from typing import Tuple, Any, Optional
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .config import ExperimentConfig


def build_kv_cache(
    context: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ExperimentConfig
) -> Tuple[int, Any]:
    """
    Build a KV cache from the given context.

    Args:
        context: The text to cache (e.g., document, or surrogate + document)
        model: The language model
        tokenizer: The tokenizer
        config: Experiment configuration

    Returns:
        Tuple of (context_length, past_key_values)
    """
    context_encoding = tokenizer(
        context, return_tensors="pt", add_special_tokens=True,
        padding=False, truncation=False
    )
    context_ids = context_encoding['input_ids'].to(config.device)

    with torch.no_grad():
        outputs = model(
            input_ids=context_ids,
            attention_mask=torch.ones_like(context_ids),
            use_cache=True,
            return_dict=True
        )

    return context_ids.shape[1], outputs.past_key_values


def extract_and_truncate_cache(past_key_values: Any, keep_last_n: int) -> DynamicCache:
    """
    Extract tensors from cache, truncate to last N positions, return new DynamicCache.

    This is used for the "surrogate-influenced document embeddings" approach where
    we generate the cache with surrogate context but then discard the surrogate portion.

    Args:
        past_key_values: The original KV cache (DynamicCache or tuple format)
        keep_last_n: Number of positions to keep (from the end)

    Returns:
        New DynamicCache with only the last keep_last_n positions
    """
    # Get the original cache as legacy tuple format
    if hasattr(past_key_values, 'to_legacy_cache'):
        legacy_cache = past_key_values.to_legacy_cache()
    elif isinstance(past_key_values, (tuple, list)):
        legacy_cache = past_key_values
    else:
        # Try to convert by iterating
        legacy_cache = tuple(past_key_values)

    # Create new DynamicCache with truncated tensors
    new_cache = DynamicCache()

    for layer_idx, layer_kv in enumerate(legacy_cache):
        key, value = layer_kv[0], layer_kv[1]
        # Shape is (batch, num_heads, seq_len, head_dim)
        truncated_key = key[:, :, -keep_last_n:, :].contiguous()
        truncated_value = value[:, :, -keep_last_n:, :].contiguous()
        # Use update() to properly add to the cache
        new_cache.update(truncated_key, truncated_value, layer_idx)

    return new_cache


def extract_and_truncate_cache_with_bos(past_key_values: Any, doc_len: int) -> DynamicCache:
    """
    Extract BOS (position 0) + last doc_len positions from cache.

    This preserves the BOS token that the model expects at the start of
    every sequence while discarding the surrogate prefix tokens.

    Args:
        past_key_values: The original KV cache
        doc_len: Number of document token positions to keep from the end

    Returns:
        New DynamicCache with [BOS] + [last doc_len positions]
    """
    if hasattr(past_key_values, 'to_legacy_cache'):
        legacy_cache = past_key_values.to_legacy_cache()
    elif isinstance(past_key_values, (tuple, list)):
        legacy_cache = past_key_values
    else:
        legacy_cache = tuple(past_key_values)

    new_cache = DynamicCache()

    for layer_idx, layer_kv in enumerate(legacy_cache):
        key, value = layer_kv[0], layer_kv[1]
        # Shape: (batch, num_heads, seq_len, head_dim)
        bos_key = key[:, :, :1, :]
        doc_key = key[:, :, -doc_len:, :]
        combined_key = torch.cat([bos_key, doc_key], dim=2).contiguous()

        bos_value = value[:, :, :1, :]
        doc_value = value[:, :, -doc_len:, :]
        combined_value = torch.cat([bos_value, doc_value], dim=2).contiguous()

        new_cache.update(combined_key, combined_value, layer_idx)

    return new_cache


def correct_rope_positions_with_bos(
    cache: DynamicCache,
    offset: int,
    model: AutoModelForCausalLM,
) -> DynamicCache:
    """
    Apply inverse RoPE rotation to document positions only, leaving BOS at position 0.

    The cache has shape [BOS, doc_0, doc_1, ...]. BOS is already at position 0
    and needs no correction. Document tokens at positions 1..N need to be shifted
    back by `offset` positions.

    Args:
        cache: DynamicCache with [BOS] + [document tokens]
        offset: Number of surrogate positions to shift back
        model: The model (for RoPE config)

    Returns:
        The same cache, modified in-place
    """
    if offset == 0:
        return cache

    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    rope_theta = getattr(config, 'rope_theta', 10000.0)

    cos_a, sin_a = _build_rope_correction(offset, head_dim, rope_theta)

    for layer_idx in range(len(cache)):
        keys = cache.key_cache[layer_idx] if hasattr(cache, 'key_cache') else cache.layers[layer_idx].keys
        device = keys.device
        dtype = keys.dtype

        c = cos_a.to(device=device, dtype=dtype)
        s = sin_a.to(device=device, dtype=dtype)

        # Only correct document tokens (index 1 onward), leave BOS (index 0) untouched
        doc_keys = keys[:, :, 1:, :]
        corrected = doc_keys * c + _rotate_half(doc_keys) * s

        if hasattr(cache, 'key_cache'):
            cache.key_cache[layer_idx] = torch.cat([keys[:, :, :1, :], corrected], dim=2)
        else:
            cache.layers[layer_idx].keys = torch.cat([keys[:, :, :1, :], corrected], dim=2)

    return cache


def build_truncated_kv_cache(
    surrogate: str,
    document: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ExperimentConfig
) -> Tuple[int, DynamicCache]:
    """
    Build a KV cache with surrogate context, then TRUNCATE to keep only document portion.

    This implements the 'surrogate-influenced document embeddings' approach:
    1. Concatenate: [surrogate_prefix][document]
    2. Generate full KV cache (document attends to surrogate)
    3. Discard surrogate KV entries, keep only document's KV cache

    Args:
        surrogate: The surrogate query text
        document: The document text
        model: The language model
        tokenizer: The tokenizer
        config: Experiment configuration

    Returns:
        Tuple of (document_length, truncated_DynamicCache)
    """
    # Build the surrogate prefix and document text
    surrogate_prefix = f"This document may be relevant to queries like: {surrogate}\n\n"
    document_text = f"Document:\n{document}"

    # Tokenize prefix with BOS to get exact prefix length
    prefix_encoding = tokenizer(
        surrogate_prefix, return_tensors="pt", add_special_tokens=True,
        padding=False, truncation=False
    )
    prefix_len = prefix_encoding['input_ids'].shape[1]

    # Tokenize full context
    full_context = surrogate_prefix + document_text
    full_encoding = tokenizer(
        full_context, return_tensors="pt", add_special_tokens=True,
        padding=False, truncation=False
    )
    full_ids = full_encoding['input_ids'].to(config.device)
    doc_len = full_ids.shape[1] - prefix_len

    # Generate full KV cache
    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids),
            use_cache=True,
            return_dict=True
        )

    full_cache = outputs.past_key_values

    # Truncate: keep BOS + document portion
    truncated_cache = extract_and_truncate_cache_with_bos(full_cache, doc_len)
    keep_len = 1 + doc_len

    return keep_len, truncated_cache


def score_answer_with_cache(
    past_key_values: Any,
    context_len: int,
    query_prompt: str,
    answer: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ExperimentConfig
) -> float:
    """
    Score an answer using a pre-built KV cache.

    This extends the cache with the query, then computes the negative log-likelihood
    of generating the answer tokens.

    Args:
        past_key_values: Pre-built KV cache
        context_len: Length of the cached context
        query_prompt: The query prompt to append
        answer: The answer to score
        model: The language model
        tokenizer: The tokenizer
        config: Experiment configuration

    Returns:
        Mean negative log-likelihood (lower is better)
    """
    # Tokenize query
    query_encoding = tokenizer(
        query_prompt, return_tensors="pt", add_special_tokens=False,
        padding=False, truncation=False
    )
    query_ids = query_encoding['input_ids'].to(config.device)
    query_len = query_ids.shape[1]

    # Tokenize answer
    answer_encoding = tokenizer(
        answer, return_tensors="pt", add_special_tokens=False,
        padding=False, truncation=False
    )
    answer_ids = answer_encoding['input_ids'].to(config.device)
    answer_len = answer_ids.shape[1]

    # Extend cache with query
    combined_len = context_len + query_len
    attention_mask = torch.ones((1, combined_len), device=config.device)

    with torch.no_grad():
        query_outputs = model(
            input_ids=query_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )
        extended_cache = query_outputs.past_key_values

    # Score answer
    combined_len_final = context_len + query_len + answer_len
    attention_mask_final = torch.ones((1, combined_len_final), device=config.device)

    with torch.no_grad():
        answer_outputs = model(
            input_ids=answer_ids,
            attention_mask=attention_mask_final,
            past_key_values=extended_cache,
            use_cache=False,
            return_dict=True
        )

    # Compute NLL
    logits = answer_outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = answer_ids[:, 1:].contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
    nll = loss_fct(shift_logits, shift_labels).item()

    num_scored = answer_len - 1
    return nll / num_scored if num_scored > 0 else 0.0


def build_suffix_kv_cache(
    passage: str,
    suffix_text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ExperimentConfig,
    separator: str = "\n\nRelated question: ",
) -> Tuple[int, Any]:
    """
    Build a KV cache with a suffix appended AFTER the passage.

    In a causal model, passage tokens never attend to suffix tokens, so
    passage KV entries are byte-identical to the bare cache. Any scoring
    improvement must come from query tokens attending to suffix KV entries
    that have "read" the full passage — a clean semantic signal.

    Args:
        passage: The document/passage text
        suffix_text: Text to append as suffix (e.g., surrogate query)
        model: The language model
        tokenizer: The tokenizer
        config: Experiment configuration
        separator: Text between passage and suffix

    Returns:
        Tuple of (context_length, past_key_values)
    """
    full_context = passage + separator + suffix_text
    return build_kv_cache(full_context, model, tokenizer, config)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Identical to HuggingFace's rotate_half: split first/second half."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _build_rope_correction(offset: int, head_dim: int, rope_theta: float):
    """Build cos/sin tensors for RoPE position correction.

    Returns cos, sin each of shape (head_dim,), matching HF's convention
    where frequencies are duplicated across both halves.
    """
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = -offset * inv_freq  # (head_dim // 2,)
    emb = torch.cat((angles, angles), dim=-1)  # (head_dim,) — HF duplication
    return emb.cos(), emb.sin()


def correct_rope_positions(
    cache: DynamicCache,
    offset: int,
    model: AutoModelForCausalLM,
) -> DynamicCache:
    """
    Apply inverse RoPE rotation to shift cached key positions by -offset.

    When a KV cache is built from [surrogate (S tokens)][document (D tokens)],
    document token i has its key stored as RoPE(K_i, S+i). After truncating
    the surrogate entries, queries at position j compute attention using
    relative position j - (S+i) instead of the correct j - i.

    This function corrects by applying RoPE(-offset) to each key, yielding
    RoPE(K_i, S+i-S) = RoPE(K_i, i).

    Uses the same rotate_half convention as HuggingFace transformers:
        RoPE(k, pos) = k * cos(pos) + rotate_half(k) * sin(pos)

    Args:
        cache: DynamicCache with key vectors to correct
        offset: Number of positions to shift back (typically surrogate length)
        model: The model (used to read RoPE config parameters)

    Returns:
        The same DynamicCache, modified in-place (also returned for convenience)
    """
    if offset == 0:
        return cache

    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    rope_theta = getattr(config, 'rope_theta', 10000.0)

    cos_a, sin_a = _build_rope_correction(offset, head_dim, rope_theta)

    for layer_idx in range(len(cache)):
        keys = cache.key_cache[layer_idx] if hasattr(cache, 'key_cache') else cache.layers[layer_idx].keys
        device = keys.device
        dtype = keys.dtype

        c = cos_a.to(device=device, dtype=dtype)
        s = sin_a.to(device=device, dtype=dtype)

        # Apply RoPE(-offset): same formula as forward, with negative angle baked into cos/sin
        new_keys = keys * c + _rotate_half(keys) * s
        if hasattr(cache, 'key_cache'):
            cache.key_cache[layer_idx] = new_keys
        else:
            cache.layers[layer_idx].keys = new_keys

    return cache


def build_truncated_kv_cache_corrected(
    surrogate: str,
    document: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ExperimentConfig,
    surrogate_prefix_template: Optional[str] = None,
    document_template: Optional[str] = None,
) -> Tuple[int, DynamicCache]:
    """
    Build a truncated KV cache with RoPE position correction.

    Like build_truncated_kv_cache but applies inverse RoPE rotation after
    truncation so that document key positions are correct (as if the document
    started at position 0).

    Args:
        surrogate: The surrogate query text
        document: The document text
        model: The language model
        tokenizer: The tokenizer
        config: Experiment configuration
        surrogate_prefix_template: Optional custom prefix template (must contain {surrogate}).
            Defaults to "This document may be relevant to queries like: {surrogate}\n\n"
        document_template: Optional custom document template (must contain {document}).
            Defaults to "Document:\n{document}"

    Returns:
        Tuple of (document_length, corrected_DynamicCache)
    """
    if surrogate_prefix_template is None:
        surrogate_prefix = f"This document may be relevant to queries like: {surrogate}\n\n"
    else:
        surrogate_prefix = surrogate_prefix_template.format(surrogate=surrogate)

    if document_template is None:
        document_text = f"Document:\n{document}"
    else:
        document_text = document_template.format(document=document)

    # Tokenize surrogate prefix with BOS to get exact prefix length.
    # This avoids tokenizer boundary mismatch from tokenizing the document
    # in isolation (which can produce different tokens at the join point).
    prefix_encoding = tokenizer(
        surrogate_prefix, return_tensors="pt", add_special_tokens=True,
        padding=False, truncation=False
    )
    prefix_len = prefix_encoding['input_ids'].shape[1]  # includes BOS

    # Tokenize full context: [BOS] + surrogate_prefix + document_text
    full_context = surrogate_prefix + document_text
    full_encoding = tokenizer(
        full_context, return_tensors="pt", add_special_tokens=True,
        padding=False, truncation=False
    )
    full_ids = full_encoding['input_ids'].to(config.device)
    full_len = full_ids.shape[1]

    # Document portion length = total - prefix (including BOS)
    doc_len = full_len - prefix_len

    # Generate full KV cache
    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids),
            use_cache=True,
            return_dict=True
        )

    # Keep BOS + document portion: first token (BOS) + last doc_len tokens
    # This ensures the truncated cache structurally matches baseline which
    # also starts with BOS.
    keep_len = 1 + doc_len  # BOS + document tokens
    truncated_cache = extract_and_truncate_cache_with_bos(
        outputs.past_key_values, doc_len
    )

    # The surrogate-only offset is prefix_len - 1 (prefix tokens minus BOS,
    # since BOS stays at position 0 and doesn't need correction).
    # We correct the document tokens (positions prefix_len .. full_len-1)
    # so they behave as if at positions 1 .. doc_len.
    surrogate_offset = prefix_len - 1
    correct_rope_positions_with_bos(truncated_cache, surrogate_offset, model)

    return keep_len, truncated_cache
