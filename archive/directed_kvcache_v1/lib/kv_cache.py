"""
KV Cache utilities for building, manipulating, and scoring with KV caches.

This module provides:
- Cache building (with optional custom attention masks)
- Cache truncation and manipulation
- RoPE position correction
- Answer scoring with pre-built caches
- Cache copying and hybrid cache construction
"""

from typing import Tuple, Any, Optional
import torch
import torch.nn.functional as F
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


def build_cache_with_mask(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[DynamicCache, int]:
    """
    Build KV cache with optional custom attention mask.

    This is a lower-level function that allows custom attention patterns
    (e.g., block-diagonal attention for prefix repetitions).

    Args:
        text: The text to encode and cache
        tokenizer: The tokenizer
        model: The language model
        attention_mask: Optional 4D attention mask (1, 1, seq_len, seq_len).
            If None, uses standard causal attention.
            For additive masks: 0 = attend, -inf = block.
            The mask will be cast to model.dtype automatically.

    Returns:
        Tuple of (DynamicCache, sequence_length)

    Example:
        # Standard causal attention
        cache, seq_len = build_cache_with_mask(text, tokenizer, model)

        # Block-diagonal attention (prefix reps can't see each other)
        from lib.block_attention import create_block_diagonal_prefix_mask
        mask = create_block_diagonal_prefix_mask(prefix_len, n_reps, passage_len,
                                                  dtype=model.dtype, device=model.device)
        cache, seq_len = build_cache_with_mask(text, tokenizer, model, mask)
    """
    ids = tokenizer.encode(text, return_tensors='pt').to(model.device)
    seq_len = ids.shape[1]

    with torch.no_grad():
        if attention_mask is not None:
            # Cast mask to model's dtype for SDPA compatibility
            attention_mask = attention_mask.to(device=model.device, dtype=model.dtype)
            out = model(ids, attention_mask=attention_mask, use_cache=True)
        else:
            out = model(ids, use_cache=True)

    return out.past_key_values, seq_len


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
    rope_theta = _get_rope_theta(config)

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


def _ensure_dynamic_cache(cache: Any) -> DynamicCache:
    """Convert any cache format (tuple, list, DynamicCache) to DynamicCache.

    Returns a DynamicCache regardless of input format.
    """
    if isinstance(cache, DynamicCache):
        return cache

    # Convert from tuple/list/legacy format
    if hasattr(cache, 'to_legacy_cache'):
        legacy = cache.to_legacy_cache()
    elif isinstance(cache, (tuple, list)):
        legacy = cache
    else:
        legacy = tuple(cache)

    new_cache = DynamicCache()
    for layer_idx, layer_kv in enumerate(legacy):
        new_cache.update(layer_kv[0].clone(), layer_kv[1].clone(), layer_idx)
    return new_cache


def _get_cache_keys(cache: DynamicCache, layer_idx: int) -> torch.Tensor:
    """Get key tensor from a DynamicCache layer (handles both API versions)."""
    if hasattr(cache, 'key_cache'):
        return cache.key_cache[layer_idx]
    return cache.layers[layer_idx].keys


def _set_cache_keys(cache: DynamicCache, layer_idx: int, keys: torch.Tensor):
    """Set key tensor on a DynamicCache layer (handles both API versions)."""
    if hasattr(cache, 'key_cache'):
        cache.key_cache[layer_idx] = keys
    else:
        cache.layers[layer_idx].keys = keys


def _get_cache_values(cache: DynamicCache, layer_idx: int) -> torch.Tensor:
    """Get value tensor from a DynamicCache layer (handles both API versions)."""
    if hasattr(cache, 'value_cache'):
        return cache.value_cache[layer_idx]
    return cache.layers[layer_idx].values


def _set_cache_values(cache: DynamicCache, layer_idx: int, values: torch.Tensor):
    """Set value tensor on a DynamicCache layer (handles both API versions)."""
    if hasattr(cache, 'value_cache'):
        cache.value_cache[layer_idx] = values
    else:
        cache.layers[layer_idx].values = values


def deepcopy_cache(cache: Any) -> DynamicCache:
    """
    Create a deep copy of a KV cache.

    This function handles different transformers versions and cache formats.
    Use this instead of copy.deepcopy() for DynamicCache objects, as the
    iteration API varies across versions.

    Args:
        cache: A DynamicCache, tuple, or list of (key, value) tensors

    Returns:
        New DynamicCache with cloned tensors (independent from original)
    """
    cache = _ensure_dynamic_cache(cache)
    new_cache = DynamicCache()
    n_layers = len(cache)

    for layer_idx in range(n_layers):
        k = _get_cache_keys(cache, layer_idx).clone()
        v = _get_cache_values(cache, layer_idx).clone()
        new_cache.update(k, v, layer_idx)

    return new_cache


def build_hybrid_cache(keys_source: Any, values_source: Any) -> DynamicCache:
    """
    Build a hybrid cache mixing keys from one source and values from another.

    Both caches must have the same number of layers and matching tensor shapes.

    Args:
        keys_source: Cache to take key tensors from
        values_source: Cache to take value tensors from

    Returns:
        New DynamicCache with keys from keys_source and values from values_source
    """
    keys_source = _ensure_dynamic_cache(keys_source)
    values_source = _ensure_dynamic_cache(values_source)

    new_cache = DynamicCache()
    n_layers = len(keys_source)
    assert len(values_source) == n_layers, "Caches must have same number of layers"

    for layer_idx in range(n_layers):
        k = _get_cache_keys(keys_source, layer_idx).clone()
        v = _get_cache_values(values_source, layer_idx).clone()
        new_cache.update(k, v, layer_idx)

    return new_cache


def swap_bos_entry(target_cache: Any, source_cache: Any) -> DynamicCache:
    """
    Replace the BOS (position 0) KV entry in target_cache with the one from source_cache.

    Args:
        target_cache: Cache whose BOS entry will be replaced
        source_cache: Cache to copy BOS entry from

    Returns:
        New DynamicCache with BOS from source, rest from target
    """
    target_cache = _ensure_dynamic_cache(target_cache)
    source_cache = _ensure_dynamic_cache(source_cache)

    new_cache = DynamicCache()
    n_layers = len(target_cache)

    for layer_idx in range(n_layers):
        tk = _get_cache_keys(target_cache, layer_idx).clone()
        tv = _get_cache_values(target_cache, layer_idx).clone()
        sk = _get_cache_keys(source_cache, layer_idx)
        sv = _get_cache_values(source_cache, layer_idx)

        tk[:, :, :1, :] = sk[:, :, :1, :]
        tv[:, :, :1, :] = sv[:, :, :1, :]

        new_cache.update(tk, tv, layer_idx)

    return new_cache


def apply_rope_roundtrip_noise(
    cache: DynamicCache,
    offset: int,
    model: AutoModelForCausalLM,
) -> DynamicCache:
    """
    Apply RoPE(+offset) then RoPE(-offset) to keys, introducing float16 roundtrip noise.

    This simulates the key perturbation from truncation+correction without any
    value contamination. In exact arithmetic this is identity, but in float16 it
    introduces ~2e-3 max error per element.

    BOS (position 0) is left untouched.

    Args:
        cache: DynamicCache to add noise to (modified in-place)
        offset: RoPE offset magnitude
        model: The model (for RoPE config)

    Returns:
        The same cache, modified in-place
    """
    cache = _ensure_dynamic_cache(cache)

    if offset == 0:
        return cache

    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    rope_theta = _get_rope_theta(config)

    # Forward: RoPE(+offset)
    cos_fwd, sin_fwd = _build_rope_correction(offset, head_dim, rope_theta)
    # Inverse: RoPE(-offset)  — note _build_rope_correction already negates
    # So we call with -offset to get the forward direction, and +offset for inverse
    # Actually _build_rope_correction uses angles = -offset * inv_freq
    # So for +offset rotation: angles = -(-offset) * inv_freq = +offset * inv_freq
    # We need: first apply +S, then apply -S
    # _build_rope_correction(offset) gives angles = -offset (i.e., inverse/correction)
    # _build_rope_correction(-offset) gives angles = +offset (i.e., forward)
    cos_fwd, sin_fwd = _build_rope_correction(-offset, head_dim, rope_theta)  # forward +S
    cos_inv, sin_inv = _build_rope_correction(offset, head_dim, rope_theta)    # inverse -S

    for layer_idx in range(len(cache)):
        keys = _get_cache_keys(cache, layer_idx)
        device = keys.device
        dtype = keys.dtype

        cf = cos_fwd.to(device=device, dtype=dtype)
        sf = sin_fwd.to(device=device, dtype=dtype)
        ci = cos_inv.to(device=device, dtype=dtype)
        si = sin_inv.to(device=device, dtype=dtype)

        # Apply to document tokens only (skip BOS at position 0)
        doc_keys = keys[:, :, 1:, :]

        # Forward rotation: RoPE(+S)
        rotated = doc_keys * cf + _rotate_half(doc_keys) * sf
        # Inverse rotation: RoPE(-S)
        corrected = rotated * ci + _rotate_half(rotated) * si

        _set_cache_keys(cache, layer_idx, torch.cat([keys[:, :, :1, :], corrected], dim=2))

    return cache


def score_answer_with_cache_and_attention(
    past_key_values: Any,
    context_len: int,
    query_prompt: str,
    answer: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ExperimentConfig,
) -> Tuple[float, Any]:
    """
    Like score_answer_with_cache but also returns attention weights.

    Args:
        past_key_values: Pre-built KV cache
        context_len: Length of the cached context
        query_prompt: The query prompt to append
        answer: The answer to score
        model: The language model
        tokenizer: The tokenizer
        config: Experiment configuration

    Returns:
        Tuple of (mean_nll, attentions) where attentions is a tuple of
        per-layer attention tensors from the answer scoring pass
    """
    query_encoding = tokenizer(
        query_prompt, return_tensors="pt", add_special_tokens=False,
        padding=False, truncation=False
    )
    query_ids = query_encoding['input_ids'].to(config.device)
    query_len = query_ids.shape[1]

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

    # Score answer with attention outputs
    combined_len_final = context_len + query_len + answer_len
    attention_mask_final = torch.ones((1, combined_len_final), device=config.device)

    with torch.no_grad():
        answer_outputs = model(
            input_ids=answer_ids,
            attention_mask=attention_mask_final,
            past_key_values=extended_cache,
            use_cache=False,
            return_dict=True,
            output_attentions=True,
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
    mean_nll = nll / num_scored if num_scored > 0 else 0.0

    return mean_nll, answer_outputs.attentions


def score_answer_with_cache_flexible(
    cache: Any,
    seq_len: int,
    query: str,
    answer: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    query_attention_mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Score P(answer | cache, query) using NLL, with optional query-time attention mask.

    This is a flexible scoring function that allows custom attention patterns
    at query time (e.g., limiting which cached positions the query can attend to).

    IMPORTANT: This function mutates the cache. Always deepcopy before calling
    if you need to reuse the cache.

    Args:
        cache: Pre-built KV cache (DynamicCache or compatible)
        seq_len: Length of the cached context
        query: The query string
        answer: The answer string to score
        model: The language model
        tokenizer: The tokenizer
        query_attention_mask: Optional 4D mask (1, 1, query_len, cache_len + query_len).
            For additive masks: 0 = attend, -inf = block.
            Will be cast to model.dtype automatically.

    Returns:
        Mean negative log-likelihood (lower is better)

    Example:
        # Score with standard attention (query sees all cached tokens)
        nll = score_answer_with_cache_flexible(cache, seq_len, query, answer, model, tokenizer)

        # Score with query-time masking (query sees only first prefix + passage)
        from lib.block_attention import create_query_time_mask_flexible
        mask = create_query_time_mask_flexible(seq_len, query_len, prefix_info, 'first_only',
                                               model.device, model.dtype)
        nll = score_answer_with_cache_flexible(cache, seq_len, query, answer, model, tokenizer, mask)
    """
    continuation = f"Query: {query}\nAnswer: {answer}"
    cont_ids = tokenizer.encode(continuation, return_tensors='pt', add_special_tokens=False).to(model.device)

    # Get answer token positions for loss computation
    answer_only = f" {answer}"
    answer_ids = tokenizer.encode(answer_only, return_tensors='pt', add_special_tokens=False)
    answer_len = answer_ids.shape[1]
    query_len = cont_ids.shape[1]

    with torch.no_grad():
        position_ids = torch.arange(seq_len, seq_len + query_len, device=model.device).unsqueeze(0)

        if query_attention_mask is not None:
            # Cast to model dtype for SDPA compatibility
            query_attention_mask = query_attention_mask.to(device=model.device, dtype=model.dtype)
            outputs = model(
                input_ids=cont_ids,
                past_key_values=cache,
                position_ids=position_ids,
                attention_mask=query_attention_mask,
                use_cache=True
            )
        else:
            outputs = model(
                input_ids=cont_ids,
                past_key_values=cache,
                position_ids=position_ids,
                use_cache=True
            )

        logits = outputs.logits[0]
        total_len = cont_ids.shape[1]
        answer_logits = logits[total_len - answer_len - 1 : total_len - 1, :]
        answer_targets = cont_ids[0, total_len - answer_len:]
        loss = F.cross_entropy(answer_logits, answer_targets, reduction='mean')

    return loss.item()


def replace_values_at_layers(
    target: Any,
    source: Any,
    layer_indices: list,
) -> DynamicCache:
    """
    Replace value tensors at specific layers in target with values from source.

    Keys are left unchanged. Useful for layer ablation experiments.

    Args:
        target: Cache to modify (cloned, not modified in-place)
        source: Cache to copy values from
        layer_indices: List of layer indices whose values to replace

    Returns:
        New DynamicCache with replaced values at specified layers
    """
    target = _ensure_dynamic_cache(target)
    source = _ensure_dynamic_cache(source)

    new_cache = DynamicCache()
    n_layers = len(target)

    for layer_idx in range(n_layers):
        k = _get_cache_keys(target, layer_idx).clone()
        if layer_idx in layer_indices:
            v = _get_cache_values(source, layer_idx).clone()
        else:
            v = _get_cache_values(target, layer_idx).clone()
        new_cache.update(k, v, layer_idx)

    return new_cache


def build_truncated_cache_variable_prefix(
    prefix_text: str,
    document: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ExperimentConfig,
) -> Tuple[int, DynamicCache, int]:
    """
    Build a truncated+corrected cache using arbitrary raw prefix text.

    Unlike build_truncated_kv_cache_corrected which wraps the surrogate in a
    template, this function uses prefix_text verbatim (with a trailing space
    separator).

    Args:
        prefix_text: Raw prefix text to prepend before document
        document: The document text
        model: The language model
        tokenizer: The tokenizer
        config: Experiment configuration

    Returns:
        Tuple of (keep_len, corrected_DynamicCache, prefix_token_len)
        where prefix_token_len is the number of prefix tokens (including BOS)
    """
    document_text = f"Document:\n{document}"

    # Tokenize prefix with BOS
    prefix_with_sep = prefix_text + " "
    prefix_encoding = tokenizer(
        prefix_with_sep, return_tensors="pt", add_special_tokens=True,
        padding=False, truncation=False
    )
    prefix_len = prefix_encoding['input_ids'].shape[1]

    # Tokenize full context
    full_context = prefix_with_sep + document_text
    full_encoding = tokenizer(
        full_context, return_tensors="pt", add_special_tokens=True,
        padding=False, truncation=False
    )
    full_ids = full_encoding['input_ids'].to(config.device)
    full_len = full_ids.shape[1]
    doc_len = full_len - prefix_len

    # Generate full KV cache
    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids),
            use_cache=True,
            return_dict=True
        )

    # Keep BOS + document portion
    keep_len = 1 + doc_len
    truncated_cache = extract_and_truncate_cache_with_bos(
        outputs.past_key_values, doc_len
    )

    # RoPE correction
    surrogate_offset = prefix_len - 1
    correct_rope_positions_with_bos(truncated_cache, surrogate_offset, model)

    return keep_len, truncated_cache, prefix_len


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


def _get_rope_theta(config) -> float:
    """Extract rope_theta from model config, checking rope_parameters first."""
    if hasattr(config, 'rope_parameters') and isinstance(config.rope_parameters, dict):
        if 'rope_theta' in config.rope_parameters:
            return float(config.rope_parameters['rope_theta'])
    return float(getattr(config, 'rope_theta', 10000.0))


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
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    angles = (-offset * inv_freq).double()  # compute in float64 for precision
    emb = torch.cat((angles, angles), dim=-1)  # (head_dim,) — HF duplication
    return emb.cos().float(), emb.sin().float()


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
    rope_theta = _get_rope_theta(config)

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
