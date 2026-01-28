"""
KV Cache utilities for building, manipulating, and scoring with KV caches.
"""

from typing import Tuple, Any
import torch
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

    # Tokenize document alone to get exact length (without special tokens)
    doc_encoding = tokenizer(
        document_text, return_tensors="pt", add_special_tokens=False,
        padding=False, truncation=False
    )
    doc_len = doc_encoding['input_ids'].shape[1]

    # Tokenize full context
    full_context = surrogate_prefix + document_text
    full_encoding = tokenizer(
        full_context, return_tensors="pt", add_special_tokens=True,
        padding=False, truncation=False
    )
    full_ids = full_encoding['input_ids'].to(config.device)

    # Generate full KV cache
    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids),
            use_cache=True,
            return_dict=True
        )

    full_cache = outputs.past_key_values

    # Truncate: keep only the last doc_len positions (the document portion)
    truncated_cache = extract_and_truncate_cache(full_cache, doc_len)

    return doc_len, truncated_cache


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
