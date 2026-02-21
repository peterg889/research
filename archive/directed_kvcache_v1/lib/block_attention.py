"""
Block-diagonal attention masks for prefix repetition experiments.

The goal is to prevent repeated prefix queries from attending to each other,
while still allowing:
1. Each prefix copy to attend to itself (causal within the block)
2. Passage tokens to attend to all prefix copies
3. Query tokens (at inference) to attend to everything

This tests whether the "semantic interference" from repeated identical tokens
can be mitigated by preventing cross-repetition attention.
"""

import torch
from typing import Tuple, Optional, List


def create_block_diagonal_prefix_mask(
    prefix_len: int,
    n_reps: int,
    passage_len: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create attention mask where prefix repetitions can't attend to each other.

    Args:
        prefix_len: Number of tokens in ONE prefix repetition
        n_reps: Number of times the prefix is repeated
        passage_len: Number of tokens in the passage
        dtype: Output dtype (float for additive mask, bool for boolean mask)
        device: Target device

    Returns:
        Attention mask of shape (1, 1, total_len, total_len)

        For additive masks (float): 0.0 = attend, -inf = block
        For boolean masks: True = attend, False = block

    Attention pattern:
        - Within each prefix repetition: causal attention allowed
        - Between prefix repetitions: blocked
        - Passage to prefix: all allowed (causal)
        - Passage to passage: causal attention allowed

    Example with prefix_len=3, n_reps=2, passage_len=2:

        Positions: [P0 P1 P2] [P3 P4 P5] [D0 D1]
                    Rep 1      Rep 2      Passage

        Attention matrix (1 = attend, 0 = block):

              P0 P1 P2 P3 P4 P5 D0 D1
        P0 [  1  0  0  0  0  0  0  0 ]
        P1 [  1  1  0  0  0  0  0  0 ]
        P2 [  1  1  1  0  0  0  0  0 ]
        P3 [  0  0  0  1  0  0  0  0 ]  <- Rep 2 can't see Rep 1
        P4 [  0  0  0  1  1  0  0  0 ]
        P5 [  0  0  0  1  1  1  0  0 ]
        D0 [  1  1  1  1  1  1  1  0 ]  <- Passage sees all
        D1 [  1  1  1  1  1  1  1  1 ]
    """
    total_prefix = prefix_len * n_reps
    total_len = total_prefix + passage_len

    if device is None:
        device = torch.device('cpu')

    # Start with zeros (will fill in allowed attention)
    # Using float mask where 0 = attend, -inf = block
    is_float_mask = dtype in (torch.float32, torch.float16, torch.bfloat16)

    if is_float_mask:
        # Initialize with -inf (blocked)
        mask = torch.full((total_len, total_len), float('-inf'), dtype=dtype, device=device)
    else:
        # Boolean mask: False = blocked
        mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)

    attend_value = 0.0 if is_float_mask else True

    # Fill in allowed attention patterns

    # 1. Each prefix repetition can attend to itself (causal within block)
    for rep in range(n_reps):
        start = rep * prefix_len
        end = start + prefix_len
        # Causal mask within this block
        for i in range(start, end):
            for j in range(start, i + 1):  # Can attend to positions <= current
                mask[i, j] = attend_value

    # 2. Passage can attend to ALL prefix tokens and itself (causal)
    passage_start = total_prefix
    for i in range(passage_start, total_len):
        # Can attend to all prefix
        for j in range(total_prefix):
            mask[i, j] = attend_value
        # Can attend to passage tokens <= current position
        for j in range(passage_start, i + 1):
            mask[i, j] = attend_value

    # Reshape to (1, 1, seq_len, seq_len) for HuggingFace compatibility
    return mask.unsqueeze(0).unsqueeze(0)


def create_query_time_mask(
    cache_len: int,
    query_len: int,
    prefix_len: int,
    n_reps: int,
    block_query_to_prefix_copies: bool = False,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create attention mask for query-time scoring with pre-built cache.

    Args:
        cache_len: Length of the KV cache (prefix + passage)
        query_len: Length of the new query being scored
        prefix_len: Length of ONE prefix repetition (for blocking)
        n_reps: Number of prefix repetitions
        block_query_to_prefix_copies: If True, query can only attend to ONE copy
                                      of each prefix token (reduces interference)
        dtype: Output dtype
        device: Target device

    Returns:
        Attention mask of shape (1, 1, query_len, cache_len + query_len)
    """
    total_len = cache_len + query_len

    if device is None:
        device = torch.device('cpu')

    is_float_mask = dtype in (torch.float32, torch.float16, torch.bfloat16)

    if is_float_mask:
        # Start with -inf (blocked), fill in allowed
        mask = torch.full((query_len, total_len), float('-inf'), dtype=dtype, device=device)
    else:
        mask = torch.zeros((query_len, total_len), dtype=torch.bool, device=device)

    attend_value = 0.0 if is_float_mask else True

    total_prefix = prefix_len * n_reps

    for i in range(query_len):
        query_pos = cache_len + i

        if block_query_to_prefix_copies:
            # Query attends to only the FIRST copy of each prefix position
            for j in range(prefix_len):
                mask[i, j] = attend_value
        else:
            # Query attends to ALL prefix tokens (standard)
            for j in range(total_prefix):
                mask[i, j] = attend_value

        # Query attends to all passage tokens
        for j in range(total_prefix, cache_len):
            mask[i, j] = attend_value

        # Query attends to previous query tokens (causal)
        for j in range(cache_len, query_pos + 1):
            mask[i, j] = attend_value

    return mask.unsqueeze(0).unsqueeze(0)


def create_query_time_mask_flexible(
    cache_len: int,
    query_len: int,
    prefix_boundaries: List[Tuple[int, int]],
    passage_start: int,
    mask_type: str,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create flexible attention mask for query-time scoring.

    This allows controlling which parts of the cache the query can attend to,
    useful for testing interference hypotheses.

    Args:
        cache_len: Length of the KV cache
        query_len: Length of the new query being scored
        prefix_boundaries: List of (start, end) tuples for each prefix repetition
        passage_start: Token index where the passage begins
        mask_type: One of:
            - 'standard': Query sees everything (normal causal)
            - 'first_only': Query sees ONLY first prefix copy + passage
            - 'last_only': Query sees ONLY last prefix copy + passage
            - 'passage_only': Query sees ONLY passage (no prefix visible)
        dtype: Output dtype (should match model.dtype for SDPA)
        device: Target device

    Returns:
        Attention mask of shape (1, 1, query_len, cache_len + query_len)
        Using additive format: 0 = attend, -inf = block

    Example usage for experiment conditions:
        # oracle_5x_qmask_first: Query attends to [BOS][first prefix copy][passage][query]
        # This tests if seeing ONE prefix copy is better than seeing ALL copies
        mask = create_query_time_mask_flexible(cache_len, query_len, boundaries,
                                               passage_start, 'first_only', model.dtype, model.device)

        # oracle_5x_qmask_last: Query attends to [BOS][last prefix copy][passage][query]
        # Compared to first_only, tests if position matters (last copy has seen more context)
        mask = create_query_time_mask_flexible(cache_len, query_len, boundaries,
                                               passage_start, 'last_only', model.dtype, model.device)

        # oracle_5x_qmask_none: Query attends to [BOS][passage][query] only
        # Tests if prefix visibility hurts (pure interference test)
        mask = create_query_time_mask_flexible(cache_len, query_len, boundaries,
                                               passage_start, 'passage_only', model.dtype, model.device)
    """
    total_len = cache_len + query_len

    if device is None:
        device = torch.device('cpu')

    is_float_mask = dtype in (torch.float32, torch.float16, torch.bfloat16)

    if is_float_mask:
        # Start with -inf (blocked)
        mask = torch.full((query_len, total_len), float('-inf'), dtype=dtype, device=device)
        attend_value = 0.0
    else:
        mask = torch.zeros((query_len, total_len), dtype=torch.bool, device=device)
        attend_value = True

    for i in range(query_len):
        # Query always sees BOS (position 0)
        mask[i, 0] = attend_value

        # Query always sees passage
        for j in range(passage_start, cache_len):
            mask[i, j] = attend_value

        # Query always sees previous query tokens (causal)
        for j in range(cache_len, cache_len + i + 1):
            mask[i, j] = attend_value

        # Handle prefix visibility based on mask_type
        if mask_type == 'first_only' and prefix_boundaries:
            # Only see first prefix copy
            start, end = prefix_boundaries[0]
            for j in range(start, end):
                mask[i, j] = attend_value

        elif mask_type == 'last_only' and prefix_boundaries:
            # Only see last prefix copy
            start, end = prefix_boundaries[-1]
            for j in range(start, end):
                mask[i, j] = attend_value

        elif mask_type == 'passage_only':
            # Only see passage (BOS already set above)
            pass  # Passage positions already set

        elif mask_type == 'standard' or mask_type is None:
            # Standard: see everything in cache
            for j in range(cache_len):
                mask[i, j] = attend_value

    return mask.unsqueeze(0).unsqueeze(0)


def get_prefix_boundaries_from_text(
    full_text: str,
    prefix_query: str,
    n_reps: int,
    tokenizer,
) -> dict:
    """
    Get precise token boundaries for prefix repetitions in a text.

    This is useful for creating query-time masks that target specific
    prefix repetitions.

    Args:
        full_text: The full text (prefix + passage)
        prefix_query: The prefix query (one copy, without trailing space)
        n_reps: Expected number of repetitions
        tokenizer: The tokenizer

    Returns:
        dict with:
            - prefix_len: tokens per repetition (including trailing space)
            - boundaries: list of (start, end) tuples for each rep
            - passage_start: token index where passage begins
            - total_prefix: total prefix tokens
            - seq_len: total sequence length
    """
    # Tokenize prefix once (with the space that follows in repetition)
    prefix_ids = tokenizer.encode(prefix_query + " ", add_special_tokens=False)
    prefix_len = len(prefix_ids)

    # Tokenize full text
    full_ids = tokenizer.encode(full_text, add_special_tokens=True)
    seq_len = len(full_ids)

    # Find boundaries (skip BOS at position 0)
    boundaries = []
    pos = 1  # Skip BOS
    for rep in range(n_reps):
        end_pos = pos + prefix_len
        if end_pos <= seq_len:
            boundaries.append((pos, end_pos))
            pos = end_pos
        else:
            break

    return {
        'prefix_len': prefix_len,
        'boundaries': boundaries,
        'passage_start': boundaries[-1][1] if boundaries else 1,
        'total_prefix': prefix_len * len(boundaries),
        'seq_len': seq_len,
    }


def validate_mask_properties(
    mask: torch.Tensor,
    prefix_len: int,
    n_reps: int,
    passage_len: int,
) -> dict:
    """
    Validate that a mask has the expected properties.

    Returns dict with validation results and any violations found.
    """
    # Remove batch dimensions for analysis
    if mask.dim() == 4:
        mask = mask.squeeze(0).squeeze(0)

    total_prefix = prefix_len * n_reps
    total_len = total_prefix + passage_len

    # Convert to boolean for analysis
    if mask.dtype in (torch.float32, torch.float16, torch.bfloat16):
        # Float mask: 0 = attend, -inf = block
        attend_mask = mask > float('-inf')
    else:
        attend_mask = mask

    results = {
        'valid': True,
        'violations': [],
        'stats': {}
    }

    # Check 1: No future attention (causal property)
    for i in range(total_len):
        for j in range(i + 1, total_len):
            if attend_mask[i, j]:
                results['violations'].append(f"Future attention: pos {i} attends to {j}")
                results['valid'] = False

    # Check 2: Within-repetition attention is causal
    for rep in range(n_reps):
        start = rep * prefix_len
        end = start + prefix_len
        for i in range(start, end):
            for j in range(start, i + 1):
                if not attend_mask[i, j]:
                    results['violations'].append(
                        f"Missing within-rep attention: rep {rep}, pos {i} should attend to {j}"
                    )
                    results['valid'] = False

    # Check 3: Cross-repetition attention is blocked
    cross_rep_violations = 0
    for rep_i in range(1, n_reps):
        start_i = rep_i * prefix_len
        end_i = start_i + prefix_len
        for rep_j in range(rep_i):
            start_j = rep_j * prefix_len
            end_j = start_j + prefix_len
            for i in range(start_i, end_i):
                for j in range(start_j, end_j):
                    if attend_mask[i, j]:
                        cross_rep_violations += 1

    if cross_rep_violations > 0:
        results['violations'].append(f"Cross-repetition attention: {cross_rep_violations} violations")
        results['valid'] = False

    # Check 4: Passage can attend to all prefix
    passage_to_prefix_violations = 0
    for i in range(total_prefix, total_len):
        for j in range(total_prefix):
            if not attend_mask[i, j]:
                passage_to_prefix_violations += 1

    if passage_to_prefix_violations > 0:
        results['violations'].append(
            f"Passage-to-prefix blocked: {passage_to_prefix_violations} violations"
        )
        results['valid'] = False

    # Stats
    results['stats'] = {
        'total_positions': total_len,
        'prefix_positions': total_prefix,
        'passage_positions': passage_len,
        'attend_count': attend_mask.sum().item(),
        'block_count': (~attend_mask).sum().item(),
        'sparsity': (~attend_mask).sum().item() / (total_len * total_len),
    }

    return results


def get_repetition_boundaries(token_ids: torch.Tensor, prefix_token_ids: torch.Tensor) -> List[Tuple[int, int]]:
    """
    Find the boundaries of each prefix repetition in a token sequence.

    Args:
        token_ids: Full sequence token IDs (1D tensor)
        prefix_token_ids: Token IDs of ONE prefix copy (1D tensor)

    Returns:
        List of (start, end) tuples for each repetition found
    """
    token_ids = token_ids.flatten().tolist()
    prefix_ids = prefix_token_ids.flatten().tolist()
    prefix_len = len(prefix_ids)

    boundaries = []
    pos = 0

    while pos <= len(token_ids) - prefix_len:
        if token_ids[pos:pos + prefix_len] == prefix_ids:
            boundaries.append((pos, pos + prefix_len))
            pos += prefix_len
        else:
            break  # Prefix repetitions must be contiguous at start

    return boundaries
