# Directed KV Cache Library
# Shared utilities for surrogate-primed KV caching experiments

from .config import ExperimentConfig
from .kv_cache import (
    build_kv_cache,
    score_answer_with_cache,
    extract_and_truncate_cache,
    extract_and_truncate_cache_with_bos,
    build_truncated_kv_cache,
    correct_rope_positions,
    correct_rope_positions_with_bos,
    build_truncated_kv_cache_corrected,
)
from .surrogate import (
    TOP_5_SURROGATE_TEMPLATES,
    STATIC_SURROGATE_QUERIES,
    generate_surrogate_with_template,
    generate_all_5_surrogates,
    generate_surrogate,
    compute_similarity,
)
from .data import (
    count_words,
    load_evaluation_samples,
    load_and_filter_dataset,
    load_ms_marco,
)
from .analysis import analyze_experiment_results

__all__ = [
    # Config
    'ExperimentConfig',
    # KV Cache
    'build_kv_cache',
    'score_answer_with_cache',
    'extract_and_truncate_cache',
    'extract_and_truncate_cache_with_bos',
    'build_truncated_kv_cache',
    'correct_rope_positions',
    'correct_rope_positions_with_bos',
    'build_truncated_kv_cache_corrected',
    # Surrogate
    'TOP_5_SURROGATE_TEMPLATES',
    'STATIC_SURROGATE_QUERIES',
    'generate_surrogate_with_template',
    'generate_all_5_surrogates',
    'generate_surrogate',
    'compute_similarity',
    # Data
    'count_words',
    'load_evaluation_samples',
    'load_and_filter_dataset',
    'load_ms_marco',
    # Analysis
    'analyze_experiment_results',
]
