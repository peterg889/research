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
    build_suffix_kv_cache,
    build_hybrid_cache,
    swap_bos_entry,
    apply_rope_roundtrip_noise,
    score_answer_with_cache_and_attention,
    replace_values_at_layers,
    build_truncated_cache_variable_prefix,
    _ensure_dynamic_cache,
)
from .chatglm_kv_cache import (
    load_chatglm,
    build_kv_cache_chatglm,
    score_answer_with_cache_chatglm,
    build_suffix_kv_cache_chatglm,
    build_prefix_kv_cache_chatglm,
    build_truncated_kv_cache_chatglm,
    correct_2d_rope_positions,
)
from .surrogate import (
    TOP_5_SURROGATE_TEMPLATES,
    STATIC_SURROGATE_QUERIES,
    generate_surrogate_with_template,
    generate_all_5_surrogates,
    generate_surrogate,
    generate_summary,
    generate_surrogate_with_template_chatglm,
    generate_all_5_surrogates_chatglm,
    generate_surrogate_chatglm,
    generate_summary_chatglm,
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
    # KV Cache (Mistral)
    'build_kv_cache',
    'score_answer_with_cache',
    'extract_and_truncate_cache',
    'extract_and_truncate_cache_with_bos',
    'build_truncated_kv_cache',
    'correct_rope_positions',
    'correct_rope_positions_with_bos',
    'build_truncated_kv_cache_corrected',
    'build_suffix_kv_cache',
    'build_hybrid_cache',
    'swap_bos_entry',
    'apply_rope_roundtrip_noise',
    'score_answer_with_cache_and_attention',
    'replace_values_at_layers',
    'build_truncated_cache_variable_prefix',
    '_ensure_dynamic_cache',
    # KV Cache (ChatGLM)
    'load_chatglm',
    'build_kv_cache_chatglm',
    'score_answer_with_cache_chatglm',
    'build_suffix_kv_cache_chatglm',
    'build_prefix_kv_cache_chatglm',
    'build_truncated_kv_cache_chatglm',
    'correct_2d_rope_positions',
    # Surrogate (Mistral)
    'TOP_5_SURROGATE_TEMPLATES',
    'STATIC_SURROGATE_QUERIES',
    'generate_surrogate_with_template',
    'generate_all_5_surrogates',
    'generate_surrogate',
    'generate_summary',
    # Surrogate (ChatGLM)
    'generate_surrogate_with_template_chatglm',
    'generate_all_5_surrogates_chatglm',
    'generate_surrogate_chatglm',
    'generate_summary_chatglm',
    'compute_similarity',
    # Data
    'count_words',
    'load_evaluation_samples',
    'load_and_filter_dataset',
    'load_ms_marco',
    # Analysis
    'analyze_experiment_results',
]
