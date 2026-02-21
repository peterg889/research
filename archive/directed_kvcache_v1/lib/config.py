"""
Experiment configuration for directed KV cache experiments.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class ExperimentConfig:
    """
    Unified configuration for all directed KV cache experiments.

    This config supports multiple experiment types:
    - Basic surrogate priming (surrogate_primed_kv_caching)
    - Production simulation with routing (production_simulation_experiment)
    - Truncated cache experiments (production_simulation_truncated_cache)
    """

    # ===== Model Settings =====
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    model_type: str = "mistral"  # "mistral" or "chatglm"
    use_4bit: bool = True

    # ===== Dataset Settings =====
    dataset_name: str = "microsoft/ms_marco"
    dataset_config: str = "v1.1"
    dataset_split: str = "validation"
    num_samples: int = 2500
    min_passage_words: int = 50
    max_passage_words: int = 300

    # ===== Surrogate Generation =====
    surrogate_max_tokens: int = 45
    surrogate_temperature: float = 0.3

    # Basic surrogate generation prompt (for simple experiments)
    surrogate_generation_prompt: str = (
        "Read the following text and write ONE short search query (3-8 words) "
        "that someone would type into Google to find this text. "
        "Output only the query with no quotes, alternatives, or explanation."
    )

    # ===== Cache Templates =====
    # Baseline: document only
    baseline_cache_template: str = "Document:\n{document}"

    # Surrogate-primed: surrogate + document
    surrogate_cache_template: str = (
        "This document may be relevant to queries like: {surrogate}\n\n"
        "Document:\n{document}"
    )

    # Alternative templates for different experiment designs
    baseline_prompt_pure: str = ""  # No framing at all
    baseline_prompt_framed: str = "Document:\n"  # Structure but no query
    surrogate_prefix_template: str = "Query: {surrogate}\n\nDocument:\n"  # Query prefix style

    # Query template for answer scoring
    query_template: str = "\n\nQuery: {query}\n\nAnswer:"

    # ===== Embedding Model =====
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # ===== Reproducibility =====
    seed: int = 42

    # ===== Device =====
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.min_passage_words >= self.max_passage_words:
            raise ValueError("min_passage_words must be less than max_passage_words")
        if self.surrogate_max_tokens < 1:
            raise ValueError("surrogate_max_tokens must be positive")
