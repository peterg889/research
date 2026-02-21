"""
Dataset loading and filtering utilities for MS MARCO experiments.
"""

from typing import List, Dict
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

from .config import ExperimentConfig


def count_words(text: str) -> int:
    """Count words in a text string."""
    return len(text.split())


def load_evaluation_samples(
    dataset,
    config: ExperimentConfig,
    require_answer: bool = True
) -> List[Dict]:
    """
    Load samples for evaluation with passage, query, and optionally answer.

    This function filters MS MARCO samples to find those with:
    - Valid passages within word count limits
    - Selected passages (if available)
    - Well-formed answers (if require_answer=True)

    Args:
        dataset: The MS MARCO dataset
        config: Experiment configuration
        require_answer: Whether to require a valid answer

    Returns:
        List of dicts with 'passage', 'query', and optionally 'answer' keys
    """
    print("Filtering samples...")

    filtered_samples = []

    for item in tqdm(dataset, desc="Filtering"):
        passages = item.get('passages', {})
        passage_texts = passages.get('passage_text', [])
        is_selected = passages.get('is_selected', [])

        query = item.get('query', '')
        answers = item.get('answers', [])
        well_formed = item.get('wellFormedAnswers', [])

        if not passage_texts or not query:
            continue

        # Get best answer if required
        answer = None
        if require_answer:
            if well_formed and len(well_formed) > 0 and well_formed[0] != '[]':
                answer = well_formed[0]
            elif answers and len(answers) > 0 and answers[0] != 'No Answer Present.':
                answer = answers[0]
            else:
                continue

        # Find valid passage
        for i, passage in enumerate(passage_texts):
            word_count = count_words(passage)
            if config.min_passage_words <= word_count <= config.max_passage_words:
                if is_selected and i < len(is_selected) and is_selected[i] == 1:
                    sample = {
                        'passage': passage,
                        'query': query,
                    }
                    if answer:
                        sample['answer'] = answer
                    filtered_samples.append(sample)
                    break

        if len(filtered_samples) >= config.num_samples * 2:
            break

    np.random.shuffle(filtered_samples)
    filtered_samples = filtered_samples[:config.num_samples]

    print(f"Selected {len(filtered_samples)} samples")
    return filtered_samples


def load_and_filter_dataset(config: ExperimentConfig) -> List[Dict]:
    """
    Load MS MARCO dataset and filter passages by word count.

    This is an alternative loader for experiments that don't need answers.

    Args:
        config: Experiment configuration

    Returns:
        List of dicts with 'passage' and 'query' keys
    """
    print(f"Loading {config.dataset_name} dataset...")

    # Load MS MARCO
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=config.dataset_split,
        trust_remote_code=True
    )

    print(f"Total samples in {config.dataset_split}: {len(dataset)}")

    # Filter and extract valid samples
    filtered_samples = []

    for item in tqdm(dataset, desc="Filtering passages"):
        passages = item.get('passages', {})
        passage_texts = passages.get('passage_text', [])
        is_selected = passages.get('is_selected', [])

        query = item.get('query', '')

        if not passage_texts or not query:
            continue

        # Find a passage that meets word count criteria
        for i, passage in enumerate(passage_texts):
            word_count = count_words(passage)
            if config.min_passage_words <= word_count <= config.max_passage_words:
                # Prefer selected passages if available
                if is_selected and i < len(is_selected) and is_selected[i] == 1:
                    filtered_samples.append({
                        'passage': passage,
                        'query': query
                    })
                    break
                elif not any(is_selected):  # No selection info, take first valid
                    filtered_samples.append({
                        'passage': passage,
                        'query': query
                    })
                    break

        # Early stop if we have enough
        if len(filtered_samples) >= config.num_samples * 2:
            break

    # Shuffle and limit
    np.random.seed(config.seed)
    np.random.shuffle(filtered_samples)
    filtered_samples = filtered_samples[:config.num_samples]

    print(f"Filtered to {len(filtered_samples)} samples")
    return filtered_samples


def load_ms_marco(config: ExperimentConfig):
    """
    Load the MS MARCO dataset.

    Args:
        config: Experiment configuration

    Returns:
        The loaded dataset
    """
    print(f"Loading {config.dataset_name} dataset...")
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=config.dataset_split,
        trust_remote_code=True
    )
    print(f"Dataset loaded: {len(dataset)} samples")
    return dataset
