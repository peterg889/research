"""
Surrogate query generation utilities.

Includes templates for generating document-specific surrogate queries
and static surrogate queries for cache routing.
"""

from typing import Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .config import ExperimentConfig


# =============================================================================
# Generated Surrogate Templates (Doc2Query-inspired)
# =============================================================================

TOP_5_SURROGATE_TEMPLATES = {
    'target_question': {
        'name': 'Target Natural Language Question',
        'description': 'The ideal, grammatically-correct question this document perfectly answers',
        'prompt': (
            "You are helping index a document for search. Write the single most likely "
            "natural language question that a user would ask that this document perfectly answers. "
            "The question should be grammatically correct, clear, and specific. "
            "Output only the question (5-12 words), nothing else.\n\n"
            "Document:"
        ),
    },
    'keyword_query': {
        'name': 'Keyword-ese Query',
        'description': 'How users actually search: keyword strings without full sentences',
        'prompt': (
            "You are helping index a document for search. Write a search query the way "
            "real users type into Google: just keywords, no complete sentences, no question marks. "
            "Think of someone quickly typing a few relevant words. "
            "Output only the keyword query (3-6 words), nothing else.\n\n"
            "Document:"
        ),
    },
    'symptom_scenario': {
        'name': 'Symptom/Scenario Query',
        'description': 'The problem or symptom the user has, not the solution they need',
        'prompt': (
            "You are helping index a document for search. This document contains a solution or answer. "
            "Write a query that describes the PROBLEM or SYMPTOM that would lead someone to need this document. "
            "Focus on what the user is experiencing, not what they want to learn. "
            "For example, if the doc is about 'fixing a leaky faucet', the query might be 'water dripping from sink handle'. "
            "Output only the problem-focused query (4-10 words), nothing else.\n\n"
            "Document:"
        ),
    },
    'misconception_negative': {
        'name': 'Misconception/Negative Query',
        'description': 'Questions about what NOT to do, common myths, or concerns',
        'prompt': (
            "You are helping index a document for search. Write a query that reflects "
            "a common misconception, concern, or 'what NOT to do' question related to this topic. "
            "Think of someone who is worried, skeptical, or wants to avoid mistakes. "
            "Examples: 'is X bad for you', 'X side effects', 'mistakes to avoid with X', 'X myths'. "
            "Output only the concern/negative query (4-10 words), nothing else.\n\n"
            "Document:"
        ),
    },
    'messy_realworld': {
        'name': 'Messy Real-World Query',
        'description': 'Abbreviations, slang, typos, urgency - how people really search',
        'prompt': (
            "You are helping index a document for search. Write a messy, realistic search query "
            "the way a rushed person might actually type it - with abbreviations, informal language, "
            "or urgency words like 'help', 'asap', 'plz'. Not everyone types perfectly. "
            "Output only the messy query (3-8 words), nothing else.\n\n"
            "Document:"
        ),
    },
}


# =============================================================================
# Static Surrogate Queries (same for all documents)
# =============================================================================

STATIC_SURROGATE_QUERIES = {
    'static_definitional': {
        'name': 'Static: Definitional Intent',
        'query': 'What is this and what does it mean?',
        'covers': 'what is, define, meaning, explanation queries',
    },
    'static_procedural': {
        'name': 'Static: Procedural Intent',
        'query': 'How do I do this step by step?',
        'covers': 'how to, how do, instructions, guide queries',
    },
    'static_quantitative': {
        'name': 'Static: Quantitative Intent',
        'query': 'How much does this cost or how long does it take?',
        'covers': 'how much, how many, cost, price, duration queries',
    },
    'static_factual': {
        'name': 'Static: Factual Intent',
        'query': 'What are the key facts I need to know?',
        'covers': 'who, when, where, what, factual detail queries',
    },
    'static_problem': {
        'name': 'Static: Problem/Solution Intent',
        'query': 'What problem does this solve?',
        'covers': 'why, troubleshooting, help, problem-solving queries',
    },
}


# =============================================================================
# Surrogate Generation Functions
# =============================================================================

def generate_surrogate_with_template(
    doc_text: str,
    template_prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ExperimentConfig
) -> str:
    """
    Generate a surrogate query using a specific template.

    Args:
        doc_text: The document text to generate a surrogate for
        template_prompt: The prompt template to use
        model: The language model
        tokenizer: The tokenizer
        config: Experiment configuration

    Returns:
        Generated surrogate query string
    """
    messages = [
        {
            "role": "user",
            "content": f"{template_prompt}\n\nText:\n{doc_text}"
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(config.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.surrogate_max_tokens,
            temperature=config.surrogate_temperature,
            do_sample=config.surrogate_temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    surrogate = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    surrogate = surrogate.strip('"\'')
    surrogate = surrogate.split('\n')[0].strip()

    return surrogate


def generate_all_5_surrogates(
    doc_text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ExperimentConfig
) -> Dict[str, str]:
    """
    Generate all 5 surrogates for a document using the top 5 templates.

    Args:
        doc_text: The document text
        model: The language model
        tokenizer: The tokenizer
        config: Experiment configuration

    Returns:
        Dictionary mapping template key to generated surrogate
    """
    surrogates = {}
    for key, template in TOP_5_SURROGATE_TEMPLATES.items():
        surrogates[key] = generate_surrogate_with_template(
            doc_text, template['prompt'], model, tokenizer, config
        )
    return surrogates


def generate_surrogate(
    doc_text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ExperimentConfig
) -> str:
    """
    Generate a single surrogate query using the default generation prompt.

    This is a simpler interface for basic surrogate generation experiments.

    Args:
        doc_text: The document text
        model: The language model
        tokenizer: The tokenizer
        config: Experiment configuration

    Returns:
        Generated surrogate query string
    """
    return generate_surrogate_with_template(
        doc_text, config.surrogate_generation_prompt, model, tokenizer, config
    )


# =============================================================================
# Similarity Functions
# =============================================================================

def compute_similarity(
    text1: str,
    text2: str,
    embed_model: SentenceTransformer
) -> float:
    """
    Compute semantic similarity between two texts using embeddings.

    Args:
        text1: First text
        text2: Second text
        embed_model: SentenceTransformer model for embeddings

    Returns:
        Cosine similarity score (0 to 1)
    """
    embeddings = embed_model.encode([text1, text2])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
