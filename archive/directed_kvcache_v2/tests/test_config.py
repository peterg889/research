"""Tests for ExperimentConfig validation."""

import pytest
from lib.config import ExperimentConfig


def test_defaults_work():
    cfg = ExperimentConfig()
    assert cfg.model_name == "mistralai/Mistral-7B-Instruct-v0.2"
    assert cfg.min_passage_words < cfg.max_passage_words


def test_min_ge_max_raises():
    with pytest.raises(ValueError, match="min_passage_words must be less"):
        ExperimentConfig(min_passage_words=300, max_passage_words=50)


def test_zero_surrogate_tokens_raises():
    with pytest.raises(ValueError, match="surrogate_max_tokens must be positive"):
        ExperimentConfig(surrogate_max_tokens=0)


def test_negative_surrogate_tokens_raises():
    with pytest.raises(ValueError, match="surrogate_max_tokens must be positive"):
        ExperimentConfig(surrogate_max_tokens=-5)


def test_device_populated():
    cfg = ExperimentConfig()
    assert cfg.device in ("cuda", "cpu")
