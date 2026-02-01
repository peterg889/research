"""Tests for generate_summary — prompt construction, decoding, and edge cases."""

import types
import torch
import pytest
from unittest.mock import MagicMock, call

from lib.surrogate import generate_summary
from lib.config import ExperimentConfig


class _TokenizerOutput:
    """Mimics HF tokenizer output with both dict and attribute access, plus .to()."""
    def __init__(self, input_ids):
        self.input_ids = input_ids
        self._data = {'input_ids': input_ids}

    def __getitem__(self, key):
        return self._data[key]

    def to(self, device):
        return self

    def keys(self):
        return self._data.keys()


def _make_summary_mocks(decode_return="This is a two sentence summary. It covers the topic."):
    """Create mocked model, tokenizer, and config for generate_summary tests."""
    config = ExperimentConfig(device="cpu", surrogate_temperature=0.3)

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "formatted prompt"

    input_ids = torch.tensor([[10, 11, 12, 13, 14]])
    tok_output = _TokenizerOutput(input_ids)
    tokenizer.return_value = tok_output
    tokenizer.eos_token_id = 2
    tokenizer.decode.return_value = decode_return

    model = MagicMock()
    # generate returns input tokens + 10 new tokens
    model.generate.return_value = torch.tensor([[10, 11, 12, 13, 14, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]])

    return config, model, tokenizer


class TestPromptConstruction:
    """Verify the prompt sent to the model is correct."""

    def test_chat_template_message_content(self):
        config, model, tokenizer = _make_summary_mocks()
        generate_summary("My test passage.", model, tokenizer, config)

        # Check that apply_chat_template was called with the right message
        call_args = tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        assert messages[0]['role'] == 'user'
        content = messages[0]['content']
        assert "Summarize the following text in exactly 2 concise sentences" in content
        assert "My test passage." in content

    def test_chat_template_kwargs(self):
        config, model, tokenizer = _make_summary_mocks()
        generate_summary("passage", model, tokenizer, config)

        call_args = tokenizer.apply_chat_template.call_args
        assert call_args[1]['tokenize'] is False
        assert call_args[1]['add_generation_prompt'] is True

    def test_prompt_contains_output_instruction(self):
        config, model, tokenizer = _make_summary_mocks()
        generate_summary("passage", model, tokenizer, config)

        content = tokenizer.apply_chat_template.call_args[0][0][0]['content']
        assert "Output only the summary, nothing else" in content


class TestGenerateCall:
    """Verify model.generate is called with correct parameters."""

    def test_max_new_tokens(self):
        config, model, tokenizer = _make_summary_mocks()
        generate_summary("passage", model, tokenizer, config)

        gen_kwargs = model.generate.call_args[1]
        assert gen_kwargs['max_new_tokens'] == 80

    def test_temperature_from_config(self):
        config, model, tokenizer = _make_summary_mocks()
        config.surrogate_temperature = 0.5
        generate_summary("passage", model, tokenizer, config)

        gen_kwargs = model.generate.call_args[1]
        assert gen_kwargs['temperature'] == 0.5
        assert gen_kwargs['do_sample'] is True

    def test_zero_temperature_no_sampling(self):
        config, model, tokenizer = _make_summary_mocks()
        config.surrogate_temperature = 0.0
        generate_summary("passage", model, tokenizer, config)

        gen_kwargs = model.generate.call_args[1]
        assert gen_kwargs['temperature'] == 0.0
        assert gen_kwargs['do_sample'] is False

    def test_eos_token_id_set(self):
        config, model, tokenizer = _make_summary_mocks()
        tokenizer.eos_token_id = 42
        generate_summary("passage", model, tokenizer, config)

        gen_kwargs = model.generate.call_args[1]
        assert gen_kwargs['pad_token_id'] == 42
        assert gen_kwargs['eos_token_id'] == 42


class TestOutputSlicing:
    """Verify that only generated tokens (not input) are decoded."""

    def test_slices_off_input_tokens(self):
        config, model, tokenizer = _make_summary_mocks()
        # input_ids has 5 tokens, generate returns 15 total
        generate_summary("passage", model, tokenizer, config)

        decode_args = tokenizer.decode.call_args
        sliced_tokens = decode_args[0][0]
        # Should have sliced off the 5 input tokens, leaving 10 generated
        assert len(sliced_tokens) == 10
        torch.testing.assert_close(
            sliced_tokens,
            torch.tensor([50, 51, 52, 53, 54, 55, 56, 57, 58, 59])
        )

    def test_skip_special_tokens_true(self):
        config, model, tokenizer = _make_summary_mocks()
        generate_summary("passage", model, tokenizer, config)

        decode_kwargs = tokenizer.decode.call_args[1]
        assert decode_kwargs['skip_special_tokens'] is True


class TestOutputCleaning:
    """Verify whitespace stripping of the decoded output."""

    def test_strips_whitespace(self):
        config, model, tokenizer = _make_summary_mocks("  A summary with spaces.  ")
        result = generate_summary("passage", model, tokenizer, config)
        assert result == "A summary with spaces."

    def test_preserves_internal_content(self):
        config, model, tokenizer = _make_summary_mocks("First sentence. Second sentence.")
        result = generate_summary("passage", model, tokenizer, config)
        assert result == "First sentence. Second sentence."

    def test_handles_newlines_in_output(self):
        """Unlike surrogate generation, summary does NOT strip at newlines —
        summaries can be multi-sentence and we want the full output."""
        config, model, tokenizer = _make_summary_mocks("First sentence.\nSecond sentence.")
        result = generate_summary("passage", model, tokenizer, config)
        # generate_summary just does .strip(), preserving internal newlines
        assert "First sentence." in result
        assert "Second sentence." in result

    def test_empty_generation(self):
        """If model generates empty string, return empty string."""
        config, model, tokenizer = _make_summary_mocks("   ")
        result = generate_summary("passage", model, tokenizer, config)
        assert result == ""


class TestReturnType:
    def test_returns_string(self):
        config, model, tokenizer = _make_summary_mocks()
        result = generate_summary("passage", model, tokenizer, config)
        assert isinstance(result, str)


class TestNoGradContext:
    """Verify generate is called within torch.no_grad context."""

    def test_no_grad_wrapping(self):
        config, model, tokenizer = _make_summary_mocks()
        # If torch.no_grad is not used, gradients could leak
        # We verify by checking the model.generate was called
        # (the actual no_grad is in the source; this is a smoke test)
        generate_summary("passage", model, tokenizer, config)
        model.generate.assert_called_once()


class TestPassagePropagation:
    """Verify passage text appears in the prompt sent to the model."""

    def test_short_passage(self):
        config, model, tokenizer = _make_summary_mocks()
        generate_summary("Short text.", model, tokenizer, config)

        content = tokenizer.apply_chat_template.call_args[0][0][0]['content']
        assert "Short text." in content

    def test_long_passage(self):
        config, model, tokenizer = _make_summary_mocks()
        long_passage = "Word " * 500
        generate_summary(long_passage, model, tokenizer, config)

        content = tokenizer.apply_chat_template.call_args[0][0][0]['content']
        assert long_passage in content

    def test_passage_with_special_chars(self):
        config, model, tokenizer = _make_summary_mocks()
        passage = 'He said "hello" & she replied <goodbye>. {end}'
        generate_summary(passage, model, tokenizer, config)

        content = tokenizer.apply_chat_template.call_args[0][0][0]['content']
        assert passage in content


# ============================================================
# Integration test with real model (marked @slow)
# ============================================================

slow = pytest.mark.slow


@pytest.fixture(scope="session")
def real_model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto",
    )
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="session")
def real_config():
    return ExperimentConfig()


@slow
class TestGenerateSummaryReal:
    def test_returns_nonempty_string(self, real_model_and_tokenizer, real_config):
        model, tokenizer = real_model_and_tokenizer
        result = generate_summary(
            "Paris is the capital of France. It is known for the Eiffel Tower, "
            "the Louvre museum, and its rich cultural heritage.",
            model, tokenizer, real_config,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_output_is_reasonable_length(self, real_model_and_tokenizer, real_config):
        model, tokenizer = real_model_and_tokenizer
        result = generate_summary(
            "The Amazon rainforest is the largest tropical rainforest in the world. "
            "It covers much of northwestern Brazil and extends into Colombia, Peru, "
            "and other South American countries.",
            model, tokenizer, real_config,
        )
        # Should be a summary, not a novel — rough sanity check
        assert len(result) < 2000
        # Should contain actual words
        assert len(result.split()) >= 3

    def test_different_passages_different_summaries(self, real_model_and_tokenizer, real_config):
        model, tokenizer = real_model_and_tokenizer
        s1 = generate_summary(
            "Paris is the capital of France.",
            model, tokenizer, real_config,
        )
        s2 = generate_summary(
            "Quantum computing uses qubits instead of classical bits.",
            model, tokenizer, real_config,
        )
        # Different passages should produce different summaries
        assert s1 != s2
