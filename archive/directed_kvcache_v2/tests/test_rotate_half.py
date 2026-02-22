"""Direct tests for _rotate_half."""

import torch
import pytest
from lib.kv_cache import _rotate_half


class TestRotateHalf:
    def test_shape_preserved(self):
        x = torch.randn(2, 4, 8, 16)
        result = _rotate_half(x)
        assert result.shape == x.shape

    def test_known_value(self):
        """[1,2,3,4] -> [-3,-4,1,2]"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = _rotate_half(x)
        expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
        torch.testing.assert_close(result, expected)

    def test_double_application_negates(self):
        """rotate_half(rotate_half(x)) == -x (rotation by pi)."""
        x = torch.randn(1, 2, 3, 8)
        result = _rotate_half(_rotate_half(x))
        torch.testing.assert_close(result, -x)

    def test_batch_multi_head(self):
        """Works with batch and multi-head dimensions."""
        x = torch.randn(3, 8, 12, 64)
        result = _rotate_half(x)
        assert result.shape == (3, 8, 12, 64)
        # First half of result should be -second half of input
        half = 32
        torch.testing.assert_close(result[..., :half], -x[..., half:])
        torch.testing.assert_close(result[..., half:], x[..., :half])

    def test_1d_input(self):
        x = torch.tensor([1.0, 2.0])
        result = _rotate_half(x)
        expected = torch.tensor([-2.0, 1.0])
        torch.testing.assert_close(result, expected)
