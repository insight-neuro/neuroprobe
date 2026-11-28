import numpy as np
import torch

from examples.eval_utils import remove_line_noise


def test_line_noise_tensor_output_matches_numpy_filter():
    values = np.random.default_rng(42).normal(size=(2, 2048)).astype(np.float32)
    expected = remove_line_noise(values)
    actual = remove_line_noise(torch.from_numpy(values))
    assert isinstance(actual, torch.Tensor)
    np.testing.assert_array_equal(actual.numpy(), expected)
