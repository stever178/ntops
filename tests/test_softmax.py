import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_cuda(shape, dtype, atol, rtol):
    device = "cuda"

    input = torch.randn(shape, dtype=dtype, device=device)
    dim = random.randint(0, input.ndim - 1)

    ninetoothed_output = ntops.softmax(input, dim)
    reference_output = torch.nn.functional.softmax(input, dim=dim)

    assert torch.allclose(ninetoothed_output, reference_output, atol=atol, rtol=rtol)
