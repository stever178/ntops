import pytest
import torch

import ntops.torch
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_cuda(shape, dtype, atol, rtol):
    device = "cuda"

    input = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.neg(input)
    reference_output = torch.neg(input)

    assert torch.allclose(ninetoothed_output, reference_output, atol=atol, rtol=rtol)
