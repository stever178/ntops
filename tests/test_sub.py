import pytest
import torch

import ntops.torch
from tests.skippers import skip_if_cuda_not_available
from tests.utils import gauss, generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_cuda(shape, dtype, atol, rtol):
    device = "cuda"

    input = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)
    alpha = gauss()

    ninetoothed_output = ntops.torch.sub(input, other, alpha=alpha)
    reference_output = torch.sub(input, other, alpha=alpha)

    assert torch.allclose(ninetoothed_output, reference_output, atol=atol, rtol=rtol)
