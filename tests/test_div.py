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
    other = torch.randn(shape, dtype=dtype, device=device)

    for rounding_mode in (None, "trunc", "floor"):
        # TODO: Test for `trunc` mode later.
        if rounding_mode == "trunc":
            continue

        ninetoothed_output = ntops.torch.div(input, other, rounding_mode=rounding_mode)
        reference_output = torch.div(input, other, rounding_mode=rounding_mode)

        assert torch.allclose(
            ninetoothed_output, reference_output, atol=atol, rtol=rtol
        )
