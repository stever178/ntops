import functools

import ninetoothed
import ninetoothed.language as ntl
import torch
from ninetoothed import Tensor

from ntops import element_wise


def application(input, output):
    output = ntl.rsqrt(ntl.cast(input, ntl.float32))  # noqa: F841


def rsqrt(input, output=None):
    if output is None:
        output = torch.empty_like(input)

    kernel = _make(input.ndim)

    kernel(input, output)

    return output


@functools.cache
def _make(ndim):
    return ninetoothed.make(
        element_wise.arrangement,
        application,
        (Tensor(ndim), Tensor(ndim)),
    )
