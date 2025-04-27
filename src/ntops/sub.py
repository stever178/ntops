import functools

import ninetoothed
import torch
from ninetoothed import Tensor

from ntops import element_wise


def application(input, other, alpha, output):
    output = input - alpha * other  # noqa: F841


def sub(input, other, alpha=1, output=None):
    if output is None:
        output = torch.empty_like(input)

    kernel = _make(input.ndim)

    kernel(input, other, alpha, output)

    return output


@functools.cache
def _make(ndim):
    return ninetoothed.make(
        element_wise.arrangement,
        application,
        (Tensor(ndim), Tensor(ndim), Tensor(0), Tensor(ndim)),
    )
