import functools

import ninetoothed
import torch
from ninetoothed import Tensor

from ntops import element_wise


def application(input, other, output):
    output = input * other  # noqa: F841


def mul(input, other, output=None):
    if output is None:
        output = torch.empty_like(input)

    kernel = _make(input.ndim)

    kernel(input, other, output)

    return output


@functools.cache
def _make(ndim):
    tensors = (Tensor(ndim), Tensor(ndim), Tensor(ndim))

    return ninetoothed.make(element_wise.arrangement, application, tensors)
