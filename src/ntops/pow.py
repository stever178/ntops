import functools

import ninetoothed
import torch
from ninetoothed import Tensor
from ninetoothed.language import libdevice

from ntops import element_wise


def application(input, exponent, output):
    output = libdevice.pow(input, exponent)  # noqa: F841


def pow(input, exponent, output=None):
    if output is None:
        output = torch.empty_like(input)

    kernel = _make(input.ndim)

    kernel(input, exponent, output)

    return output


@functools.cache
def _make(ndim):
    tensors = (Tensor(ndim), Tensor(ndim), Tensor(ndim))

    return ninetoothed.make(element_wise.arrangement, application, tensors)
