import functools

import ninetoothed
import ninetoothed.language as ntl
import torch
from ninetoothed import Tensor

from ntops import element_wise


def default_application(input, other, output):
    output = input / other  # noqa: F841


def trunc_application(input, other, output):
    output = ntl.cast(input / other, ntl.int64)  # noqa: F841


def floor_application(input, other, output):
    output = ntl.floor(input / other)  # noqa: F841


def div(input, other, rounding_mode=None, output=None):
    if output is None:
        output = torch.empty_like(input)

    kernel = _make(input.ndim, rounding_mode)

    kernel(input, other, output)

    return output


@functools.cache
def _make(ndim, rounding_mode):
    tensors = (Tensor(ndim), Tensor(ndim), Tensor(ndim))

    if rounding_mode == "trunc":
        application = trunc_application
    elif rounding_mode == "floor":
        application = floor_application
    else:
        application = default_application

    return ninetoothed.make(element_wise.arrangement, application, tensors)
