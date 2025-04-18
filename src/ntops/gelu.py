import functools
import math

import ninetoothed
import ninetoothed.language as ntl
import torch
from ninetoothed import Tensor

from ntops import element_wise


def default_application(input, output):
    output = input * 0.5 * (1 + ntl.erf(input / ntl.sqrt(2.0)))  # noqa: F841


def tanh_application(input, output):
    input_loaded = input

    output = (  # noqa: F841
        0.5
        * input_loaded
        * (
            1
            + ntl.tanh(
                ntl.sqrt(2 / math.pi) * (input_loaded + 0.044715 * input_loaded**3)
            )
        )
    )


def gelu(input, approximate="none"):
    output = torch.empty_like(input)

    kernel = _make(input.ndim, approximate)

    kernel(input, output)

    return output


@functools.cache
def _make(ndim, approximate):
    tensors = (Tensor(ndim), Tensor(ndim))

    if approximate == "tanh":
        application = tanh_application
    else:
        application = default_application

    return ninetoothed.make(element_wise.arrangement, application, tensors)
