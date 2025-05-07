import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def default_application(input, other, output):
    output = input / other  # noqa: F841


def trunc_application(input, other, output):
    output = ntl.cast(input / other, ntl.int64)  # noqa: F841


def floor_application(input, other, output):
    output = ntl.floor(input / other)  # noqa: F841


@functools.cache
def make(ndim, rounding_mode):
    if rounding_mode == "trunc":
        application = trunc_application
    elif rounding_mode == "floor":
        application = floor_application
    else:
        application = default_application

    tensors = (Tensor(ndim), Tensor(ndim), Tensor(ndim))

    return ninetoothed.make(arrangement, application, tensors)
