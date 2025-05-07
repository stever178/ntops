import functools

import ninetoothed
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, other, alpha, output):
    output = input + alpha * other  # noqa: F841


@functools.cache
def make(ndim):
    tensors = (Tensor(ndim), Tensor(ndim), Tensor(0), Tensor(ndim))

    return ninetoothed.make(arrangement, application, tensors)
