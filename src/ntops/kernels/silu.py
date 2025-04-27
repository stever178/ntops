import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output):
    output = input / (1 + ntl.exp(-ntl.cast(input, ntl.float32)))  # noqa: F841


@functools.cache
def make(ndim):
    tensors = (Tensor(ndim), Tensor(ndim))

    return ninetoothed.make(arrangement, application, tensors)
