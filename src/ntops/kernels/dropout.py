import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, p, seed, output):
    output = ntl.where(ntl.rand(seed, input.offsets()) > p, input / (1 - p), 0)  # noqa: F841


@functools.cache
def make(ndim):
    tensors = (Tensor(ndim), Tensor(0), Tensor(0), Tensor(ndim))

    return ninetoothed.make(arrangement, application, tensors)
