import functools

import ninetoothed
from ninetoothed import Tensor
from ninetoothed.language import libdevice

from ntops.kernels.element_wise import arrangement


def application(input, exponent, output):
    output = libdevice.pow(input, exponent)  # noqa: F841


@functools.cache
def make(ndim):
    tensors = (Tensor(ndim), Tensor(ndim), Tensor(ndim))

    return ninetoothed.make(arrangement, application, tensors)
