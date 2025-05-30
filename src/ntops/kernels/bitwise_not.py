import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def bitwise_application(input, output):
    output = ~input  # noqa: F841


def logical_application(input, output):
    output = ntl.where(input, False, True)  # noqa: F841


@functools.cache
def make(ndim, logical=False):
    tensors = (Tensor(ndim), Tensor(ndim))

    application = logical_application if logical else bitwise_application

    return ninetoothed.make(arrangement, application, tensors)
