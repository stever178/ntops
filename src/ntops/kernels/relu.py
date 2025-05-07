import functools

import ninetoothed
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def default_application(input, output):
    output = max(0.0, input)  # noqa: F841


def True_application(input, output):
    input = max(0.0, input)  
    output = input  # noqa: F841


@functools.cache
def make(ndim, replace):
    if replace == True:
        application = True_application
    else:
        application = default_application

    tensors = (Tensor(ndim), Tensor(ndim))

    return ninetoothed.make(arrangement, application, tensors)
