import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output):
    output = ntl.sin(input)  # noqa: F841


@functools.cache
def make(ndim):
    return ninetoothed.make(arrangement, application, (Tensor(ndim), Tensor(ndim)))
