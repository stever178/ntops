import functools
import math

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


def application(input, weight, eps, output, num_normalized_elements):
    _rms = ntl.zeros(input.dtype.shape, dtype=ntl.float32)

    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        _rms += input_i * input_i

    rms = ntl.sqrt(ntl.sum(_rms) / num_normalized_elements + eps)

    for i in range(input.shape[0]):
        output[i] = input[i] / rms * weight[i]


def premake(ndim, normalized_shape, dtype=None, block_size=None):
    dims = tuple(-(dim + 1) for dim in range(len(normalized_shape)))

    arrangement_ = functools.partial(arrangement, dim=dims, block_size=block_size)

    tensors = (
        Tensor(ndim, other=0, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(0, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(0, dtype=dtype, constexpr=True, value=math.prod(normalized_shape)),
    )

    return arrangement_, application, tensors
