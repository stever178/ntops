import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

import ntops.kernels.mm as mm


def arrangement(
    input,
    x,
    y,
    beta,
    alpha,
    output,
    block_size_m=None,
    block_size_n=None,
    block_size_k=None,
):
    if block_size_m is None:
        block_size_m = mm.BLOCK_SIZE_M

    if block_size_n is None:
        block_size_n = mm.BLOCK_SIZE_N

    if block_size_k is None:
        block_size_k = mm.BLOCK_SIZE_K

    _, _, input_arranged = mm.arrangement(
        x,
        y,
        input,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )

    x_arranged, y_arranged, output_arranged = mm.arrangement(x, y, output)

    return input_arranged, x_arranged, y_arranged, beta, alpha, output_arranged


def application(input, x, y, beta, alpha, output):
    mm_output = ntl.zeros(output.shape, dtype=ntl.float32)
    mm.application(x, y, mm_output)
    output = beta * input + alpha * mm_output


def premake(dtype=None, block_size_m=None, block_size_n=None, block_size_k=None):
    arrangement_ = functools.partial(
        arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )

    tensors = (
        Tensor(2, dtype=dtype),
        Tensor(2, dtype=dtype),
        Tensor(2, dtype=dtype),
        Tensor(0, dtype=dtype),
        Tensor(0, dtype=dtype),
        Tensor(2, dtype=dtype),
    )

    return arrangement_, application, tensors
