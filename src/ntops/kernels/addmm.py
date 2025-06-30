import functools

import ninetoothed
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


@functools.cache
def make():
    tensors = (Tensor(2), Tensor(2), Tensor(2), Tensor(0), Tensor(0), Tensor(2))

    return ninetoothed.make(arrangement, application, tensors)
