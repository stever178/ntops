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
    block_size_m=mm.BLOCK_SIZE_M,
    block_size_n=mm.BLOCK_SIZE_N,
    block_size_k=mm.BLOCK_SIZE_K,
):
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
