import ninetoothed
import torch
from ninetoothed import Tensor

from ntops.mm import BLOCK_SIZE_K, BLOCK_SIZE_M, BLOCK_SIZE_N, application


def arrangement(input, other, output):
    output_arranged = output.tile((1, BLOCK_SIZE_M, BLOCK_SIZE_N))
    output_arranged.dtype = output_arranged.dtype.squeeze(0)

    input_arranged = input.tile((1, BLOCK_SIZE_M, BLOCK_SIZE_K))
    input_arranged = input_arranged.tile((1, 1, -1))
    input_arranged = input_arranged.expand((-1, -1, output_arranged.shape[-1]))
    input_arranged.dtype = input_arranged.dtype.squeeze((0, 1))
    input_arranged.dtype.dtype = input_arranged.dtype.dtype.squeeze(0)

    other_arranged = other.tile((1, BLOCK_SIZE_K, BLOCK_SIZE_N))
    other_arranged = other_arranged.tile((1, -1, 1))
    other_arranged = other_arranged.expand((-1, output_arranged.shape[-2], -1))
    other_arranged.dtype = other_arranged.dtype.squeeze((0, 2))
    other_arranged.dtype.dtype = other_arranged.dtype.dtype.squeeze(0)

    return input_arranged, other_arranged, output_arranged


kernel = ninetoothed.make(arrangement, application, (Tensor(3), Tensor(3), Tensor(3)))


def bmm(input, other, output=None):
    b, m, _ = input.shape
    _, _, n = other.shape

    if output is None:
        output = torch.empty((b, m, n), dtype=input.dtype, device=input.device)

    kernel(input, other, output)

    return output
