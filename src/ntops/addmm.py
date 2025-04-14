import ninetoothed
import ninetoothed.language as ntl
import torch
from ninetoothed import Tensor

import ntops.mm as mm


def arrangement(input, x, y, beta, alpha, output):
    _, _, input_arranged = mm.arrangement(x, y, input)

    x_arranged, y_arranged, output_arranged = mm.arrangement(x, y, output)

    return input_arranged, x_arranged, y_arranged, beta, alpha, output_arranged


def application(input, x, y, beta, alpha, output):
    mm_output = ntl.zeros(output.shape, dtype=ntl.float32)
    mm.application(x, y, mm_output)
    output = beta * input + alpha * mm_output


tensors = (Tensor(2), Tensor(2), Tensor(2), Tensor(0), Tensor(0), Tensor(2))
kernel = ninetoothed.make(arrangement, application, tensors)


def addmm(input, x, y, beta, alpha, output=None):
    m, _ = x.shape
    _, n = y.shape

    if output is None:
        output = torch.empty((m, n), dtype=input.dtype, device=input.device)

    kernel(input, x, y, beta, alpha, output)

    return output
