import functools

import ninetoothed
import ninetoothed.language as ntl
import torch
from ninetoothed import Tensor


def arrangement(input, output, dim):
    assert input.ndim == output.ndim

    def create_axis_tile_shape(dim, dim_block):
        return tuple(1 for _ in range(dim)) + (dim_block,) + tuple(1 for _ in range(input.ndim - dim - 1))
    
    inner_block_shape = create_axis_tile_shape(dim, input.shape[dim])
    outer_block_shape = create_axis_tile_shape(dim, -1)
    
    def arrange(input):
        input_arranged = input.tile(inner_block_shape).tile(outer_block_shape)

        input_arranged.dtype = input_arranged.dtype.squeeze(
            tuple(d for d in range(input.ndim) if d != dim)
        )
        input_arranged.dtype.dtype = input_arranged.dtype.dtype.squeeze(
            tuple(d for d in range(input.ndim) if d != dim)
        )
        return input_arranged
    
    input_arranged = arrange(input)
    output_arranged = arrange(output)

    return input_arranged, output_arranged


def application(input, output):
    for i in range(input.shape[0]):
        input_i = input[i]
        row_minus_max = input_i - ntl.max(input_i)
        numerator = ntl.exp(ntl.cast(row_minus_max, ntl.float32))
        denominator = ntl.sum(numerator)
        output[i] = numerator / denominator  # noqa: F841


def softmax(input, dim, output=None):
    if output is None:
        output = torch.empty_like(input)

    kernel = _make(input.ndim, dim)

    kernel(input, output)

    return output


@functools.cache
def _make(ndim, dim):
    return ninetoothed.make(
        functools.partial(arrangement, dim=dim),
        application,
        (
            Tensor(ndim, other=float("-inf"), shape_options={"constexpr": True}),
            Tensor(ndim),
        ),
    )
