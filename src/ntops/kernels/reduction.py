import ninetoothed


def arrangement(*tensors, dim, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    ndim = max(tensor.ndim for tensor in tensors)

    assert all(tensor.ndim == ndim or tensor.ndim == 0 for tensor in tensors)

    if dim < 0:
        dim += ndim

    inner_block_shape = tuple(1 if i != dim else block_size for i in range(ndim))
    outer_block_shape = tuple(1 if i != dim else -1 for i in range(ndim))

    non_target_dims = tuple(i for i in range(ndim) if i != dim)

    def _arrange(tensor):
        arranged = tensor.tile(inner_block_shape)
        arranged = arranged.tile(outer_block_shape)
        arranged.dtype = arranged.dtype.squeeze(non_target_dims)
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze(non_target_dims)

        return arranged

    return tuple(_arrange(tensor) if tensor.ndim != 0 else tensor for tensor in tensors)
