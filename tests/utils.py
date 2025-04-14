import math
import random

import torch


def generate_arguments():
    arguments = []

    for ndim in range(1, 5):
        for dtype in (torch.float32, torch.float16):
            if dtype is torch.float32:
                atol = 0.001
                rtol = 0.001
            else:
                atol = 0.01
                rtol = 0.01

            arguments.append((_random_shape(ndim), dtype, atol, rtol))

    return "shape, dtype, atol, rtol", arguments


def _random_shape(ndim, min_num_elements=2**8, max_num_elements=2**10):
    num_elements = random.randint(min_num_elements, max_num_elements)

    shape = []
    remaining = num_elements

    for _ in range(ndim - 1):
        size = random.randint(1, max(1, math.isqrt(remaining)))
        shape.append(size)
        remaining //= size

    shape.append(remaining)
    random.shuffle(shape)

    return shape
