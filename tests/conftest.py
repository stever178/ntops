import random

import torch


def pytest_configure():
    _set_random_seed(42)


def _set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
