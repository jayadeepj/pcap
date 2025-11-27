import inspect
import torch
import random
import os
import numpy as np


def disp(snippet):
    print(inspect.getsource(snippet))


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def linspace(lo: torch.Tensor, hi: torch.Tensor, num: int):
    """  Creates a torch tensor of shape [num, *lo.shape] with evenly spaced values from lo -> hi.
    Similar to np.linspace but for multidimensional tensor in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=lo.device) / (num - 1)

    for i in range(lo.ndim):
        steps = steps.unsqueeze(-1)

    # start at 'lo' and increment til 'hi' in each dim.
    return lo[None] + steps * (hi - lo)[None]


def tensor_tuple_to_tensor(_tup):
    """ Convert a tuple of tensors into a combined tensor.

        In {_tup} Shape: (tensor(1), tensor(1), ..... N)
        Out Shape : tensor [N x 1]"""
    assert type(_tup) == tuple
    assert _tup[0].shape == torch.Size([1])
    assert all([_t.shape == _tup[0].shape for _t in _tup]) is True
    return torch.cat(_tup).unsqueeze(0)


def tensor_to_tuple(_ten):
    """ Convert a tensor into a split tuples tensor.

        In {_ten} Shape: tensor [N x 1]
        Out tuple Shape : (tensor(1), tensor(1), ..... N)"""

    assert _ten.shape[0] == 1  # change 2 to the number of parameters
    return torch.split(_ten.squeeze(0), 1)


def np_tuple_to_tensor(row):
    """ Convert a tuple of np arrays into a combined tensor.

        In {_tup} Shape: (ndarray(1), ndarray(1), ..... N)
        Out Shape : tensor [N x 1]"""

    _tup = tuple(torch.tensor(e) for e in row)
    assert _tup[0].shape == torch.Size([1])
    assert all([_t.shape == _tup[0].shape for _t in _tup]) is True
    return tensor_tuple_to_tensor(_tup).squeeze(0)
