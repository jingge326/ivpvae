import torch
import torch.nn as nn
import torch.nn.functional as F

class Flip(nn.Module):
    """
    Flip indices transformation.

    Example:
    >>> f = stribor.Flip()
    >>> x = torch.tensor([[1, 2], [3, 4]])
    >>> f(x)[0]
    tensor([[2, 1],
            [4, 3]])
    >>> f = stribor.Flip([0, 1])
    >>> f(x)[0]
    tensor([[4, 3],
            [2, 1]])

    Args:
        dims (List[int]): Dimensions along which to flip the order of values.
            Default: [-1]
    """
    def __init__(self, dims=[-1]):
        super().__init__()
        self.dims = dims

    def forward(self, x, **kwargs):
        y = torch.flip(x, self.dims)
        return y, torch.zeros_like(y)

    def inverse(self, y, **kwargs):
        x = torch.flip(y, self.dims)
        return x, torch.zeros_like(x)


class Permute(nn.Module):
    """
    Permute indices along the last dimension.

    Example:
    >>> torch.manual_seed(123)
    >>> f = stribor.Permute(3)
    >>> f(torch.tensor([1, 2, 3]))
    (tensor([2, 3, 1]), tensor([0, 0, 0]))
    >>> f.inverse(torch.tensor(tensor([2, 3, 1])))
    (tensor([1, 2, 3]), tensor([0, 0, 0]))

    Args:
        dim (int): Dimension of data
    """
    def __init__(self, dim):
        super().__init__()
        self.permutation = torch.randperm(dim)

        self.inverse_permutation = torch.empty(dim).long()
        self.inverse_permutation[self.permutation] = torch.arange(dim)

    def forward(self, x):
        y = x[..., self.permutation]
        return y, torch.zeros_like(y)

    def inverse(self, y):
        x = y[..., self.inverse_permutation]
        return x, torch.zeros_like(x)
