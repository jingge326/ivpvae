import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Code adapted from Pyro


class ELU(nn.Module):
    """
    Exponential linear unit and its inverse.
    """
    def forward(self, x, **kwargs):
        y = F.elu(x)
        ljd = -F.relu(-x)
        return y, ljd

    def inverse(self, y, **kwargs):
        zero = torch.zeros_like(y)
        log_term = torch.log1p(y + 1e-8)
        x = torch.max(y, zero) + torch.min(log_term, zero)
        ljd = F.relu(-log_term)
        return x, ljd


class LeakyReLU(nn.Module):
    """
    Leaky ReLU and its inverse.
    For `x >= 0` returns `x`, else returns `negative_slope * x`.

    Args:
        negative_slope (float): Controls the angle of the negative slope. Default: 0.01
    """
    def __init__(self, negative_slope=0.01, **kwargs):
        super().__init__()
        assert negative_slope > 0, '`negative_slope` must be positive'
        self.negative_slope = negative_slope

    def _leaky_relu(self, x, negative_slope):
        zeros = torch.zeros_like(x)
        y = torch.max(zeros, x) + negative_slope * torch.min(zeros, x)
        ljd = torch.where(x >= 0., torch.zeros_like(x), torch.ones_like(x) * math.log(negative_slope))
        return y, ljd

    def forward(self, x, reverse=False, **kwargs):
        y, ljd = self._leaky_relu(x, 1 / self.negative_slope if reverse else self.negative_slope)
        return y, ljd

    def inverse(self, y, **kwargs):
        return self.forward(y, reverse=True, **kwargs)
