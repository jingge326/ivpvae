import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Code adapted from torch.distributions.transforms


class Sigmoid(nn.Module):
    """
    Sigmoid transformation.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, **kwargs):
        finfo = torch.finfo(x.dtype)
        y = torch.clamp(torch.sigmoid(x), min=finfo.tiny, max=1. - finfo.eps)
        ljd = -F.softplus(-x) - F.softplus(x)
        return y, ljd

    def inverse(self, y, **kwargs):
        finfo = torch.finfo(y.dtype)
        y = y.clamp(min=finfo.tiny, max=1.0 - finfo.eps)

        x = y.log() - (-y).log1p()
        ljd = -y.log() - (-y).log1p()
        return x, ljd

class Logit(nn.Module):
    """
    Logit transformation. Inverse of sigmoid.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.base_flow = Sigmoid(**kwargs)

    def forward(self, x, **kwargs):
        return self.base_flow.inverse(x)

    def inverse(self, x, **kwargs):
        return self.base_flow.forward(x)
