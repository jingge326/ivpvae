import torch
import torch.nn as nn

class Identity(nn.Module):
    """
    Identity transformation.
    Doesn't change the input, log-Jacobian is 0.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, **kwargs):
        return x, torch.zeros_like(x)

    def inverse(self, y, **kwargs):
        return y, torch.zeros_like(y)
