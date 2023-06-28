import torch
import torch.nn as nn

from .. import net


class InvertibleResNetBlock(nn.Module):
    """
    Single invertible ResNet block.
    """

    def __init__(self, dim, hidden_dims, activation, final_activation, n_power_iterations, **kwargs):
        super().__init__()
        def wrapper(layer): return torch.nn.utils.spectral_norm(
            layer, n_power_iterations=n_power_iterations)
        self.net = net.MLP(dim, hidden_dims, dim, activation,
                           final_activation, wrapper_func=wrapper)

    def forward(self, x):
        return x + self.net(x)

    def inverse(self, y, iterations=100):
        # fixed-point iteration
        x = y
        for _ in range(iterations):
            residual = self.net(x)
            x = y - residual
        return x


class InvertibleResNet(nn.Module):
    """
    Invertible ResNet.
    "Invertible Residual Networks" (https://arxiv.org/abs/1811.00995)

    Args:
        dim (int): Input and output size
        hidden_dims (List[int]): Hidden dimensions
        num_layers (int): Number of layers
        activation (str, optional): Activation function from `torch.nn`.
            Default: 'ReLU'
        final_activation (str, optional): Last activation. Default: None
        n_power_iterations (float, optional): Number of power iterations. Default: 5
    """

    def __init__(
        self,
        dim,
        hidden_dims,
        num_layers,
        activation='ReLU',
        final_activation=None,
        n_power_iterations=5,
        **kwargs
    ):
        super().__init__()
        blocks = []
        for _ in range(num_layers):
            blocks.append(InvertibleResNetBlock(dim, hidden_dims, activation,
                                                final_activation, n_power_iterations))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def inverse(self, y):
        for block in reversed(self.blocks):
            y = block.inverse(y)
        return y
