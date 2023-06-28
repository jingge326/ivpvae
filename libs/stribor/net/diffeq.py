import torch
import torch.nn as nn

from .. import net

# Regular nets, use `divergence=approximate` in CNF


class DiffeqConcat(nn.Module):
    """
    Differential equation that concatenates the input and time.

    Args:
        net (Type[nn.Module]): Neural network that concatenates
            input `x`, time `t` and `latent` (optional) and outputs
            the derivative of the same size as `x`.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, t, x, latent=None, **kwargs):
        """
        Args:
            t (tensor): Time with shape (..., 1)
            x (tensor): Input with shape (..., dim)
            latent (tensor, optional): Latent vector with shape (..., latent_dim).
                Default: None

        Returns:
            dx (tensor): Derivative in `x` with shape (..., dim)
        """
        t = torch.ones_like(x[..., :1]) * t
        input = torch.cat([t, x], -1)
        if latent is not None:
            input = torch.cat([input, latent], -1)
        return self.net(input, **kwargs)


class DiffeqMLP(DiffeqConcat):
    """
    Differential equation defined with MLP.

    Example:
    >>> batch, dim = 32, 3
    >>> net = stribor.net.DiffeqMLP(dim + 1, [64, 64], dim)
    >>> x = torch.randn(batch, dim)
    >>> t = torch.rand(batch, 1)
    >>> net(t, x).shape
    torch.Size([32, 3])

    Args:
        Same as in `st.net.MLP`
    """

    def __init__(self, in_dim, hidden_dims, out_dim, activation='Tanh', final_activation=None, **kwargs):
        super().__init__(net.MLP(in_dim, hidden_dims, out_dim, activation, final_activation))


class DiffeqDeepset(DiffeqConcat):
    """
    Differential equation defined with permutation equivariant network.

    Args:
        Same as in `st.net.EquivariantNet`
    """

    def __init__(self, in_dim, hidden_dims, out_dim, activation='Tanh', final_activation=None, **kwargs):
        super().__init__(net.EquivariantNet(
            in_dim, hidden_dims, out_dim, activation, final_activation))


class DiffeqSelfAttention(DiffeqConcat):
    """
    Differential equation defined with self attention network.

    Args:
        Same as in `st.net.SelfAttention`
    """

    def __init__(self, in_dim, hidden_dim, out_dim, n_heads=1, mask_diagonal=False, **kwargs):
        super().__init__(net.SelfAttention(
            in_dim, hidden_dim, out_dim, n_heads, mask_diagonal))
