import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import util


class Coupling(nn.Module):
    """
    Coupling transformation via elementwise flows. If `dim = 1`, set `mask = 'none'`.
    Splits data into 2 parts based on a mask. One part generates parameters of the flow
    that will transform the rest. Efficient (identical) computation in both directions.

    Example:
    >>> 
    >>> torch.manual_seed(123)
    >>> dim, n_bins, latent_dim = 2, 5, 50
    >>> f = st.Coupling(st.Affine(dim, st.net.MLP(dim + latent_dim, [64], 2 * dim)), mask='ordered_left_half')
    >>> f(torch.randn(1, 2), latent=torch.randn(1, latent_dim))
    tensor([[ 0.4974, -1.6288]], tensor([[0.0000, 0.3708]]
    >>> f = st.Coupling(st.Spline(dim, n_bins, st.net.MLP(dim, [64], dim * (3 * n_bins - 1))), mask='random_half')
    >>> f(torch.rand(1, dim))
    tensor([[0.9165, 0.7125]], tensor([[0.6281, -0.0000]]

    Args:
        flow (Type[stribor.flows]): Elementwise flow with `latent_net` property.
            Latent network takes input of size `dim` and returns the parameters of the flow.
        mask (str): Mask name from `stribor.util.mask`. Options: `none`,
            `ordered_right_half` (right transforms left), `ordered_left_half`, `random_half`,
            `parity_even` (even indices transform odd), `parity_odd`.
        set_data (bool, optional): Whether data has shape (..., N, dim). Default: False
    """

    def __init__(self, flow, mask, set_data=False, **kwargs):
        super().__init__()

        self.flow = flow
        self.mask_func = util.mask.get_mask(
            mask)  # Initializes mask generator
        self.set_data = set_data

    def get_mask(self, x):
        if self.set_data:
            *rest, N, D = x.shape
            return self.mask_func(N).unsqueeze(-1).expand(*rest, N, D).to(x)
        else:
            return self.mask_func(x.shape[-1]).expand_as(x).to(x)

    def forward(self, x, latent=None, reverse=False, **kwargs):
        mask = self.get_mask(x)

        z = x * mask
        if x.shape[-1] == 1:
            z = z * 0
        if latent is not None:
            z = torch.cat([z, latent], -1)

        if reverse:
            y, ljd = self.flow.inverse(x, latent=z, **kwargs)
        else:
            y, ljd = self.flow.forward(x, latent=z, **kwargs)

        y = y * (1 - mask) + x * mask
        ljd = ljd * (1 - mask)
        return y, ljd

    def inverse(self, y, latent=None, **kwargs):
        return self.forward(y, latent=latent, reverse=True, **kwargs)


class ContinuousAffineCoupling(nn.Module):
    """
    Continuous affine coupling layer. If `dim = 1`, set `mask = 'none'`.
    Similar to `Coupling` but applies only an affine transformation
    which here depends on time `t` such that it's identity map at `t = 0`.

    Example:
    >>> 
    >>> torch.manual_seed(123)
    >>> dim = 2
    >>> f = st.ContinuousAffineCoupling(st.net.MLP(dim+1, [64], 2 * dim), st.net.TimeLinear(2 * dim), 'parity_odd')
    >>> f(torch.rand(1, 2), t=torch.rand(1, 1))
    (tensor([[0.8188, 0.4037]], tensor([[-0.0000, -0.1784]])

    Args:
        latent_net (Type[nn.Module]): Inputs concatenation of `x` and `t` (and optionally
            `latent`) and outputs affine transformation parameters (size `2 * dim`)
        time_net (Type[stribor.net.time_net]): Time embedding with the same output
            size as `latent_net`
        mask (str): Mask name from `stribor.util.mask`
    """

    def __init__(self, latent_net, time_net, mask, **kwargs):
        super().__init__()

        self.latent_net = latent_net
        self.mask_func = util.mask.get_mask(
            mask)  # Initializes mask generator
        self.time_net = time_net

    def get_mask(self, x):
        return self.mask_func(x.shape[-1]).expand_as(x).to(x)

    def forward(self, x, t, latent=None, reverse=False, **kwargs):
        """
        Args:
            x (tensor): Input with shape (..., dim)
            t (tensor): Time input with shape (..., 1)
            latent (tensor): Conditioning vector with shape (..., latent_dim)
            reverse (bool, optional): Whether to calculate inverse. Default: False

        Returns:
            y (tensor): Transformed input with shape (..., dim)
            ljd (tensor): Log-Jacobian diagonal with shape (..., dim)
        """
        mask = self.get_mask(x)
        z = torch.cat([x * 0 if x.shape[-1] == 1 else x * mask, t], -1)
        if latent is not None:
            z = torch.cat([z, latent], -1)

        scale, shift = self.latent_net(z).chunk(2, dim=-1)
        t_scale, t_shift = self.time_net(t).chunk(2, dim=-1)

        if reverse:
            y = (x - shift * t_shift) * torch.exp(-scale * t_scale)
            ljd = -scale * t_scale * (1 - mask)
        else:
            y = x * torch.exp(scale * t_scale) + shift * t_shift
            ljd = scale * t_scale * (1 - mask)
        y = y * (1 - mask) + x * mask

        return y, ljd

    def inverse(self, y, t, latent=None, **kwargs):
        return self.forward(y, t=t, latent=latent, reverse=True)
