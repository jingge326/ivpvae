import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


class Flow(nn.Module):
    """
    Building both normalizing flows and neural flows.

    Example:
    >>> 
    >>> torch.manual_seed(123)
    >>> dim = 2
    >>> flow = st.Flow(st.UnitNormal(dim), [st.Affine(dim)])
    >>> x = torch.rand(1, dim)
    >>> y, ljd = flow(x)
    >>> y_inv, ljd_inv = flow.inverse(y)

    Args:
        base_dist (Type[torch.distributions]): Base distribution
        transforms (List[st.flows]): List of invertible transformations
    """

    def __init__(self, base_dist=None, transforms=[]):
        super().__init__()
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x, latent=None, mask=None, t=None, reverse=False, **kwargs):
        """
        Args:
            x (tensor): Input sampled from base density with shape (..., dim)
            latent (tensor, optional): Conditional vector with shape (..., latent_dim)
                Default: None
            mask (tensor): Masking tensor with shape (..., 1)
                Default: None
            t (tensor, optional): Flow time end point. Default: None
            reverse (bool, optional): Whether to perform an inverse. Default: False

        Returns:
            y (tensor): Output that follows target density (..., dim)
            log_jac_diag (tensor): Log-Jacobian diagonal (..., dim)
        """
        transforms = self.transforms[::-1] if reverse else self.transforms
        _mask = 1 if mask is None else mask

        log_jac_diag = torch.zeros_like(x).to(x)
        for f in transforms:
            if reverse:
                x, ld = f.inverse(x * _mask, latent=latent,
                                  mask=mask, t=t, **kwargs)
            else:
                x, ld = f.forward(x * _mask, latent=latent,
                                  mask=mask, t=t, **kwargs)
            log_jac_diag += ld * _mask
        return x, log_jac_diag

    def inverse(self, y, latent=None, mask=None, t=None, **kwargs):
        """ Inverse of forward function with the same arguments. """
        return self.forward(y, latent=latent, mask=mask, t=t, reverse=True, **kwargs)

    def log_prob(self, x, **kwargs):
        """
        Calculates log-probability of a sample.

        Args:
            x (tensor): Input with shape (..., dim)

        Returns:
            log_prob (tensor): Log-probability of the input with shape (..., 1)
        """
        if self.base_dist is None:
            raise ValueError(
                'Please define `base_dist` if you need log-probability')
        x, log_jac_diag = self.inverse(x, **kwargs)
        log_prob = self.base_dist.log_prob(x) + log_jac_diag.sum(-1)
        return log_prob.unsqueeze(-1)

    def sample(self, num_samples, latent=None, mask=None, **kwargs):
        """
        Transforms samples from the base to the target distribution.
        Uses reparametrization trick.

        Args:
            num_samples (tuple or int): Shape of samples
            latent (tensor): Latent conditioning vector with shape (..., latent_dim)

        Returns:
            x (tensor): Samples from target distribution with shape (*num_samples, dim)
        """
        if self.base_dist is None:
            raise ValueError('Please define `base_dist` if you need sampling')
        if isinstance(num_samples, int):
            num_samples = (num_samples,)

        x = self.base_dist.rsample(num_samples)
        x, log_jac_diag = self.forward(x, **kwargs)
        return x
