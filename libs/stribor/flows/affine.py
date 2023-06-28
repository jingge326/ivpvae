import torch
import torch.nn as nn

class Affine(nn.Module):
    """
    Affine flow.
    `y = a * x + b`

    Example:
    >>> torch.manual_seed(123)
    >>> dim, latent_dim = 2, 50
    >>> f = stribor.Affine(dim, stribor.net.MLP(latent_dim, [64, 64], 2 * dim))
    >>> f(torch.ones(1, dim), latent=torch.ones(1, latent_dim))
    (tensor([[0.7575, 0.9410]], tensor([[-0.1745, -0.1350]])

    Args:
        dim (int): Dimension of data
        latent_net (Type[nn.Module], optional): Neural network that takes `latent`
            and outputs transformation parameters of size `[2*dim]`. Default: None
    """
    def __init__(self, dim, latent_net=None, **kwargs):
        super().__init__()

        self.latent_net = latent_net

        if latent_net is None:
            lim = (3 / dim)**0.5 # Xavier uniform
            self.log_scale = nn.Parameter(torch.Tensor(1, dim).uniform_(-lim, lim))
            self.shift = nn.Parameter(torch.Tensor(1, dim).uniform_(-lim, lim))

    def get_params(self, latent):
        if self.latent_net is None:
            return self.log_scale, self.shift
        else:
            log_scale, shift = self.latent_net(latent).chunk(2, dim=-1)
            return log_scale, shift

    def forward(self, x, latent=None, **kwargs):
        log_scale, shift = self.get_params(latent)
        y = x * torch.exp(log_scale) + shift
        return y, log_scale.expand_as(y)

    def inverse(self, y, latent=None, **kwargs):
        log_scale, shift = self.get_params(latent)
        x = (y - shift) * torch.exp(-log_scale)
        return x, -log_scale.expand_as(x)


class AffineFixed(nn.Module):
    """
    Fixed affine transformation with predefined (non-learnable) parameters.

    Example:
    >>> f = stribor.AffineFixed(torch.tensor([2.]), torch.tensor([3.]))
    >>> f(torch.tensor([1, 2]))
    (tensor([5., 7.]), tensor([0.6931, 0.6931]))
    >>> f.inverse(torch.tensor([5, 7]))
    (tensor([1., 2.]), tensor([-0.6931, -0.6931]))

    Args:
        scale (tensor): Scaling factor, all positive values
        shift (tensor): Shifting factor
    """
    def __init__(self, scale, shift, **kwargs):
        super().__init__()

        self.scale = scale
        self.shift = shift

        assert (self.scale > 0).all(), '`scale` mush have positive values'

        self.log_scale = self.scale.log()

    def forward(self, x, **kwargs):
        y = self.scale * x + self.shift
        return y, self.log_scale.expand_as(x)

    def inverse(self, y, **kwargs):
        x = (y - self.shift) / self.scale
        return x, -self.log_scale.expand_as(x)


class AffinePLU(nn.Module):
    """
    Invertible linear layer `Wx+b` where `W=PLU` is PLU factorized.

    Args:
        dim: Dimension of input data.
    """
    def __init__(self, dim, latent_net=None, **kwargs):
        super().__init__()

        self.P = torch.eye(dim)[torch.randperm(dim)]
        self.weight = nn.Parameter(torch.empty(dim, dim))
        self.log_diag = nn.Parameter(torch.empty(1, dim))
        self.bias = nn.Parameter(torch.empty(1, dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.log_diag)
        nn.init.xavier_uniform_(self.bias)

    def get_LU(self):
        eye = torch.eye(self.weight.shape[1])
        L = torch.tril(self.weight, -1) + eye
        U = torch.triu(self.weight, 1) + eye * self.log_diag.exp()
        return L, U

    def forward(self, x, **kwargs):
        """ Input: x (..., dim) """
        L, U = self.get_LU()
        y = (self.P @ (L @ (U @ x.unsqueeze(-1)))).squeeze(-1) + self.bias
        ljd = self.log_diag.expand_as(x)
        return y, ljd

    def inverse(self, y, **kwargs):
        L, U = self.get_LU()
        y = torch.triangular_solve(self.P.T @ (y - self.bias).unsqueeze(-1), L, upper=False)[0]
        x = torch.triangular_solve(y, U, upper=True)[0].squeeze(-1)
        ljd = -self.log_diag.expand_as(x)
        return x, ljd


class MatrixExponential(nn.Module):
    """
    Matrix exponential transformation `y = exp(W*t)x`.
    Corresponds to a solution of a linear ODE `dx/dt = Wx`.

    Example:
    >>> torch.manual_seed(123)
    >>> f = stribor.MatrixExponential(2)
    >>> x = torch.rand(1, 2)
    >>> f(x, t=1.)
    (tensor([[0.0798, 1.3169]], tensor([[-0.4994,  0.4619]])
    >>> f(x, t=torch.ones(1, 1))
    (tensor([[0.0798, 1.3169]], tensor([[-0.4994,  0.4619]])

    Args:
        dim (int): Dimension of data
    """
    def __init__(self, dim, **kwargs):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, t=1., reverse=False, **kwargs):
        """
        Args:
            x (tensor): Input with shape (..., dim)
            t (tensor or float, optional): Time with shape (..., 1). Default: 1.
            reverse (bool, optional): Whether to do inverse. Default: False
        """
        if reverse:
            t = -t

        if isinstance(t, float):
            t = torch.ones(*x.shape[:-1], 1) * t
        t = t.unsqueeze(-1)

        W = torch.matrix_exp(self.weight * t)
        y = (W @ x.unsqueeze(-1)).squeeze(-1)

        ljd = (self.weight * t).diagonal(dim1=-2, dim2=-1).expand_as(x)
        return y, ljd

    def inverse(self, y, t=1., **kwargs):
        return self.forward(y, t=t, reverse=True)
