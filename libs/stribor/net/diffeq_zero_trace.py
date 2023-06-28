import torch
import torch.nn as nn

from .. import net

# Volume preserving - zero trace nets, use `divergence=exact` in CNF


class DiffeqZeroTraceMLP(nn.Module):
    """
    Zero trace MLP transformation based on MADE network.

    Example:
    >>> dim = 3
    >>> net = stribor.net.DiffeqZeroTraceMLP(dim, [64], dim, return_log_det_jac=False)
    >>> x = torch.randn(32, dim)
    >>> t = torch.rand(32, 1)
    >>> net(t, x).shape
    torch.Size([32, 3])

    Args:
        in_dim (int): Input size
        hidden_dims (List[int]): Hidden dimensions
        out_dim (int): Output size
        return_log_det_jac (bool, optional): Whether to return the log-Jacobian
            diagonal values (always zero). Default: True
    """

    def __init__(self, in_dim, hidden_dims, out_dim, return_log_det_jac=True, **kwargs):
        super().__init__()
        self.return_log_det_jac = return_log_det_jac
        self.net1 = net.MADE(in_dim, hidden_dims, out_dim, natural_ordering=True,
                             reverse_ordering=False, return_per_dim=True)
        self.net2 = net.MADE(in_dim, hidden_dims, out_dim, natural_ordering=True,
                             reverse_ordering=True, return_per_dim=True)

    def forward(self, t, x, **kwargs):
        y = self.net1(x, **kwargs) + self.net2(x, **kwargs)
        y = y.reshape(*y.shape[:-2], -1)
        return (y, torch.zeros_like(x)) if self.return_log_det_jac else y


def exclusive_sum_pooling(x, mask):
    emb = x.sum(-2, keepdims=True)
    return emb - x


def exclusive_mean_pooling(x, mask):
    emb = exclusive_sum_pooling(x, mask)
    N = mask.sum(-2, keepdim=True)
    y = emb / torch.max(N - 1, torch.ones_like(N))[0]
    return y


def exclusive_max_pooling(x, mask):
    if x.shape[-2] == 1:  # If only one element in set
        return torch.zeros_like(x)

    first, second = torch.topk(x, 2, dim=-2).values.chunk(2, dim=-2)
    indicator = (x == first).float()
    y = (1 - indicator) * first + indicator * second
    return y


class ZeroTraceEquivariantEncoder(nn.Module):
    """
    Pooling layer with zero trace Jacobian.

    Args:
        in_dim (int): Input size
        hidden_dims (List[int]): Hidden dimensions
        out_dim (int): Output size
    """

    def __init__(self, in_dim, hidden_dims, out_dim, pooling, **kwargs):
        super().__init__()
        self.pooling = pooling
        self.in_dim = in_dim
        self.set_emb = net.DiffeqMLP(in_dim + 1, hidden_dims, out_dim)

    def forward(self, t, x, mask=None, **kwargs):
        if mask is None:
            mask = torch.ones(*x.shape[:-1], 1)
        else:
            mask = mask[..., 0, None]

        x = self.set_emb(t, x) * mask
        if self.pooling == 'mean':
            y = exclusive_mean_pooling(x, mask)
        elif self.pooling == 'max':
            y = exclusive_max_pooling(x, mask)
        elif self.pooling == 'sum':
            y = exclusive_sum_pooling(x, mask)
        y = y.unsqueeze(-2).repeat_interleave(self.in_dim, dim=-2)
        return y


class DiffeqZeroTraceDeepSet(nn.Module):
    """
    Zero trace deepset transformation.

    Args:
        in_dim (int): Input size
        hidden_dims (List[int]): Hidden dimensions
        out_dim (int): Output size
        pooling (str, optional): Which pooling to use. Default: 'max'
        return_log_det_jac (bool, optional): Whether to return the log-Jacobian
            diagonal values (always zero). Default: True
    """

    def __init__(self, in_dim, hidden_dims, out_dim, pooling='max', return_log_det_jac=True, **kwargs):
        super().__init__()
        self.elementwise = net.MADE(
            in_dim, hidden_dims, out_dim, return_per_dim=True)
        self.interaction = ZeroTraceEquivariantEncoder(
            in_dim, hidden_dims, out_dim // in_dim, pooling)
        self.return_log_det_jac = return_log_det_jac

    def forward(self, t, x, mask=None, latent=None, **kwargs):
        if latent is not None:
            x = torch.cat([x, latent], -1)
        if mask is None:
            mask = torch.ones(*x.shape[:-1], 1).to(x)
        y = self.elementwise(x) + self.interaction(t, x, mask=mask)
        y = y * mask.unsqueeze(-1)
        y = y.reshape(*y.shape[:-2], -1)
        return (y, torch.zeros_like(x)) if self.return_log_det_jac else y


class DiffeqZeroTraceAttention(nn.Module):
    """
    Zero trace attention transformation.

    Args:
        in_dim (int): Input size
        hidden_dims (List[int]): Hidden dimensions
        out_dim (int): Output size
        n_heads (int, optional): Number of heads in multihead attention. Default: 1
        return_log_det_jac (bool, optional): Whether to return the log-Jacobian
            diagonal values (always zero). Default: True
    """

    def __init__(self, in_dim, hidden_dims, out_dim, n_heads=1, return_log_det_jac=True, **kwargs):
        super().__init__()
        self.n_heads = n_heads
        self.return_log_det_jac = return_log_det_jac

        self.q = net.MADE(
            in_dim, hidden_dims[:-1], hidden_dims[-1] * in_dim, return_per_dim=True)
        self.k = net.MLP(in_dim, hidden_dims[:-1], hidden_dims[-1])
        self.v = net.MLP(
            in_dim, hidden_dims[:-1], hidden_dims[-1], return_per_dim=True)
        self.proj = net.MLP(hidden_dims[-1], [], out_dim // in_dim)

    def forward(self, t, x, mask=None, **kwargs):
        query = self.q(x).transpose(-2, -3)  # (B, N, D) -> (B, D, N, H)
        # value = self.v(x).transpose(-2, -3)
        # (B, D, N, H)
        key = self.k(
            x).unsqueeze(-2).repeat_interleave(x.shape[-1], dim=-2).transpose(-2, -3)
        value = self.v(x).unsqueeze(-2).repeat_interleave(
            x.shape[-1], dim=-2).transpose(-2, -3)  # (B, D, N, H)

        y = net.attention(query, key, value, self.n_heads,
                          True, mask)  # (B, D, N, H)
        y = y.transpose(-2, -3)  # (B, N, D, H)
        # (B, N, D, O) -> (B, N, D * O)
        y = self.proj(y).view(*y.shape[:-2], -1)

        return (y, torch.zeros_like(x)) if self.return_log_det_jac else y
