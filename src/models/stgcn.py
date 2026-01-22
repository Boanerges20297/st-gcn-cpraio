import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def normalize_adjacency(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Symmetric normalization: D^-0.5 A D^-0.5
    A = A.clone()
    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


class GraphConv(nn.Module):
    """Simple graph convolution using precomputed normalized adjacency."""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # x: (batch*T, N, F)
        # A_norm: (N, N)
        out = torch.matmul(x, self.weight)  # (batch*T, N, out)
        # apply adjacency along node dimension using einsum for correct batching
        out = torch.einsum('ij,bjk->bik', A_norm, out)
        if self.bias is not None:
            out = out + self.bias
        return out


class STGCNBlock(nn.Module):
    """A minimal ST-GCN block: GraphConv -> Temporal Conv (Conv2d with kernel (k,1))."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.gc = GraphConv(in_channels, out_channels)
        pad = (kernel_size - 1) // 2
        # Temporal conv expects input shape (B, C, T, N)
        self.tconv = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, N, F)
        B, T, N, C = x.shape
        x_ = x.reshape(B * T, N, C)  # (B*T, N, C)
        g = self.gc(x_, A_norm)  # (B*T, N, out)
        g = g.reshape(B, T, N, -1)  # (B, T, N, out)
        # temporal conv: (B, C, T, N)
        g = g.permute(0, 3, 1, 2)
        g = self.tconv(g)
        g = self.bn(g)
        g = F.relu(g)
        g = g.permute(0, 2, 3, 1)  # (B, T, N, out)
        return g


class STGCNModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, A: torch.Tensor):
        super().__init__()
        # A is adjacency matrix (N,N) numpy or tensor
        if isinstance(A, np.ndarray):
            A = torch.from_numpy(A).float()
        self.register_buffer('A_norm', normalize_adjacency(A))
        self.block1 = STGCNBlock(in_channels, hidden_channels)
        self.block2 = STGCNBlock(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, F)
        A_norm = self.A_norm
        h = self.block1(x, A_norm)
        h = self.block2(h, A_norm)
        # take last timestep
        h_last = h[:, -1, :, :]  # (B, N, hidden)
        out = self.fc(h_last)  # (B, N, out_features)
        return out
