"""
Streaming metrics.
"""

import torch


class RunningPearson:
    """
    Streaming Pearson correlation per dimension without storing full predictions.

    Maintains sufficient statistics per voxel/unit:
      sum_x, sum_y, sum_x2, sum_y2, sum_xy, n
    """
    def __init__(self, dim: int, device: torch.device, eps: float = 1e-8):
        self.dim = dim
        self.device = device
        self.eps = eps
        self.reset()

    def reset(self):
        d = self.dim
        dev = self.device
        self.n = 0
        self.sum_x = torch.zeros(d, device=dev)
        self.sum_y = torch.zeros(d, device=dev)
        self.sum_x2 = torch.zeros(d, device=dev)
        self.sum_y2 = torch.zeros(d, device=dev)
        self.sum_xy = torch.zeros(d, device=dev)

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor):
        # x,y: (B, V)
        b = x.shape[0]
        self.n += b
        self.sum_x += x.sum(dim=0)
        self.sum_y += y.sum(dim=0)
        self.sum_x2 += (x * x).sum(dim=0)
        self.sum_y2 += (y * y).sum(dim=0)
        self.sum_xy += (x * y).sum(dim=0)

    @torch.no_grad()
    def corr(self) -> torch.Tensor:
        n = float(self.n)
        num = n * self.sum_xy - self.sum_x * self.sum_y
        den_x = n * self.sum_x2 - self.sum_x * self.sum_x
        den_y = n * self.sum_y2 - self.sum_y * self.sum_y
        den = torch.sqrt(torch.clamp(den_x, min=0.0) + self.eps) * torch.sqrt(torch.clamp(den_y, min=0.0) + self.eps)
        return num / (den + self.eps)
