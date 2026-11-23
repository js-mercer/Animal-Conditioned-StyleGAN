import torch
import torch.nn as nn

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Force noise arrays into a uniform spherical layout
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)