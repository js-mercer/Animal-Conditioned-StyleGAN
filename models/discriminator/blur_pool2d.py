import torch
import torch.nn as nn
import torch.nn.functional as F

class BlurPool2d(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        # Filter kernel: [1, 2, 1] x [1, 2, 1]^T
        kernel = torch.tensor([
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0]
        ], dtype=torch.float32)

        # Normalize the kernel to preserve scale
        kernel = kernel / kernel.sum()

        # Register as a buffer so it moves to GPU automatically but isn't trained
        kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('kernel', kernel)
        self.channels = channels

    def forward(self, x):
        # Apply the blur filter
        x = F.conv2d(x, self.kernel, padding=1, groups=self.channels)

        # Downsample by factor of 2
        return F.avg_pool2d(x, kernel_size=2, stride=2)