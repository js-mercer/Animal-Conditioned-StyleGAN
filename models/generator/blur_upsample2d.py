import torch
import torch.nn as nn

class BlurUpsample2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 2nd-order binomial filter kernel
        kernel = torch.tensor([
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0]
        ], dtype=torch.float32)

         # Normalise the kernel to preserve scale
        kernel = kernel / kernel.sum()

        # Register as a buffer so it moves to GPU automatically but not trained
        kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('kernel', kernel)
        self.channels = channels

    def forward(self, x):
        # Bilinear upsampling to double spatial size
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Anti-aliasing blur step via depthwise grouped convolution
        return nn.functional.conv2d(x, self.kernel, padding=1, groups=self.channels)