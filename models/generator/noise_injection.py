import torch
import torch.nn as nn

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # One learnable weight parameter per channel, initialised to zero
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # Generate zero-mean, unit-variance Gaussian noise
        noise = torch.randn_like(x[:, :1, :, :])

        # Scale the noise by the learned channel weights and add it to the features
        return x + self.weight * noise