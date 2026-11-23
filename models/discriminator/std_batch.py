import torch
import torch.nn as nn

class std_batch(nn.Module):
    def __init__(self):
     super().__init__()

    def forward(self, x):
      # Extract the shape of the input tensor
      batch, channels, height, width = x.shape

      # Compute the standard deviation across the batch dimension
      std = torch.sqrt(torch.var(x, dim=0, keepdim=True, correction=0) + 1e-8) 
      # Average across channels
      std_mean = torch.mean(std, dim=1, keepdim=True) 
      # Match dimension of input tensor
      std_map = std_mean.expand(batch, 1, height, width)

      return torch.cat([x, std_map], dim=1)