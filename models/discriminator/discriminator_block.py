import torch
import torch.nn as nn

from models.discriminator.blur_pool2d import BlurPool2d

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
      super().__init__()

      self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
      self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
      self.downsample_features = BlurPool2d(out_channels)
      self.downsample_residual = BlurPool2d(out_channels)
      self.leaky = nn.LeakyReLU(0.2)

      self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
      # First layer feature extraction
      out = self.leaky(self.conv1(x)) 
      # Second layer out channels are changed
      out = self.leaky(self.conv2(out))
      # Downsample the features using BlurPool2d
      out = self.downsample_features(out) 

      # Skip connection for the residual path
      residual = self.skip_conv(x) 
      # # Downsample to match size
      residual = self.downsample_residual(residual)

       # Scale to preserve variance
      return (out + residual) * 0.7071