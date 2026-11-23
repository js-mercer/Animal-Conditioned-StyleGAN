import torch
import torch.nn as nn

from models.generator.blur_upsample2d import BlurUpsample2d
from models.generator.noise_injection import NoiseInjection
from models.generator.adain import AdaIN

class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim=512, upsample=True):
      super().__init__()
      self.upsample = upsample

      # Bilinear upsampling
      if self.upsample:
          self.upsample_layer = BlurUpsample2d(in_channels)

      # Conv layers add in local features without changing size
      self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
      # Inject noise
      self.noise1 = NoiseInjection(out_channels)
      # Inject style latent vector w
      self.adain1 = AdaIN(out_channels, latent_dim)

      self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
      self.noise2 = NoiseInjection(out_channels)
      self.adain2 = AdaIN(out_channels, latent_dim)

      self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, w):
      # double size of image
      if self.upsample:
          x = self.upsample_layer(x)

      # Apply convolution, inject noise, inject latent vector w, apply activation function
      x = self.conv1(x)
      x = self.noise1(x)
      x = self.adain1(x, w)
      x = self.leaky(x)

      x = self.conv2(x)
      x = self.noise2(x)
      x = self.adain2(x, w)
      x = self.leaky(x)

      return x