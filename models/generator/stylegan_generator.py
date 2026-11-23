import torch
import torch.nn as nn

from models.generator.conditional_mapping_network import ConditionalMappingNetwork
from models.generator.generator_block import GeneratorBlock

class StyleGANGenerator(nn.Module):
    def __init__(self, latent_dim=512, num_classes=3):
      super().__init__()

      # Map latent space into disentangled space with embedded classes
      self.mapping = ConditionalMappingNetwork(latent_dim, num_classes)

      # Constant 4x4 learnable starting tensor that is built into a 128x128 image
      self.const_starting_tensor = nn.Parameter(torch.randn(1, 512, 4, 4))

      # Upsample and apply features to custom image, reduce channels down from 512 to 32
      self.block_4x4 = GeneratorBlock(512, 512, latent_dim, upsample=False)
      self.block_8x8 = GeneratorBlock(512, 256, latent_dim, upsample=True)
      self.block_16x16 = GeneratorBlock(256, 256, latent_dim, upsample=True)
      self.block_32x32 = GeneratorBlock(256, 128, latent_dim, upsample=True)
      self.block_64x64 = GeneratorBlock(128, 64, latent_dim, upsample=True)
      self.block_128x128 = GeneratorBlock(64, 32, latent_dim, upsample=True)

      # Final conv layer reduces to 3 512x512 channels for colour images
      self.to_rgb = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, z, labels, inject_w=None, mix_prob=0.9):
      batch_size = z.size(0)

      # If manual interpolation skip latent space mapping
      if inject_w is not None:
        w1 = inject_w
      else:
        w1 = self.mapping(z, labels)

      # Decide whether to apply style mixing for this batch step
      if self.training and torch.rand(1).item() < mix_prob:
          # Generate a secondary independent style vector
          z2 = torch.randn_like(z)
          w2 = self.mapping(z2, labels)
          # Pick a random block index to switch styles (between block 2 and 6)
          crossover = torch.randint(2, 7, (1,)).item()
      else:
          w2 = w1
          crossover = 7 # no crossover

      x = self.const_starting_tensor.expand(batch_size, -1, -1, -1)

      x = self.block_4x4(x, w1)
      x = self.block_8x8(x, w1 if crossover > 2 else w2)
      x = self.block_16x16(x, w1 if crossover > 3 else w2)
      x = self.block_32x32(x, w1 if crossover > 4 else w2)
      x = self.block_64x64(x, w1 if crossover > 5 else w2)
      x = self.block_128x128(x, w1 if crossover > 6 else w2)

      rgb_image = self.to_rgb(x)

      # tanh activation to scale between [-1, 1]
      return torch.tanh(rgb_image)
    