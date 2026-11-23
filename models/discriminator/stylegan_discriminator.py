import torch
import torch.nn as nn

from models.discriminator.discriminator_block import DiscriminatorBlock
from models.discriminator.std_batch import std_batch

class StyleGANDiscriminator(nn.Module):
    def __init__(self):
      super(StyleGANDiscriminator, self).__init__()

      # Expand 3 colour channels to 32 channels
      self.conv_in = nn.Conv2d(3, 32, kernel_size=1) # 128x128x3 -> 128x128x32
      self.leaky = nn.LeakyReLU(0.2)

      # Extract feature size and downsample
      self.block1 = DiscriminatorBlock(32, 64) # 128x128x32 -> 64x64x64
      self.block2 = DiscriminatorBlock(64, 128) # 64x64x64 -> 32x32x128
      self.block3 = DiscriminatorBlock(128, 256) # 32x32x128 -> 16x16x256
      self.block4 = DiscriminatorBlock(256, 256) # 16x16x256 -> 8x8x512
      self.block5 = DiscriminatorBlock(256, 512) # 8x8x512 -> 4x4x512

      # Measure std across batch and add as 513th channel
      self.std_batch = std_batch()

      # Downsample to 512 1x1 features
      self.final_conv = nn.Conv2d(513, 512, kernel_size=4)

      # Evaluate image on realism
      self.uncond_score = nn.Linear(512,1)

      # Embed class vector into 128 dimension space
      self.class_embed = nn.Embedding(3, 128)

      self.feature_projector = nn.Linear(512, 128)

    def forward(self, x, labels):

      # Downsample to 4x4x512 features
      x = self.leaky(self.conv_in(x))
      x = self.block1(x)
      x = self.block2(x)
      x = self.block3(x)
      x = self.block4(x)
      x = self.block5(x)

      # Consider variation in batch
      x = self.std_batch(x) # [batch, 513, 4, 4]
      x = self.final_conv(x) # [batch, 512, 1, 1]

      # Flatten to [batch, 512]
      img_features = x.view(x.size(0), -1) 

      # Unconditioned score is a linear layer on the image features
      uc_score = self.uncond_score(img_features)

      c_embed = self.class_embed(labels)

      projected_features = self.feature_projector(img_features)

      # Conditioned score is dot product of features and class embedding
      c_score = torch.sum(projected_features * c_embed, dim=1, keepdim=True)

      # Return total score
      return uc_score + c_score 