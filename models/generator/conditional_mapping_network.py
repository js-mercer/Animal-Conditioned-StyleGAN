import torch
import torch.nn as nn

from models.generator.pixel_norm import PixelNorm

class ConditionalMappingNetwork(nn.Module):
    def __init__(self, latent_dim=512, num_classes=3):
      super().__init__()

      self.pixel_norm = PixelNorm()
      # Embed classes into higher dimensional space
      self.label_embed = nn.Embedding(num_classes, 128)
      # Direct linear z projection to help prevent mode collapse
      self.z_projection = nn.Linear(latent_dim, latent_dim)

      layers = []
      in_features = latent_dim + 128

      # 4 layer MLP to disentangle latent space and embedded class vector
      for _ in range(4):
        layers.append(nn.Linear(in_features, latent_dim))
        layers.append(nn.LeakyReLU(0.2))
        in_features = latent_dim

      self.mapping = nn.Sequential(*layers)

    def forward(self, z, labels):
      z_norm = self.pixel_norm(z)

      # Convert integer labels or validate externally supplied class embeddings
      if labels.dtype == torch.long:
        c_embed = self.label_embed(labels)
      else:
        expected_shape = (z.shape[0], self.label_embed.embedding_dim)
        if labels.ndim != 2 or labels.shape != expected_shape:
          raise ValueError(
              "Embedding labels must have shape "
              f"[{expected_shape[0]}, {expected_shape[1]}]."
          )
        c_embed = labels.to(device=z.device, dtype=z.dtype)

      # Preserve a direct linear relationship with Z
      projected_z = self.z_projection(z_norm)

      # Allow MLP to learn nonlinear transformations
      combined = torch.cat([projected_z, c_embed], dim=1)
      w = self.mapping(combined)

      # Return the sum of projected z and mapped w to preserve information from both 
      return w + projected_z