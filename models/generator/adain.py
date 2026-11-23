import torch.nn as nn

class AdaIN(nn.Module):
    def __init__(self, channels, latent_dim=512):
      super().__init__()
      # Normalise each feature map of each channel individually, retains structure and clears style
      self.norm = nn.InstanceNorm2d(channels, affine=False, eps=1e-8)
      # Linear layer generates scale y_s and shift y_b of each injected feature
      self.style_layer = nn.Linear(latent_dim, channels * 2)

      # Initialise weights
      nn.init.ones_(self.style_layer.bias[:channels])
      nn.init.zeros_(self.style_layer.bias[channels:])

    def forward(self, x, w):
      # Normalise and compute style params
      x_norm = self.norm(x) 
      style = self.style_layer(w)
      y_s, y_b = style.chunk(2, dim=1)

      y_s = y_s.unsqueeze(-1).unsqueeze(-1)
      y_b = y_b.unsqueeze(-1).unsqueeze(-1)

      # Return correct strength of local features
      return y_s * x_norm + y_b