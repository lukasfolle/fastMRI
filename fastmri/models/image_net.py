import torch
from torch import nn

from fastmri.models.unet_3d_1d import Conv3dMod


class ImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        num_channels = 8
        self.model = nn.ModuleList([
            nn.Sequential(
                Conv3dMod(1, num_channels, 3, 1, 1),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                Conv3dMod(num_channels, num_channels, 3, 1, 1),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                Conv3dMod(num_channels, num_channels, 3, 1, 1),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                Conv3dMod(num_channels, num_channels, 3, 1, 1),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                Conv3dMod(num_channels, num_channels, 3, 1, 1),
                nn.LeakyReLU(),
            )
        ])
        self.final_conv = Conv3dMod(num_channels, 1, 3, 1, 1)
        for i, layer in enumerate(self.model):
            if isinstance(layer, Conv3dMod):
                unit_init = torch.zeros(self.model[i].weight.shape)
                unit_init[..., 1, 1, 1] = 1.0
                unit_init = torch.nn.Parameter(unit_init)
                self.model[i].weight = unit_init
            else:
                for j, l in enumerate(layer):
                    if isinstance(l, Conv3dMod):
                        unit_init = torch.zeros(self.model[i][j].weight.shape)
                        unit_init[..., 1, 1, 1] = 1.0
                        unit_init = torch.nn.Parameter(unit_init)
                        self.model[i][j].weight = unit_init
    
    def forward(self, x):
        for layer in self.model:
            x = layer(x) + x
        x = self.final_conv(x)
        return x


if __name__ == "__main__":
    im = ImageNet()
