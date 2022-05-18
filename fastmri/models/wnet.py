import torch
from torch import nn

from fastmri.models.unet_3d_1d import Unet3D1D
from fastmri.models.unet_1d import Unet1D


class WNet(nn.Module):
    def __init__(self, in_chans=2, out_chans=2, chans=8, num_pool_layers=3, drop_prob=0):
        super().__init__()
        self.unet3d = Unet3D1D(in_chans, out_chans, chans, num_pool_layers, drop_prob)
        self.unet1d = Unet1D(in_chans, out_chans, chans, num_pool_layers, drop_prob)
    
    def forward(self, x):
        x = self.unet3d(x)
        x = self.unet1d(x)
        return x


if __name__ == "__main__":
    wnet = WNet()
    print(f"Number of parameters: ({sum(p.numel() for p in wnet.parameters() if p.requires_grad)})")
    ret = wnet(torch.rand((1, 2, 8, 8, 8, 8)))
    print(ret.shape)