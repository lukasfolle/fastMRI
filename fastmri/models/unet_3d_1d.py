"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv3d, ConvTranspose3d, InstanceNorm3d
from fastmri.models.convnd import InstanceNorm4d


class Unet3D1D(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        self.downsample_conv_layers = nn.ModuleList([Conv3dMod(chans, chans, kernel_size=1, stride=2)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
            self.downsample_conv_layers.append(Conv3dMod(ch, ch, kernel_size=1, stride=2))

        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                Conv3dMod(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 6D tensor of shape `(N, in_chans, O, D, H, W)`.

        Returns:r
            Output tensor of shape `(N, out_chans, O, D, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer, downsample_conv_layer in zip(self.down_sample_layers, self.downsample_conv_layers):
            output = layer(output)
            stack.append(output)
            # avg_pool4d not available ..., use downsample instead. Potentially also use conv to sample down instead
            # output = F.avg_pool3d(output, kernel_size=2, stride=2, padding=0)
            output = downsample_conv_layer(output)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if output.shape[-3] != downsample_layer.shape[-3]:
                padding[5] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                # output = F.pad(output, padding, "reflect")
                output = F.pad(output, padding, "constant")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


class Conv3dMod(Conv3d):
    def offset_to_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, o, d, h, w = x.shape
        self.b = b
        return x.permute(0, 2, 1, 3, 4, 5).reshape(b * o, c, d, h, w)

    def batch_offset_to_third_dim(self, x: torch.Tensor) -> torch.Tensor:
        bo, c, d, h, w = x.shape
        o = bo // self.b
        return x.view(self.b, c, o, d, h, w).contiguous()
    
    def forward(self, x):
        x = self.offset_to_batch_dim(x)
        x = super().forward(x)
        x = self.batch_offset_to_third_dim(x)
        return x
    
    
class ConvTranspose3dMod(ConvTranspose3d):
    def offset_to_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, o, d, h, w = x.shape
        self.b = b
        return x.permute(0, 2, 1, 3, 4, 5).reshape(b * o, c, d, h, w)

    def batch_offset_to_third_dim(self, x: torch.Tensor) -> torch.Tensor:
        bo, c, d, h, w = x.shape
        o = bo // self.b
        return x.view(self.b, c, o, d, h, w).contiguous()
    
    def forward(self, x):
        x = self.offset_to_batch_dim(x)
        x = super().forward(x)
        x = self.batch_offset_to_third_dim(x)
        return x 
    

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float = 0, stride: int = 1):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        if drop_prob > 0:
            raise Warning("No Droupout implemented.")
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        
        self.layers = nn.Sequential(
            Conv3dMod(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            InstanceNorm4d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Dropout3d(drop_prob),
            Conv3dMod(out_chans, out_chans, kernel_size=3, padding=1, bias=False, stride=stride),
            InstanceNorm4d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Dropout3d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 5D tensor of shape `(N, in_chans, D, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, D, H, W)`.
        """
        image = self.layers(image)
        return image


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            ConvTranspose3dMod(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            InstanceNorm4d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 5D tensor of shape `(N, in_chans, D, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, D*2, H*2, W*2)`.
        """
        return self.layers(image)


if __name__ == "__main__":
    unet = Unet3D1D(1, 1, chans=4, num_pool_layers=2)
    ret = unet(torch.rand((15, 1, 8, 8, 8, 8)))
    print(ret.shape)
