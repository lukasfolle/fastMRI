"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import torch
from fastmri.models import Unet
from fastmri.models.unet_4d import Unet4D
from torch.nn import functional as F
import fastmri
from fastmri.data import transforms
from fastmri.losses import combined_loss_offsets

from .mri_module import MriModule
from monai.transforms import RandSpatialCrop
from fastmri_examples.cs.hamming import HammingWindowNetwork


class UnetModule(MriModule):
    """
    Unet training module.

    This can be used to train baseline U-Nets from the paper:

    J. Zbontar et al. fastMRI: An Open Dataset and Benchmarks for Accelerated
    MRI. arXiv:1811.08839. 2018.
    """

    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        **kwargs,
    ):
        """
        Args:
            in_chans (int, optional): Number of channels in the input to the
                U-Net model. Defaults to 1.
            out_chans (int, optional): Number of channels in the output to the
                U-Net model. Defaults to 1.
            chans (int, optional): Number of output channels of the first
                convolution layer. Defaults to 32.
            num_pool_layers (int, optional): Number of down-sampling and
                up-sampling layers. Defaults to 4.
            drop_prob (float, optional): Dropout probability. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_step_size (int, optional): Learning rate step size. Defaults to
                40.
            lr_gamma (float, optional): Learning rate gamma decay. Defaults to
                0.1.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.rand_spatial_crop = RandSpatialCrop([8, 32, 32], random_size=False)

        # self.unet = Unet4D(
        #     in_chans=self.in_chans,
        #     out_chans=self.out_chans,
        #     chans=self.chans,
        #     num_pool_layers=3,
        #     drop_prob=self.drop_prob,
        # )
        self.hamming_window_network = HammingWindowNetwork((20, 128, 256))

    def forward(self, image):
        return self.hamming_window_network(image.unsqueeze(0).unsqueeze(1))
    
    def rand_augment(self, image, target):
        with torch.no_grad():
            stack = torch.stack((image, target), 0).squeeze()
            stack = stack.reshape((-1, 20, 128, 128))
            stack = self.rand_spatial_crop(stack)
            stack = stack.reshape((2, 8, 8, 32, 32))
            image = stack[0]
            target = stack[1]
            return image, target


    def reco(self, kspace):
        with torch.no_grad():
            images = []
            for offset in range(kspace.shape[1]):
                image = fastmri.ifft3c(kspace[:, offset])
                image = fastmri.complex_abs(image)
                image = fastmri.rss(image, dim=0).squeeze()
                image = transforms.complex_center_crop_3d(image, (image.shape[0], 128, 128))
                images.append(image)
            images = torch.stack(images, 0)
            return images

    def training_step(self, batch, batch_idx):
        # image = self.reco(batch[0].masked_kspace)
        target = batch[0].target
        # image, target = self.rand_augment(image, target)    
        # TODO: Should we do this?   
        # image = (image - image.mean()) / image.std()
        # image = image[:, 2:-2]
        # target = (target - target.mean()) / target.std()
        # target = target[:, 2:-2]
        output = self(batch[0].masked_kspace).unsqueeze(0).unsqueeze(0)
        # loss = F.l1_loss(output, target)
        
        loss = combined_loss_offsets(output.squeeze(0), target.unsqueeze(0))

        self.log("train_loss", loss.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        # image = self.reco(batch[0].masked_kspace)
        # image = (image - image.mean()) / image.std()
        # image = image[:, 2:-2]
        target = batch[0].target
        # target = (target - target.mean()) / target.std()
        # target = target[:, 2:-2]
        output = self(batch[0].masked_kspace).unsqueeze(0).unsqueeze(0)
        # loss = F.l1_loss(output, target)
        loss = combined_loss_offsets(output.squeeze(0), target.unsqueeze(0))
        self.log("validation_loss", loss)
        return {
            "batch_idx": batch_idx,
            "fname": "test",
            "slice_num": 0,
            "max_value": 0,
            "output": output.squeeze(0),
            "target": target.squeeze(0),
            "val_loss": loss,
            "masked_kspace": batch[0].masked_kspace,
            "hamming_window": self.hamming_window_network.hamming_window_layer.weight.detach().cpu(),
        }

    # def test_step(self, batch, batch_idx):
    #     output = self.forward(batch.image)
    #     mean = batch.mean.unsqueeze(1).unsqueeze(2)
    #     std = batch.std.unsqueeze(1).unsqueeze(2)

    #     return {
    #         "fname": batch.fname,
    #         "slice": batch.slice_num,
    #         "output": (output * std + mean).cpu().numpy(),
    #     }

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # network params
        parser.add_argument(
            "--in_chans", default=1, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=1, type=int, help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--chans", default=1, type=int, help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pool_layers",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--drop_prob", default=0.0, type=float, help="U-Net dropout probability"
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.001, type=float, help="RMSProp learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma", default=0.1, type=float, help="Amount to decrease step size"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
