"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser
import numpy as np
import os
import nibabel as nib

import torch
from fastmri.models import Unet
from fastmri.models.unet_4d import Unet4D
from fastmri.models.unet_3d_1d import Unet3D1D
from fastmri.models.varnet_3_1d import NormUnet
from fastmri.models.unet_1d import Unet1D
from torch.nn import functional as F
import fastmri
from fastmri.data import transforms
from fastmri.losses import CombinedLoss

from .mri_module import MriModule
from monai.transforms import RandSpatialCrop, RandSpatialCropSamplesd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse



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
        num_pool_layers=2,
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
        # self.loss = combined_loss_offsets
        self.loss = CombinedLoss()
        self.num_augmentation_samples = 2
        self.train_augmentation = None  # RandSpatialCropSamplesd(keys=["volume", "target"], roi_size=(8, 8, 92, 92), num_samples=self.num_augmentation_samples, random_size=False)
        # self.model = Unet3D1D(1, 1, chans=chans, num_pool_layers=num_pool_layers)
        self.model = Unet1D(1, 1, chans=chans, num_pool_layers=num_pool_layers)

    def forward(self, image):
        return self.model(image)

    def reco(self, kspace):
        kspace = kspace.squeeze()
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
        batch = batch[0]
        kspace, acs, mask = batch.filled_kspace[None], batch.acs[None], batch.mask[None]

        with torch.no_grad():
            volume = self.reco(kspace)
            volume = volume.unsqueeze(0).unsqueeze(0)
            target = batch.target.unsqueeze(0)
            if self.train_augmentation is not None:
                ret = self.train_augmentation({"volume": volume.squeeze(0), "target": target})
                volume = torch.stack([ret[i]["volume"] for i in range(self.num_augmentation_samples)], 0)
                target = torch.stack([ret[i]["target"] for i in range(self.num_augmentation_samples)], 0)
                if isinstance(volume, list):
                    volume = torch.stack(volume)
            else:
                target = target.unsqueeze(0)
        output = self(volume)
        
        output = (output - output.min()) / (output.max() - output.min())
        output = (volume.max() - volume.min()) * output - volume.min()

        loss = self.loss(output, target)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch[0]
        kspace, acs, mask = batch.filled_kspace[None], batch.acs[None], batch.mask[None]

        with torch.no_grad():
            volume = self.reco(kspace)
            volume = volume.unsqueeze(0).unsqueeze(0)
        output = self(volume)

        target = batch.target.unsqueeze(0).unsqueeze(1)
        
        output = (output - output.min()) / (output.max() - output.min())
        output = (volume.max() - volume.min()) * output - volume.min()
        
        loss = self.loss(output, target)

        self.log("validation_loss", loss)
        if self.current_epoch % 10 == 0:
            self.log_zero_filling_metrics(batch.filled_kspace[None], batch.target[None])
            self.save_predictions_to_nifti(output.float(), target, batch.filled_kspace, batch_idx)
            
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output.squeeze(0),
            "mask": batch.mask,
            "target": target.squeeze(0),
            "masked_kspace": batch.masked_kspace[None],
            "val_loss": loss,
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
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optim, mode="min", factor=0.5, patience=50, verbose=True
        # )

        return {"optimizer": optim}  # , "lr_scheduler": scheduler, "monitor": "validation_loss"}

    def log_zero_filling_metrics(self, kspace, target):
        metrics = {"mse": 0, "ssim": 0, "psnr": 0}
        for offset in range(kspace.shape[2]):
            t = target[:, offset].squeeze()
            k_space_downsampled = kspace[:, :, offset].squeeze()
            k_space_downsampled = torch.view_as_real(k_space_downsampled[..., 0] + 1j * k_space_downsampled[..., 1])
            volume = fastmri.ifft3c(k_space_downsampled)
            volume = fastmri.complex_abs(volume)
            volume = fastmri.rss(volume, dim=0)
            t, volume = transforms.center_crop_to_smallest(t, volume)
            t = (t - t.min()) / (t.max() - t.min())
            volume = (volume - volume.min()) / (volume.max() - volume.min())
            t = t.cpu().numpy()
            volume = volume.cpu().numpy()
            metrics["mse"] = metrics["mse"] + normalized_root_mse(t, volume)
            metrics["psnr"] = metrics["psnr"] + peak_signal_noise_ratio(t, volume, data_range=(t.max() - t.min()))
            metrics["ssim"] = metrics["ssim"] + structural_similarity(t, volume, win_size=3)
        metrics["mse"] = metrics["mse"] / kspace.shape[2]
        metrics["psnr"] = metrics["psnr"] / kspace.shape[2]
        metrics["ssim"] = metrics["ssim"] / kspace.shape[2]
        self.log("val_metrics/nrmse_zfil", metrics["mse"])
        self.log("val_metrics/psnr_zfil", metrics["psnr"])
        self.log("val_metrics/ssim_zfil", metrics["ssim"])

    def save_predictions_to_nifti(self, output, target, masked_kspace, batch_idx):
        affine_matrix = np.array([[4, 0, 0, 0], [0, 1.203, 0, 0], [0, 0, 1.203, 0], [0, 0, 0, 1]])
        reco_nii = nib.Nifti1Image(np.flip(output.cpu().numpy().squeeze().transpose((1, 2, 3, 0)), 2), affine=affine_matrix)
        nib.save(reco_nii, os.path.join(r"E:\Lukas\cest_data\Probanden\Mareike\prediction", f"version_{self.logger.version}_val_prediction_{batch_idx}_epoch_{self.current_epoch}.nii.gz"))
        reco_nii = nib.Nifti1Image(np.flip(target.cpu().numpy().squeeze().transpose((1, 2, 3, 0)), 2), affine=affine_matrix)
        nib.save(reco_nii, os.path.join(r"E:\Lukas\cest_data\Probanden\Mareike\prediction", f"val_target_{batch_idx}.nii.gz"))
        default_reco = []
        for offset in range(masked_kspace.shape[1]):
            k_space_downsampled = masked_kspace[:, offset].squeeze()
            k_space_downsampled = torch.view_as_real(k_space_downsampled[..., 0] + 1j * k_space_downsampled[..., 1])
            volume = fastmri.ifft3c(k_space_downsampled)
            volume = fastmri.complex_abs(volume)
            volume = fastmri.rss(volume, dim=0)
            default_reco.append(volume)
        target, default_reco = transforms.center_crop_to_smallest(target, torch.stack(default_reco, 0))
        reco_nii = nib.Nifti1Image(np.flip(default_reco.cpu().numpy().squeeze().transpose((1, 2, 3, 0)), 2), affine=affine_matrix)
        nib.save(reco_nii, os.path.join(r"E:\Lukas\cest_data\Probanden\Mareike\prediction", f"val_default_reco_{batch_idx}.nii.gz"))

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
