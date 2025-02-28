"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import fastmri
import torch
import torch.nn.functional as F
from fastmri.data import transforms
from fastmri.models import VarNet, VarNet3D, VarNet4D, VarNet3D1D, Unet3D1D, Unet4D
from fastmri.losses import combined_loss, ssim3D_loss
import nibabel as nib
import numpy as np
import os
from copy import deepcopy
from monai.transforms import RandSpatialCropSamples

from .mri_module import MriModule
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse


class VarNetModule(MriModule):
    """
    VarNet training module.

    This can be used to train variational networks from the paper:

    A. Sriram et al. End-to-end variational networks for accelerated MRI
    reconstruction. In International Conference on Medical Image Computing and
    Computer-Assisted Intervention, 2020.

    which was inspired by the earlier paper:

    K. Hammernik et al. Learning a variational network for reconstruction of
    accelerated MRI data. Magnetic Resonance inMedicine, 79(6):3055–3071, 2018.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 18,
        sens_pools: int = 4,
        sens_chans: int = 8,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        volume_training=False,
        mask_center=True,
        accelerations=[],
        loss="combined_loss_offsets",
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            chans: Number of channels for cascade U-Net.
            sens_pools: Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            sens_chans: Number of channels for sensitivity map U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            num_sense_lines: Number of low-frequency lines to use for sensitivity map
                computation, must be even or `None`. Default `None` will automatically
                compute the number from masks. Default behaviour may cause some slices to
                use more low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults. To prevent this, either set
                `num_sense_lines`, or set `skip_low_freqs` and `skip_around_low_freqs`
                to `True` in the EquispacedMaskFunc. Note that setting this value may
                lead to undesired behaviour when training on multiple accelerations
                simultaneously.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.sens_pools = sens_pools
        self.sens_chans = sens_chans
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.mask_center = mask_center
                
        if volume_training:
            print("Using VarNet3D1D")
            self.varnet = Unet3D1D(1, 1, num_pool_layers=2)
            # self.varnet = VarNet3D1D(
            #     num_cascades=self.num_cascades,
            #     sens_chans=self.sens_chans,
            #     sens_pools=self.sens_pools,
            #     chans=self.chans,
            #     pools=self.pools,
            #     mask_center=self.mask_center
            # )
        else:
            self.varnet = VarNet(
                num_cascades=self.num_cascades,
                sens_chans=self.sens_chans,
                sens_pools=self.sens_pools,
                chans=self.chans,
                pools=self.pools,
            )

        if loss == "combined":
            self.loss = combined_loss
        elif loss == "combined_loss_offsets":
            self.loss = combined_loss_offsets
        elif loss == "l1":
            self.loss = torch.nn.L1Loss()
        elif loss == "ssim":
            self.loss = ssim3D_loss
        else:
            self.loss = fastmri.SSIMLoss()

    def forward(self, masked_kspace, acs, mask, num_low_frequencies):
        return self.varnet(masked_kspace, acs, mask, num_low_frequencies)

    def training_step(self, batch, batch_idx):
        batch = batch[0]
        kspace, acs, mask = batch.filled_kspace[None], batch.acs[None], batch.mask[None]
        acs = torch.stack([acs for _ in range(8)], 2)
        acs = torch.stack((torch.real(acs), torch.imag(acs)), -1)
        acs = F.pad(acs, (0, 0, 0, 0, 52, 52))
        # output = self(kspace, acs, mask, batch.num_low_frequencies)        
        with torch.no_grad():
            volume = self.reco(kspace)
            volume = volume.unsqueeze(0).unsqueeze(0)
        output = self.varnet(volume)
        output = output.squeeze()
        
        target, output = transforms.center_crop_to_smallest(batch.target, output)
        output = output.unsqueeze(0)
        loss = self.loss(
            output, target.unsqueeze(0),
        )
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch[0]
        kspace, acs, mask = batch.filled_kspace[None], batch.acs[None], batch.mask[None]
        acs = torch.stack([acs for _ in range(8)], 2)
        acs = torch.stack((torch.real(acs), torch.imag(acs)), -1)
        acs = F.pad(acs, (0, 0, 0, 0, 52, 52))
        # output = self(kspace, acs, mask, batch.num_low_frequencies)        
        with torch.no_grad():
            volume = self.reco(kspace)
            volume = volume.unsqueeze(0).unsqueeze(0)
        output = self.varnet(volume)
        output = output.squeeze()
        
        
        target, output = transforms.center_crop_to_smallest(batch.target, output)
        output = output.unsqueeze(0)
        loss = self.loss(
            output, target.unsqueeze(0),
        )
        self.log("validation_loss", loss)
        if self.current_epoch % 10 == 0:
            self.log_zero_filling_metrics(batch.filled_kspace[None], batch.target[None])
            self.save_predictions_to_nifti(output.float(), target, batch.filled_kspace, batch_idx)
            
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "mask": batch.mask,
            "target": target,
            "masked_kspace": batch.masked_kspace[None],
            "val_loss": loss,
        }
    
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
        
    def test_step(self, batch, batch_idx):
        output = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies)

        # check for FLAIR 203
        if output.shape[-1] < batch.crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])
        else:
            crop_size = batch.crop_size

        output = transforms.center_crop(output, crop_size)

        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": output.cpu().numpy(),
        }

    def validation_epoch_end(self, val_logs):
        super().validation_epoch_end(val_logs)

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
        nib.save(reco_nii, os.path.join(r"E:\Lukas\cest_data\Probanden\Mareike\prediction", f"val_prediction_{batch_idx}_epoch_{self.current_epoch}.nii.gz"))
        reco_nii = nib.Nifti1Image(np.flip(target.cpu().numpy().squeeze().transpose((1, 2, 3, 0)), 2), affine=affine_matrix)
        nib.save(reco_nii, os.path.join(r"E:\Lukas\cest_data\Probanden\Mareike\prediction", f"val_target_{batch_idx}_epoch_{self.current_epoch}.nii.gz"))
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
        nib.save(reco_nii, os.path.join(r"E:\Lukas\cest_data\Probanden\Mareike\prediction", f"val_default_reco_{batch_idx}_epoch_{self.current_epoch}.nii.gz"))

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optim, self.lr_step_size, self.lr_gamma
        # )

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)

        return {"optimizer": optim}  # , "lr_scheduler": scheduler, "monitor": "train_loss"}

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of VarNet cascades",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in VarNet blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in VarNet blocks",
        )
        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help="Number of pooling layers for sense map estimation U-Net in VarNet",
        )
        parser.add_argument(
            "--sens_chans",
            default=8,
            type=float,
            help="Number of channels for sense map estimation U-Net in VarNet",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
