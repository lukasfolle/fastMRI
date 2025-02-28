"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
from argparse import ArgumentParser
from collections import defaultdict

import fastmri
import numpy as np
import pytorch_lightning as pl
import torch
from fastmri import evaluate
from fastmri.data import transforms
from torchmetrics.metric import Metric
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse


class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity


class MriModule(pl.LightningModule):
    """
    Abstract super class for deep larning reconstruction models.

    This is a subclass of the LightningModule class from pytorch_lightning,
    with some additional functionality specific to fastMRI:
        - Evaluating reconstructions
        - Visualization

    To implement a new reconstruction model, inherit from this class and
    implement the following methods:
        - training_step, validation_step, test_step:
            Define what happens in one step of training, validation, and
            testing, respectively
        - configure_optimizers:
            Create and return the optimizers

    Other methods from LightningModule can be overridden as needed.
    """

    def __init__(self, num_log_images: int = 100):
        """
        Args:
            num_log_images: Number of images to log. Defaults to 16.
        """
        super().__init__()

        self.num_log_images = num_log_images
        self.val_log_indices = None

        # self.NMSE = DistributedMetricSum()
        # self.SSIM = DistributedMetricSum()
        # self.PSNR = DistributedMetricSum()
        # self.ValLoss = DistributedMetricSum()
        # self.TotExamples = DistributedMetricSum()
        # self.TotSliceExamples = DistributedMetricSum()

    def validation_step_end(self, val_logs):
        # check inputs
        for k in (
            "batch_idx",
            "fname",
            "slice_num",
            "max_value",
            "output",
            "target",
            "masked_kspace",
            "val_loss",
        ):
            if k not in val_logs.keys():
                raise RuntimeError(
                    f"Expected key {k} in dict returned by validation_step."
                )

        def reco(kspace, target):
            offset = 0
            k_space_downsampled = kspace[:, offset].squeeze()
            k_space_downsampled = torch.view_as_real(k_space_downsampled[..., 0] + 1j * k_space_downsampled[..., 1])
            volume = fastmri.ifft3c(k_space_downsampled)
            volume = fastmri.complex_abs(volume)
            volume = fastmri.rss(volume, dim=0)
            _, volume = transforms.center_crop_to_smallest(target[offset], volume)
            return volume.cpu().numpy()


        target = val_logs["target"].cpu().numpy()
        target = ((target - target.min()) / (target.max() - target.min() + 1e-6)).squeeze()
        prediction = val_logs["output"].cpu().numpy()
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-6)
        prediction = prediction.squeeze()
        self.log("val_metrics/psnr", peak_signal_noise_ratio(target, prediction))
        self.log("val_metrics/nrmse", normalized_root_mse(target, prediction))
        self.log("val_metrics/ssim", structural_similarity(target, prediction, win_size=3))

        if val_logs["output"].ndim == 2:
            val_logs["output"] = val_logs["output"].unsqueeze(0)
        if val_logs["target"].ndim == 2:
            val_logs["target"] = val_logs["target"].unsqueeze(0)

        # pick a set of images to log if we don't have one already
        if self.val_log_indices is None:
            self.val_log_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # log images to tensorboard
        if isinstance(val_logs["batch_idx"], int):
            batch_indices = [val_logs["batch_idx"]]
        else:
            batch_indices = val_logs["batch_idx"]
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                key = f"val_images_idx_{batch_idx}"
                target = val_logs["target"][i].unsqueeze(0)
                output = val_logs["output"][i].unsqueeze(0)
                # hamming_window = val_logs["hamming_window"][10].unsqueeze(0)
                # hamming_window = (hamming_window - hamming_window.min()) / (hamming_window.max() - hamming_window.min())
                error = torch.abs(target - output)
                output = (output - output.min()) / (output.max() - output.min() + 1e-10)
                target = (target - target.min()) / (target.max() - target.min() + 1e-10)
                error = (error - error.min()) / (error.max() - error.min() + 1e-10)
                default_reco = reco(val_logs["masked_kspace"].unsqueeze(0)[i], val_logs["target"].squeeze()[i])
                default_reco = (default_reco - default_reco.min()) / (default_reco.max() - default_reco.min() + 1e-10)
                self.log_image(f"{key}/target", target[:, 0, target.shape[2] // 2, ...])
                self.log_image(f"{key}/reconstruction", output[:, 0, output.shape[2] // 2, ...])
                self.log_image(f"{key}/error", error[:, 0, error.shape[2] // 2, ...])
                # self.log_image(f"{key}/default_reco", default_reco[default_reco.shape[0] // 2][None])
                # self.log_image(f"{key}/hamming_window_central", hamming_window)
                if "mask" in val_logs.keys():
                    mask = val_logs["mask"][i]
                    mask = mask / mask.max()
                    mask = mask.squeeze()[..., 0]
                    if len(mask.shape) > 2:
                        mask = mask[None, 0]
                    else:
                        mask = mask[None]
                    self.log_image(f"{key}/mask", mask)  # TODO: CHange back to just mask

            return {
                "val_loss": val_logs["val_loss"],
            }

    def log_image(self, name, image):
        self.logger.experiment.add_image(name, image, global_step=self.global_step)

    def validation_epoch_end(self, val_logs):
        pass
        # aggregate losses
        # losses = []
        # mse_vals = defaultdict(dict)
        # target_norms = defaultdict(dict)
        # ssim_vals = defaultdict(dict)
        # max_vals = dict()
        #
        # # use dict updates to handle duplicate slices
        # for val_log in val_logs:
        #     losses.append(val_log["val_loss"].view(-1))
        #
        #     for k in val_log["mse_vals"].keys():
        #         mse_vals[k].update(val_log["mse_vals"][k])
        #     for k in val_log["target_norms"].keys():
        #         target_norms[k].update(val_log["target_norms"][k])
        #     # for k in val_log["ssim_vals"].keys():
        #     #     ssim_vals[k].update(val_log["ssim_vals"][k])
        #     for k in val_log["max_vals"]:
        #         max_vals[k] = val_log["max_vals"][k]
        #
        # # check to make sure we have all files in all metrics
        # assert (
        #     mse_vals.keys()
        #     == target_norms.keys()
        #     # == ssim_vals.keys()
        #     == max_vals.keys()
        # )
        #
        # # apply means across image volumes
        # metrics = {"nmse": 0,
        #         #    "ssim": 0,
        #            "psnr": 0}
        # local_examples = 0
        # for fname in mse_vals.keys():
        #     local_examples = local_examples + 1
        #     mse_val = torch.mean(
        #         torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])
        #     )
        #     target_norm = torch.mean(
        #         torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
        #     )
        #     metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
        #     metrics["psnr"] = (
        #         metrics["psnr"]
        #         + 20
        #         * torch.log10(
        #             torch.tensor(
        #                 max_vals[fname], dtype=mse_val.dtype, device=mse_val.device
        #             )
        #         )
        #         - 10 * torch.log10(mse_val)
        #     )
        #     # metrics["ssim"] = metrics["ssim"] + torch.mean(
        #     #     torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
        #     # )
        #
        # # reduce across ddp via sum
        # metrics["nmse"] = self.NMSE(metrics["nmse"])
        # # metrics["ssim"] = self.SSIM(metrics["ssim"])
        # metrics["psnr"] = self.PSNR(metrics["psnr"])
        # tot_examples = self.TotExamples(torch.tensor(local_examples))
        # val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        # tot_slice_examples = self.TotSliceExamples(
        #     torch.tensor(len(losses), dtype=torch.float)
        # )
        #
        # self.log("validation_loss", val_loss / tot_slice_examples, prog_bar=True)
        # for metric, value in metrics.items():
        #     self.log(f"val_metrics/{metric}", value / tot_examples)

    def test_epoch_end(self, test_logs):
        outputs = defaultdict(dict)

        # use dicts for aggregation to handle duplicate slices in ddp mode
        for log in test_logs:
            for i, (fname, slice_num) in enumerate(zip(log["fname"], log["slice"])):
                outputs[fname][int(slice_num.cpu())] = log["output"][i]

        # stack all the slices for each file
        for fname in outputs:
            outputs[fname] = np.stack(
                [out for _, out in sorted(outputs[fname].items())]
            )

        # pull the default_root_dir if we have a trainer, otherwise save to cwd
        if hasattr(self, "trainer"):
            save_path = pathlib.Path(self.trainer.default_root_dir) / "reconstructions"
        else:
            save_path = pathlib.Path.cwd() / "reconstructions"
        self.print(f"Saving reconstructions to {save_path}")

        fastmri.save_reconstructions(outputs, save_path)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # logging params
        parser.add_argument(
            "--num_log_images",
            default=16,
            type=int,
            help="Number of images to log to Tensorboard",
        )

        return parser
