"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import sys
import pathlib
from argparse import ArgumentParser

path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, path)

import pytorch_lightning as pl
from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransformVolume4D, VarNetDataTransformVolume4DGrappa
from fastmri.pl_modules import FastMriDataModule, VarNetModule


def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type("poisson_3d", args.center_fractions, args.accelerations)
    train_transform = VarNetDataTransformVolume4D(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransformVolume4D(mask_func=mask)
    test_transform = VarNetDataTransformVolume4D()

    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
        volume_training=True,
        use_dataset_cache_file=False,
        cache_dir=args.cache_dir,
        number_of_simultaneous_offsets=args.number_of_simultaneous_offsets
    )

    # ------------
    # model
    # ------------
    model = VarNetModule(
        num_cascades=args.num_cascades,
        pools=args.pools,
        chans=args.chans,
        sens_pools=args.sens_pools,
        sens_chans=args.sens_chans,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        volume_training=True,
        mask_center=False,
        accelerations=args.accelerations,
        loss=args.loss,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "valid":
        trainer.validate(model, dataloaders=data_module.val_dataloader())
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path(r"C:\Users\follels\Documents\fastMRI\fastmri_dirs.yaml")
    backend = None  # "ddp"  # "ddp"  # "ddp"
    num_gpus = 1 if backend == "ddp" else 1
    batch_size = 1

    # set defaults based on optional directory config
    data_path = fetch_dir("knee_path", path_config)
    default_root_dir = fetch_dir("log_path", path_config) / "varnet" / "varnet_demo"

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "valid", "test"),
        type=str,
        help="Operation mode",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.0],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[6],
        type=int,
        help="Acceleration rates to use for masks",
    )

    parser.add_argument(
        "--loss",
        default="combined_loss_offsets",
        type=str,
        help="Loss function to use",
    )
    
    parser.add_argument(
        "--number_of_simultaneous_offsets",
        default=8,
        type=int,
        help="Number of simultaneous offsets",
    )

    # data config
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,  # path to fastMRI data
        challenge="multicoil",  # only multicoil implemented for VarNet
        batch_size=batch_size,  # number of samples per batch
        test_path=None,  # path for test split, overwrites data_path
        number_of_simultaneous_offsets=8,
    )

    parser.add_argument(
        "--cache_dir",
        default=fetch_dir("cache_path", path_config),
        type=str,
        help="Folder to save cache to",
    )

    # module config
    parser = VarNetModule.add_model_specific_args(parser)
    parser.set_defaults(
        num_cascades=6,  # number of unrolled iterations
        pools=3,  # number of pooling layers for U-Net
        chans=8,  # number of top-level channels for U-Net
        sens_pools=3,  # number of pooling layers for sense est. U-Net
        sens_chans=8,  # number of top-level channels for sense est. U-Net
        lr=0.001,  # Adam learning rate
        lr_step_size=100000,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight regularization strength
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        accelerator=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=False,  # makes things slower, but deterministic
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=1000,  # max number of epochs
        num_workers=0,
        log_every_n_steps=1,
        precision=16,
        # overfit_batches=1,
        # check_val_every_n_epoch=5,
    )

    args = parser.parse_args()

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        ),
    ]

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])
    # args.resume_from_checkpoint = r"C:\Users\follels\Documents\fastMRI\logs\varnet\varnet_demo\checkpoints\fastmri_checkpoint_35732.ckpt"
    # print("Not resuming from checkpoint!")
    # args.resume_from_checkpoint = None

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()

# TODO: Try out pretrained network as init
# Run 30: 3 training cases
# Run 32: 1 training case
# Run 33: Overfit one sample
# Test less downsampling influence
# Run 55: +tv, denser outer sampling
# Run 56: dc_weight 1 -> 0.5
# Run 57: 6 3 8 3 8 (7.2M) -> 4 3 8 3 8 (5.2M)
# Run 58: US 4 6 -> 3 5
# Run 59: US 4 6 -> 5 7
# Run 60: US 4 6 -> 2 4
# Run 61/62: US 4 6 -> 1 1
# Run 63: US 4 6 -> 2 4, mse loss
# Run 64: US 4 6 -> 3 5, mse loss
# Run 71: US 3 5, mse loss, all offsets (2x8) --> Prediction an Kathi schicken
# Run 72: US 3 5, mse + ssim loss, all offsets (2x8) (better than mse alone)
# Run 73: US 3 5, mse(l2) + ssim loss, all offsets (2x8)
# Run 74: US 3 5, ssim loss, all offsets (2x8) (best so far)
# Run 75: US 3 5, ssim loss, all offsets (2x8), 4 3 8 3 8 (5.2M)
# Run 76: US 3 5, ssim loss, all offsets (2x8), 2 3 8 3 8 (3.1M)
# Run 77: US 3 5, ssim loss, all offsets (2x8), 8 3 8 3 8 (9.3M)
# Run 78: US 3 5, ssim loss, all offsets (2x8), dense center sampling (better than 74), sampling closer to center more important
# Changed to offset-wise training, US 2 6 (7.2M & 45GB -> 2.5M & 20GB)
# Run 79: US 2 6, ssim loss, single offsets
# Run 80: US 2 6, mse loss, single offset
# Back to (almost) all offset training
# Run 81: US 2 6, ssim loss (2x8 offsets) (best so far)

# Run 82: US 2 6 , Model 6 2 8 2 8 (1.8M), 4 offsets
# Run 83: US 2 6 , Model 6 2 8 2 8, 8 offsets
# Run 84: US 2 6 , Model 6 3 8 3 8, 8 offsets (best so far)
# Run 85: US 2 6 , Model 6 3 8 3 8, 8 offsets (pretrained fastMRI)
# Run 86: US 2 6 , Model 6 3 8 3 8, 8 offsets, ssim loss, no kspace imputation (considerably worse than 84)

# Optimizer channels for 4 offsets
# Run 87: US 26, Model 6 1 8 2 8 (274K), 4 offsets
# Run 88: US 26, Model 6 2 8 2 8, 4 offsets 
# Run 89: US 26, Model 6 4 8 2 8, 4 offsets 
# Run 90: US 26, Model 6 8 8 2 8, 4 offsets 
# Run 91: US 26, Model 6 12 8 2 8, 4 offsets 
# Run 92: US 26, Model 6 16 8 2 8 (6.2M), 4 offsets

# Run 98: replicate Run 84
# Run 99: GRAPPA Init

# Run 105: Grappa init and hamming window layer last
# Run 106: GRAPPA init with alternating masks over offsets
# Run 107: GRAPPA init alternating, R=9 (3x3), removes some artifacts
# Run 108: GRAPPA init alternating, R=9 (3x3), Hamming parametrized as last layer
# Run 109: GRAPPA init alternating, R=9 (3x3), No VarNet, only Hamming parametrized as last layer
# Run 110: Pure VarNet, Poisson undersampling factor 8.6
# Run 111 Try Grappa init for poisson undersampling -> Error: Singular matrix

# Run 112: GRAPPA init alternating, R=16 (4x4)
# Run 113: No Grappa, imputed kspace as input to sens network, masked kspace input to cascades
# Grappa baseline not comparable so far due to using center kspace for all offsets individually, now switched to first offset.
# -> But doesnt change much ...
# Run 114: Grappa Init, 3x3, only first offset as acs 
# Previous grappa results were wrong since center of kspace was kept instead of acs region.

# Run 115 masked_kspace input, Corrected Grappa baseline, acs input to sensmodel, 3x3 
# Run 116 masked_kspace input, Corrected Grappa baseline, acs input to sensmodel, 3x3, ssim + mse
# Run 117 filled_kspace input, acs to SenseNet, ssim + mse 
# Run 118 filled_kspace input, acs to SenseNet, ssim (**Better than GRAPPA, NRMSE - 1, SSIM + 1.4, PSNR + 0.8**)
# Run 119 filled_kspace input, acs to SenseNet, ssim, reduces lr to 0.0001

# Run 120 GRAPPA Init, Esprit sens instead of SensNet, 0.0001 lr (6.2M)
# Run 121 No Init, Esprit sens instead of SensNet
# Run 122 Kspace imputation, Esprit sens instead of SensNet

# Run 123 GRAPPA Init, ACS to SenseNet, Denser center sampling (x8)
# Run 124 Like 123, w/ hamming window parametrized (HWP)
# Run 126 Like 123, w/ hamming window parametrized (prev cascade) (**Best so far** SSIM +3%, PSNR +0.1%, NRMSE +1%)
# Run 127 Like 123, w/ hamming window parametrized (prev cascade), mse
# Run 128 Continue 126

# Run 129: No HWP, added U-Net after to_image_space_transform
# Run 130: No HWP, added U-Net after to_image_space_transform, mse
# Run 131: VarNet3D1D, no image_space_net, 3.1M

# Run 132: VarNet3D1D, poisson sampling
# Run 133: Continue 132 /w mae loss

# TODO: Switch to poisson sampling /wo grappa init, save 100 us patterns -> load during training

