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
from fastmri.pl_modules import FastMriDataModule, UnetModule


def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type("equispaced_fraction_3d", args.center_fractions, args.accelerations)
    # use random masks for train transform, fixed masks for val transform
    print("INFO: Grappa init")
    train_transform = VarNetDataTransformVolume4DGrappa(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransformVolume4DGrappa(mask_func=mask)
    test_transform = VarNetDataTransformVolume4DGrappa()

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
    model = UnetModule(
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
    print("Not resuming from checkpoint!")
    args.resume_from_checkpoint = None

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()

# 93:   No normalization, 4 offsets
# 94:   Zero-mean-unit-variance norm, 4 offsets
# 95:   Bigger model, 8 offsets, patch-wise
# 96:   Bigger model, 8 offsets
# 97:   Bigger model, 8 offsets, ssim loss

# 100:    KSpace init with Grappa, Unet as denoiser
# 101:    KSpace init with Grappa, Unet as denoiser, mse loss
# 102/3:  KSpace init with Grappa, Hamming Window as denoiser, ssim loss
# 104:    KSpace init with Grappa, Hamming Window as denoiser + hamming init, ssim loss