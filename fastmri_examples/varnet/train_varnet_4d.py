"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import sys
import pathlib
from argparse import ArgumentParser

import pytorch_lightning as pl
from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransformVolume4D
from fastmri.pl_modules import FastMriDataModule, VarNetModule


def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    # mask = create_mask_for_mask_type("equispaced_fraction_3d", args.center_fractions, args.accelerations)
    # use random masks for train transform, fixed masks for val transform
    # train_transform = VarNetDataTransformVolume4D(mask_func=mask, use_seed=False)
    # val_transform = VarNetDataTransformVolume4D(mask_func=mask)
    # test_transform = VarNetDataTransformVolume4D()

    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
        volume_training=True,
        use_dataset_cache_file=False,
        cache_dir=args.cache_dir
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
        mask_center=False
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
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path(r"C:\Users\follels\Documents\fastMRI\fastmri_dirs.yaml")
    backend = None  # "ddp"  # "ddp"  # "ddp"
    num_gpus = 1 if backend == "ddp" else 1
    batch_size = num_gpus

    # set defaults based on optional directory config
    data_path = fetch_dir("knee_path", path_config)
    default_root_dir = fetch_dir("log_path", path_config) / "varnet" / "varnet_demo"

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[2],
        type=int,
        help="Acceleration rates to use for masks",
    )

    # data config
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,  # path to fastMRI data
        challenge="multicoil",  # only multicoil implemented for VarNet
        batch_size=batch_size,  # number of samples per batch
        test_path=None,  # path for test split, overwrites data_path
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
        num_cascades=5,  # number of unrolled iterations
        pools=3,  # number of pooling layers for U-Net
        chans=4,  # number of top-level channels for U-Net
        sens_pools=3,  # number of pooling layers for sense est. U-Net
        sens_chans=2,  # number of top-level channels for sense est. U-Net
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
        num_workers=4,
        # overfit_batches=1,
        log_every_n_steps=10,
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

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()
    # version 6/7: 4 2 4 3 2 -> 316k
    # version 8:   5 3 4 3 2 -> 1.4M & no mask center for sens est. unet
    # TODO: Different us pattern for each of the 8 offsets
    # TODO: Vergleichsmethode: cs eg espirit or enlive (/w and /wo grappa init)
