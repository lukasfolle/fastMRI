"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from fastmri.data.mri_data import CombinedSliceDataset, SliceDataset
from torch.utils.data import Sampler


class VolumeSampler(Sampler):
    """
    Sampler for volumetric MRI data.

    Based on pytorch DistributedSampler, the difference is that all instances
    from the same MRI volume need to go to the same node for distributed
    training. Dataset example is a list of tuples (fname, instance), where
    fname is essentially the volume name (actually a filename).
    """

    def __init__(
        self,
        dataset: Union[CombinedSliceDataset, SliceDataset],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        """
        Args:
            dataset: An MRI dataset (e.g., SliceData).
            num_replicas: Number of processes participating in distributed
                training. By default, :attr:`rank` is retrieved from the
                current distributed group.
            rank: Rank of the current process within :attr:`num_replicas`. By
                default, :attr:`rank` is retrieved from the current distributed
                group.
            shuffle: If ``True`` (default), sampler will shuffle the indices.
            seed: random seed used to shuffle the sampler if
                :attr:`shuffle=True`. This number should be identical across
                all processes in the distributed group.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed

        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        # get all file names and split them based on number of processes
        self.all_volume_names = np.array(
            sorted([example[0] for example in self.dataset.examples])
        )
        self.all_volumes_split = np.array_split(
            self.all_volume_names, self.num_replicas
        )

        # get slice indices for each file name
        indices: List[List[int]] = [[] for _ in range(self.num_replicas)]

        for i, example in enumerate(self.dataset.examples):
            vname = example[0]
            for rank in range(self.num_replicas):
                if vname in self.all_volumes_split[rank]:
                    indices[rank].append(i)

        # need to send equal number of samples to each process - take the max
        self.num_samples = max([len(l) for l in indices])
        self.total_size = self.num_samples * self.num_replicas
        self.indices = indices[self.rank]

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            ordering = torch.randperm(len(self.indices), generator=g).tolist()
            indices = list(np.array(self.indices)[ordering])
        else:
            indices = self.indices

        # add extra samples to match num_samples
        indices = indices + indices[: self.num_samples - len(indices)]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
