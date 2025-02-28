"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import pickle
import random
import shutil
import xml.etree.ElementTree as etree
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn
import pathlib
import shutil
from scipy.io import loadmat

import h5py
import tempfile

import numpy as np
import torch
import yaml
from pygrappa.mdgrappa import mdgrappa
import sys
path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, path)
import fastmri
from fastmri.data import transforms
from fastmri.data.transforms import VarNetSample, VarNetDataTransformVolume4DGrappa
from fastmri.data.subsample import create_mask_for_mask_type


def et_query(
        root: etree.Element,
        qlist: Sequence[str],
        namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.

    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.

    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.

    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


def fetch_dir(
        key: str, data_config_file: Union[str, Path, os.PathLike] = "fastmri_dirs.yaml"
) -> Path:
    """
    Data directory fetcher.

    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `knee_path` and `brain_path`
    and this function will retrieve the requested subsplit of the data for use.

    Args:
        key: key to retrieve path from data_config_file. Expected to be in
            ("knee_path", "brain_path", "log_path").
        data_config_file: Optional; Default path config file to fetch path
            from.

    Returns:
        The path to the specified directory.
    """
    data_config_file = Path(data_config_file)
    if not data_config_file.is_file():
        default_config = {
            "knee_path": "/path/to/knee",
            "brain_path": "/path/to/brain",
            "log_path": ".",
        }
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        data_dir = default_config[key]

        warn(
            f"Path config at {data_config_file.resolve()} does not exist. "
            "A template has been created for you. "
            "Please enter the directory paths for your system to have defaults."
        )
    else:
        with open(data_config_file, "r") as f:
            data_dir = yaml.safe_load(f)[key]

    return Path(data_dir)


class CombinedSliceDataset(torch.utils.data.Dataset):
    """
    A container for combining slice datasets.
    """

    def __init__(
            self,
            roots: Sequence[Path],
            challenges: Sequence[str],
            transforms: Optional[Sequence[Optional[Callable]]] = None,
            sample_rates: Optional[Sequence[Optional[float]]] = None,
            volume_sample_rates: Optional[Sequence[Optional[float]]] = None,
            use_dataset_cache: bool = False,
            dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
            num_cols: Optional[Tuple[int]] = None,
    ):
        """
        Args:
            roots: Paths to the datasets.
            challenges: "singlecoil" or "multicoil" depending on which
                challenge to use.
            transforms: Optional; A sequence of callable objects that
                preprocesses the raw data into appropriate form. The transform
                function should take 'kspace', 'target', 'attributes',
                'filename', and 'slice' as inputs. 'target' may be null for
                test data.
            sample_rates: Optional; A sequence of floats between 0 and 1.
                This controls what fraction of the slices should be loaded.
                When creating subsampled datasets either set sample_rates
                (sample by slices) or volume_sample_rates (sample by volumes)
                but not both.
            volume_sample_rates: Optional; A sequence of floats between 0 and 1.
                This controls what fraction of the volumes should be loaded.
                When creating subsampled datasets either set sample_rates
                (sample by slices) or volume_sample_rates (sample by volumes)
                but not both.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
        """
        if sample_rates is not None and volume_sample_rates is not None:
            raise ValueError(
                "either set sample_rates (sample by slices) or volume_sample_rates (sample by volumes) but not both"
            )
        if transforms is None:
            transforms = [None] * len(roots)
        if sample_rates is None:
            sample_rates = [None] * len(roots)
        if volume_sample_rates is None:
            volume_sample_rates = [None] * len(roots)
        if not (
                len(roots)
                == len(transforms)
                == len(challenges)
                == len(sample_rates)
                == len(volume_sample_rates)
        ):
            raise ValueError(
                "Lengths of roots, transforms, challenges, sample_rates do not match"
            )

        self.datasets = []
        self.examples: List[Tuple[Path, int, Dict[str, object]]] = []
        for i in range(len(roots)):
            self.datasets.append(
                SliceDataset(
                    root=roots[i],
                    transform=transforms[i],
                    challenge=challenges[i],
                    sample_rate=sample_rates[i],
                    volume_sample_rate=volume_sample_rates[i],
                    use_dataset_cache=use_dataset_cache,
                    dataset_cache_file=dataset_cache_file,
                    num_cols=num_cols,
                )
            )

            self.examples = self.examples + self.datasets[-1].examples

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, i):
        for dataset in self.datasets:
            if i < len(dataset):
                return dataset[i]
            else:
                i = i - len(dataset)


class SliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
            self,
            root: Union[str, Path, os.PathLike],
            challenge: str,
            transform: Optional[Callable] = None,
            use_dataset_cache: bool = False,
            sample_rate: Optional[float] = None,
            volume_sample_rate: Optional[float] = None,
            dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
            num_cols: Optional[Tuple[int]] = None,
    ):
        """
        Args:
            root: Path to the dataset.
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
        """
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.dataset_cache_file = Path(dataset_cache_file)

        self.transform = transform
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.examples = []

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())
            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname)

                self.examples += [
                    (fname, slice_ind, metadata) for slice_ind in range(num_slices)
                ]

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.examples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as f:
                    pickle.dump(dataset_cache, f)
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.examples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list(set([f[0].stem for f in self.examples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.examples = [
                example for example in self.examples if example[0].stem in sampled_vols
            ]

        if num_cols:
            self.examples = [
                ex
                for ex in self.examples
                if ex[2]["encoding_size"][1] in num_cols  # type: ignore
            ]

    @staticmethod
    def _retrieve_metadata(fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
        }

        return metadata, num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, dataslice, metadata = self.examples[i]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]

            mask = np.asarray(hf["mask"]) if "mask" in hf else None

            target = hf[self.recons_key][dataslice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)
            attrs.update(metadata)

        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname.name, dataslice)
        else:
            sample = self.transform(kspace, mask, target, attrs, fname.name, dataslice)

        return sample


class VolumeDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root: Union[str, Path, os.PathLike],
            challenge: str,
            transform: Optional[Callable] = None,
            use_dataset_cache: bool = False,
            sample_rate: Optional[float] = None,
            volume_sample_rate: Optional[float] = None,
            dataset_cache_file: Union[str, Path,
                                      os.PathLike] = "/opt/tmp/dataset_cache.pkl",
            num_cols: Optional[Tuple[int]] = None,
            cache_path=None,
    ):
        """
                Args:
                    root: Path to the dataset.
                    challenge: "singlecoil" or "multicoil" depending on which challenge
                        to use.
                    transform: Optional; A callable object that pre-processes the raw
                        data into appropriate form. The transform function should take
                        'kspace', 'target', 'attributes', 'filename', and 'slice' as
                        inputs. 'target' may be null for test data.
                    use_dataset_cache: Whether to cache dataset metadata. This is very
                        useful for large datasets like the brain data.
                    sample_rate: Optional; A float between 0 and 1. This controls what fraction
                        of the slices should be loaded. Defaults to 1 if no value is given.
                        When creating a sampled dataset either set sample_rate (sample by slices)
                        or volume_sample_rate (sample by volumes) but not both.
                    volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                        of the volumes should be loaded. Defaults to 1 if no value is given.
                        When creating a sampled dataset either set sample_rate (sample by slices)
                        or volume_sample_rate (sample by volumes) but not both.
                    dataset_cache_file: Optional; A file in which to cache dataset
                        information for faster load times.
                    num_cols: Optional; If provided, only slices with the desired
                        number of columns will be considered.
                """
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.dataset_cache_file = Path(dataset_cache_file)

        self.transform = transform
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.examples = []
        if cache_path is None:
            self.cache_path = "."
        else:
            self.cache_path = cache_path
        print(f"Saving cache at {self.cache_path}")

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        # if dataset_cache.get(root) is None or not use_dataset_cache:
        #     files = list(Path(root).iterdir())
        #     for fname in sorted(files):
        #         metadata, num_slices = SliceDataset._retrieve_metadata(fname)
        #         self.examples += [(fname, metadata)]

        #     if dataset_cache.get(root) is None and use_dataset_cache:
        #         dataset_cache[root] = self.examples
        #         logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
        #         with open(self.dataset_cache_file, "wb") as f:
        #             pickle.dump(dataset_cache, f)
        # else:
        #     logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
        #     self.examples = dataset_cache[root]

        # # subsample if desired
        # if sample_rate < 1.0:  # sample by slice
        #     random.shuffle(self.examples)
        #     num_examples = round(len(self.examples) * sample_rate)
        #     self.examples = self.examples[:num_examples]
        # elif volume_sample_rate < 1.0:  # sample by volume
        #     vol_names = sorted(list(set([f[0].stem for f in self.examples])))
        #     random.shuffle(vol_names)
        #     num_volumes = round(len(vol_names) * volume_sample_rate)
        #     sampled_vols = vol_names[:num_volumes]
        #     self.examples = [
        #         example for example in self.examples if example[0].stem in sampled_vols
        #     ]

        # if num_cols:
        #     self.examples = [
        #         ex
        #         for ex in self.examples
        #         if ex[1]["encoding_size"][1] in num_cols  # type: ignore
        #     ]

    def __len__(self):
        return len(self.examples)

    def reco(self, kspace, down_sampling_factor, z_extend=None):
        k_space_downsampled = kspace
        kspace_center_z = kspace.shape[-3] // 2
        kspace_center_y = kspace.shape[-2] // 2
        kspace_center_x = kspace.shape[-1] // 2
        new_kspace_extend_z_half = z_extend // 2
        new_kspace_extend_y_half = int(kspace.shape[-2] // (down_sampling_factor * 2))
        new_kspace_extend_x_half = int(kspace.shape[-1] // (down_sampling_factor * 2))
        k_space_downsampled = k_space_downsampled[:,
                              kspace_center_z - new_kspace_extend_z_half:kspace_center_z + new_kspace_extend_z_half,
                              kspace_center_y - new_kspace_extend_y_half:kspace_center_y + new_kspace_extend_y_half,
                              kspace_center_x - new_kspace_extend_x_half:kspace_center_x + new_kspace_extend_x_half]
        k_space_downsampled = torch.view_as_real(torch.from_numpy(k_space_downsampled))
        volume = fastmri.ifft3c(k_space_downsampled)
        volume = fastmri.complex_abs(volume)
        volume = fastmri.rss(volume, dim=0)
        return volume, k_space_downsampled

    def get_cache(self, i: int):
        file_location = os.path.join(self.cache_path, f"{i}.pkl")
        if os.path.exists(file_location):
            with open(file_location, "rb") as handle:
                sample = pickle.load(handle)
        else:
            sample = self.generate_sample(i)
            with open(file_location, 'wb') as handle:
                pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if self.transform is None:
            return sample

        sample = self.transform(sample[0], sample[1],
                                sample[2], sample[3], sample[4], -1)
        return sample

    def __getitem__(self, i: int):
        return self.get_cache(i)

    def generate_sample(self, i: int):
        if len(self.examples[i]) > 2:
            fname, _, metadata = self.examples[i]
        else:
            fname, metadata = self.examples[i]

        with h5py.File(fname, "r") as hf:
            kspace = np.asarray(hf["kspace"])
            kspace = np.transpose(kspace, (1, 0, 2, 3))

            mask = np.asarray(hf["mask"]) if "mask" in hf else None

            # Random slice selection
            num_slices = 10
            downsampling_factor = 2
            x_y_extend = 320 // downsampling_factor
            rand_first_slice = random.randint(0, kspace.shape[1] - num_slices)
            rand_last_slice = rand_first_slice + num_slices
            kspace = kspace[:, rand_first_slice:rand_last_slice]
            target, k_space_downsampled = self.reco(kspace, downsampling_factor)
            target = transforms.complex_center_crop_3d(
                target, (num_slices, x_y_extend, x_y_extend))
            kspace = k_space_downsampled
            attrs = dict(hf.attrs)
            attrs.update(metadata)
            # TODO: Investigate effect of padding
            # attrs["padding_left"] = 0
            # attrs["padding_right"] = -1

            sample = (kspace, mask, target, attrs, fname.name, -1)
        return sample


class RealCESTData(VolumeDataset):
    def __init__(self,
                 root: Union[str, Path, os.PathLike],
                 challenge: str,
                 transform: Optional[Callable] = None,
                 use_dataset_cache: bool = False,
                 sample_rate: Optional[float] = None,
                 volume_sample_rate: Optional[float] = None,
                 dataset_cache_file: Union[str, Path,
                                           os.PathLike] = "/opt/tmp/dataset_cache.pkl",
                 num_cols: Optional[Tuple[int]] = None,
                 cache_path=None,
                 num_offsets: int = 16,
                 number_of_simultaneous_offsets=16,
                 grappa_init=True):
        super().__init__(root, challenge, transform, use_dataset_cache, sample_rate,
                         volume_sample_rate, dataset_cache_file, num_cols, cache_path)
        self.cases = []
        self.root = root
        self.number_of_simultaneous_offsets = number_of_simultaneous_offsets
        self.grappa_init = grappa_init
        print(f"Using {number_of_simultaneous_offsets} of offsets simultaneously.")
        self.load_data()
        self.ram_cache = {}

    def load_data(self):
        root_path = self.root  # r"E:\Lukas\cest_data\Probanden\Mareike\output\train"
        scans = [os.path.join(root_path, file) for file in os.listdir(root_path) if "cest" in file.lower() and not "lowres" in file.lower()]
        for scan in scans:
            with h5py.File(scan, "r") as f:
                kspace = np.array(f["kspace"])  # offset, slices, readout, channels, phase
            # kspace = 100 * (kspace["real"] + 1j* kspace["imag"])
            kspace = kspace["real"] + 1j* kspace["imag"]
            kspace = np.transpose(kspace, (3, 0, 1, 2, 4))  # channels, offsets, slices, readout, phase
            kspace = torch.from_numpy(np.stack((kspace.real, kspace.imag), -1))
            offset_targets = []
            for offset in range(kspace.shape[1]):
                target = fastmri.ifft3c(kspace[:, offset])
                target = fastmri.complex_abs(target)
                target = fastmri.rss(target, dim=0).squeeze()
                target = transforms.complex_center_crop_3d(target, (target.shape[0], 128, 128))
                offset_targets.append(target)
                
            offset_targets = torch.stack(offset_targets, 0)
            factor = int(16 / self.number_of_simultaneous_offsets)
            for k in range(factor):
                kspace_offset_stack = kspace[:, int(k * self.number_of_simultaneous_offsets):int((k + 1) * self.number_of_simultaneous_offsets)]
                offset_targets_stack = offset_targets[int(k * self.number_of_simultaneous_offsets):int((k + 1) * self.number_of_simultaneous_offsets)]
                sample = (kspace_offset_stack, None, offset_targets_stack, {"max": 0, "padding_left": 0, "padding_right": 0, "recon_size": [0, 0]}, "test")
                self.cases.append(sample)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        sample = self.get_cache(i)
        # sample = self.generate_sample(i)
        return sample
    
    def get_cache(self, i: int):
        if str(i) in self.ram_cache.keys():
            return self.ram_cache[str(i)]
        file_location = os.path.join(self.cache_path, f"{i}.pkl")
        if os.path.exists(file_location):
            with open(file_location, "rb") as handle:
                samples = pickle.load(handle)
        else:
            samples = self.generate_sample(i)
            with tempfile.TemporaryDirectory() as tempdir:
                temp_location = os.path.join(tempdir, f"{i}.pkl")
                with open(temp_location, 'wb') as handle:
                    pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if Path(temp_location).is_file() and not Path(file_location).is_file():
                    try:
                        shutil.move(temp_location, file_location)
                    except FileExistsError:
                        pass
        self.ram_cache[str(i)] = samples
        return samples
    
    def generate_sample(self, i):
        sample = self.cases[i]
        sample = self.transform(sample[0], sample[1],
                        sample[2], sample[3], sample[4], -1)
        return sample
    

class CenterKspaceCEST(RealCESTData):
    def generate_sample(self, i):
        sample = self.cases[i]
        kspace, mask, target, attrs, fname = sample
        center_kspace = kspace[:, :, 5:15, 32:96, 64:192]
        mask = torch.ones(center_kspace.shape)
        center_kspace = torch.nn.functional.pad(center_kspace, (0, 0, 64, 64, 32, 32, 5, 5))
        mask = torch.nn.functional.pad(mask, (0, 0, 64, 64, 32, 32, 5, 5))
        acs = center_kspace[..., 64-12:64+12, :, :]
        sample = VarNetSample(
            masked_kspace=center_kspace,
            mask=mask.to(torch.bool),
            filled_kspace=center_kspace,
            acs=acs,
            num_low_frequencies=0,
            target=target,
            fname=fname,
            slice_num=-1,
            max_value=1,
            crop_size=[1],
            masked_kspace_imputed=center_kspace,
        )
        return sample


class CESTDataset(VolumeDataset):
    def __init__(self,
                 root: Union[str, Path, os.PathLike],
                 challenge: str,
                 transform: Optional[Callable] = None,
                 use_dataset_cache: bool = False,
                 sample_rate: Optional[float] = None,
                 volume_sample_rate: Optional[float] = None,
                 dataset_cache_file: Union[str, Path,
                                           os.PathLike] = "/opt/tmp/dataset_cache.pkl",
                 num_cols: Optional[Tuple[int]] = None,
                 cache_path=None,
                 num_offsets: int = 8):
        super().__init__(root, challenge, transform, use_dataset_cache, sample_rate,
                         volume_sample_rate, dataset_cache_file, num_cols, cache_path)
        self.num_offsets = num_offsets
        # Undersampling mask is fixed for now
        self.apply_grappa = False

    def apply_virtual_cest_contrast(self, kspace, target, offset: int):
        random_num = 1e3 * np.random.rand() + 1e4
        kspace = deepcopy(kspace) * random_num
        target = deepcopy(target) * random_num
        return kspace, target

    def generate_offset(self, kspace, hf, metadata, fname, offset, grappa_weights=None):
        downsampling_factor = 2.5  # 3.48
        x_y_extend = 320 // downsampling_factor
        z_extend = 16  # kspace.shape[-3]  # 8
        target, k_space_downsampled = self.reco(kspace, downsampling_factor, z_extend)
        z_extend = target.shape[0]  # Shape might change due to rounding error
        mask = None
        if self.apply_grappa:
            mask = create_mask_for_mask_type("equispaced_fraction_3d", [0.08], [2]).calculate_acceleration_mask_3D(
                None, None, None, None, [1, 8, k_space_downsampled.shape[-2], 1])
            mask = mask[None, :, None, :, None]
            mask = np.repeat(np.repeat(np.repeat(mask, k_space_downsampled.shape[0], 0), 2, -1),
                             k_space_downsampled.shape[2], 2)
            k_space_downsampled_undersampled = mask * k_space_downsampled.numpy()
            acs = k_space_downsampled[:, :,
                  k_space_downsampled.shape[2] // 2 - 10:k_space_downsampled.shape[2] // 2 + 10,
                  k_space_downsampled.shape[3] // 2 - 10:k_space_downsampled.shape[3] // 2 + 10]
            acs = acs.numpy()
            if grappa_weights is None:
                grappa_weights = self.calculate_grappa_weights(k_space_downsampled_undersampled, acs)

            k_space_downsampled_undersampled_grappa = self.apply_grappa_weights(k_space_downsampled_undersampled,
                                                                                grappa_weights)
            k_space_downsampled_undersampled_grappa = torch.from_numpy(k_space_downsampled_undersampled_grappa)
            k_space_downsampled = k_space_downsampled_undersampled_grappa
        target = transforms.complex_center_crop_3d(
            target, (z_extend, x_y_extend, x_y_extend))
        kspace = k_space_downsampled
        attrs = dict(hf.attrs)
        attrs.update(metadata)

        kspace, target = self.apply_virtual_cest_contrast(kspace, target, offset)

        sample = (kspace, mask, target, attrs, fname.name, -1)
        return sample, grappa_weights

    def calculate_grappa_weights(self, kspace, acs):
        acs = acs[..., 0] + 1j * acs[..., 1]
        kspace = kspace[..., 0] + 1j * kspace[..., 1]
        acs = np.swapaxes(acs, 1, -1)
        kspace = np.swapaxes(kspace, 1, -1)
        _, grappa_weights = mdgrappa(kspace, acs, coil_axis=0, kernel_size=(5, 5, 5), ret_weights=True)
        return grappa_weights

    def apply_grappa_weights(self, sample, grappa_weights):
        sample = sample[..., 0] + 1j * sample[..., 1]
        sample = np.swapaxes(sample, 1, -1)
        sample_grappa = mdgrappa(sample, sample, weights=grappa_weights, coil_axis=0, kernel_size=(5, 5, 5))
        sample_grappa = np.swapaxes(sample_grappa, -1, 1)
        sample_grappa = np.stack((np.real(sample_grappa), np.imag(sample_grappa)), -1)
        return sample_grappa

    def generate_sample(self, i: int):
        if len(self.examples[i]) > 2:
            fname, _, metadata = self.examples[i]
        else:
            fname, metadata = self.examples[i]

        samples = []

        with h5py.File(fname, "r") as hf:
            kspace = np.asarray(hf["kspace"])
            kspace = np.transpose(kspace, (1, 0, 2, 3))

            kspace = fastmri.fft1c(torch.from_numpy(np.stack((np.real(kspace), np.imag(kspace)), -1)), dim=-4).numpy()
            kspace = kspace[..., 0] + 1j * kspace[..., 1]

            grappa_weights = None
            for o in range(self.num_offsets):
                sample, grappa_weights = self.generate_offset(kspace, hf, metadata, fname, o, grappa_weights)
                samples.append(sample)

        kspace = torch.stack([s[0] for s in samples], dim=1)
        target = torch.stack([s[2] for s in samples], 0)
        return (kspace, sample[1], target, sample[3], fname.name, -1)

    def get_cache(self, i: int):
        file_location = os.path.join(self.cache_path, f"{i}.pkl")
        if os.path.exists(file_location):
            with open(file_location, "rb") as handle:
                samples = pickle.load(handle)
        else:
            samples = self.generate_sample(i)
            with tempfile.TemporaryDirectory() as tempdir:
                temp_location = os.path.join(tempdir, f"{i}.pkl")
                with open(temp_location, 'wb') as handle:
                    pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if Path(temp_location).is_file() and not Path(file_location).is_file():
                    try:
                        shutil.move(temp_location, file_location)
                    except FileExistsError:
                        pass

        if self.transform is None:
            masked_kspace = samples[0]
            mask_torch = np.repeat(samples[1][:, None], masked_kspace.shape[1], 1)

            samples = VarNetSample(
                masked_kspace=masked_kspace.to(torch.float32),
                mask=torch.from_numpy(mask_torch).to(torch.bool),
                num_low_frequencies=0,
                target=samples[2],
                fname=samples[4],
                slice_num=-1,
                max_value=-1,
                crop_size=-1,
            )
            return samples
        samples = self.transform(*samples)

        return samples


if __name__ == "__main__":
    from fastmri.data.transforms import VarNetDataTransformVolume4D
    from fastmri.data.subsample import create_mask_for_mask_type
    import matplotlib.pyplot as plt
    from utils.matplotlib_viewer import scroll_slices
    from tqdm import trange
    from pygrappa.mdgrappa import mdgrappa
    import matplotlib.pyplot as plt

    mask = create_mask_for_mask_type("equispaced_fraction_full_center_3d", [0], [9])
    # use random masks for train transform, fixed masks for val transform
    transform = VarNetDataTransformVolume4DGrappa(mask_func=mask, use_seed=False)
    # cest_ds = CESTDataset("/home/woody/iwi5/iwi5044h/fastMRI/multicoil_train", "multicoil", transform, use_dataset_cache=False, cache_path="/home/woody/iwi5/iwi5044h/Code/fastMRI/cache_test")
    cest_ds = RealCESTData(r"C:\Zwischenspeicher\output\multicoil_val", "multicoil", transform=transform, use_dataset_cache=False,
                          cache_path=r"C:\Users\follels\Documents\fastMRI\cache\center_fully\cache_val")

    for i in trange(len(cest_ds)):
        item = cest_ds.__getitem__(i)
        print(f"\n\nItem {i}")
    
    mask = create_mask_for_mask_type("equispaced_fraction_full_center_3d", [0], [9])
    # use random masks for train transform, fixed masks for val transform
    transform = VarNetDataTransformVolume4DGrappa(mask_func=mask, use_seed=False)
    # cest_ds = CESTDataset("/home/woody/iwi5/iwi5044h/fastMRI/multicoil_train", "multicoil", transform, use_dataset_cache=False, cache_path="/home/woody/iwi5/iwi5044h/Code/fastMRI/cache_test")
    cest_ds = RealCESTData(r"C:\Zwischenspeicher\output\multicoil_train", "multicoil", transform=transform, use_dataset_cache=False,
                          cache_path=r"C:\Users\follels\Documents\fastMRI\cache\center_fully\cache_train")

    for i in trange(len(cest_ds)):
        item = cest_ds.__getitem__(i)
        print(f"\n\nItem {i}")
        
        # plt.subplot(1,2,1)
        # plt.plot(item.target[:, 10, 24:32, 65:85].reshape((8, -1)).sum(1))
        # plt.title("Fully-sampled")
        # k_space_downsampled = item.filled_kspace
        # k_space_downsampled = torch.view_as_real(k_space_downsampled[..., 0] + 1j * k_space_downsampled[..., 1])
        # volume = fastmri.ifft3c(k_space_downsampled)
        # volume = fastmri.complex_abs(volume)
        # volume = fastmri.rss(volume, dim=0)
        # volume = volume[..., 64:192]
        # plt.subplot(1,2,2)
        # plt.title("GRAPPA Undersampling")
        # plt.plot(volume[:, 10, 24:32, 65:85].reshape((8, -1)).sum(1))
        # plt.show()
        # print()
        
        # offset = 0
        # k_space_downsampled = item.filled_kspace[:, offset]
        # k_space_downsampled = torch.view_as_real(k_space_downsampled[..., 0] + 1j * k_space_downsampled[..., 1])
        # volume = fastmri.ifft3c(k_space_downsampled)
        # volume = fastmri.complex_abs(volume)
        # volume = fastmri.rss(volume, dim=0)
        # volume = (volume - volume.min()) / (volume.max() - volume.min())
        # plt.imshow(volume[10])
        # plt.show()
        # print()
        # for offset in range(item.target.shape[0]):
        # offset = 0
        # mask = item.mask.numpy().squeeze()[..., 0]
        # # vol = item.target[offset].numpy().squeeze()
        # vol = item.target.numpy().squeeze()
        # print(vol.shape)
        # # mask = mask[offset, ..., 0]
        # plt.imshow(mask)
        # plt.title(f"Sample {i}, offset {offset}")
        # plt.show()
        # vol = (vol - vol.min()) / (vol.max() - vol.min())
        # vol = np.moveaxis(vol, 0, -1)
        # scroll_slices(vol, title=f"Sample {i} Offset {offset}")

        # k_space_downsampled = item.masked_kspace#[:, offset]
        # k_space_downsampled = torch.view_as_real(k_space_downsampled[..., 0] + 1j * k_space_downsampled[..., 1])
        # volume = fastmri.ifft3c(k_space_downsampled)
        # volume = fastmri.complex_abs(volume)
        # volume = fastmri.rss(volume, dim=0)
        # volume = (volume - volume.min()) / (volume.max() - volume.min())
        # volume = np.moveaxis(volume.numpy(), 0, -1)
        # scroll_slices(volume, title=f"Sample {i} Offset {offset}")


    # from fastmri.models.varnet_4d import VarNet4D
    # varnet = VarNet4D(4, 2, 4, 3, 2).to("cuda")
    # item = cest_ds.__getitem__(0)
    # print(item.masked_kspace.shape)
    # print(item.mask.shape)
    # print(item.target.shape)
    # print(item.num_low_frequencies)
    # ret = varnet(item.masked_kspace.unsqueeze(0).to("cuda"), item.mask.unsqueeze(0).to("cuda"), item.num_low_frequencies)
    # print(ret.shape)
    # print(f"GPU GB allocated {torch.cuda.max_memory_allocated() / 10**9}")

    # rcd = RealCESTData()
    # print(rcd.__getitem__(0)[0].shape)
