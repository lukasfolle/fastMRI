from functools import lru_cache
from copy import deepcopy

import numpy as np
import torch
from scipy.io import loadmat

import fastmri
from fastmri.data.subsample import create_mask_for_mask_type


@lru_cache
def generate_test_sample(device="cpu", apply_grappa=False):
    kspace = np.stack([loadmat(r"U:\testCEST_CS\real.mat")["re"], loadmat(r"U:\testCEST_CS\imag.mat")["im"]], -1)
    kspace = kspace.transpose((1, 3, 0, 2, 4, 5))
    kspace = kspace[..., 0] + 1j * kspace[..., 1]

    us_masks = np.stack([create_mask_for_mask_type("poisson_3d", [0], [6]).calculate_acceleration_mask_3D(None, None, None, None,
                                                                                               [1, 16, kspace.shape[3], 1], seed) for seed in range(kspace.shape[-1])], -1)
    us_masks = us_masks.reshape((1, 16, 1, 128, kspace.shape[-1]))
    us_kspace = kspace * us_masks
    us_kspace = np.stack((np.real(us_kspace), np.imag(us_kspace)), -1)
    us_kspace = torch.from_numpy(us_kspace)
    us_masks = np.repeat(us_masks[..., None], 2, -1)
    us_masks = torch.from_numpy(us_masks)
    us_masks = us_masks[None]
    us_kspace = us_kspace[None]

    def impute_missing_kspace_over_offsets(kspace):
        filled_kspace = deepcopy(kspace)
        for offset in range(kspace.shape[-2]):
            filled_kspace[..., offset, :] = torch.where(filled_kspace[..., offset, :] == 0, torch.mean(kspace, -2),
                                                   filled_kspace[..., offset, :])
        return filled_kspace

    us_kspace = impute_missing_kspace_over_offsets(us_kspace)

    reco_offsets = []
    kspace = np.stack((np.real(kspace), np.imag(kspace)), -1)
    kspace = torch.from_numpy(kspace)
    for offset in range(kspace.shape[-2]):
        reco = fastmri.ifft3c(kspace[..., offset, :])
        reco = fastmri.complex_abs(reco)
        reco = fastmri.rss(reco, dim=0).squeeze()
        reco_offsets.append(reco)
    reco_offsets = torch.stack(reco_offsets, -1)

    # kspace, mask, num_low_frequencies
    return us_kspace.to(device), us_masks.to(device), 1, reco_offsets


if __name__ == "__main__":
    from utils.matplotlib_viewer import scroll_slices

    sample = generate_test_sample()
    k_space_downsampled = sample[0]
    # 1 dim, ch, off, slices
    k_space_downsampled = k_space_downsampled[0, :, 0].squeeze()
    k_space_downsampled = k_space_downsampled[..., 0] + 1j * k_space_downsampled[..., 1]
    k_space_downsampled = torch.view_as_real(k_space_downsampled)
    volume = fastmri.ifft3c(k_space_downsampled)
    volume = fastmri.complex_abs(volume)
    volume = fastmri.rss(volume, dim=0)
    volume = np.swapaxes(volume, 0, -1)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-6)
    scroll_slices(volume)
