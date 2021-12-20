from functools import lru_cache

import numpy as np
import torch
from pygrappa import mdgrappa

import fastmri
from fastmri.data.mri_data import RealCESTData


@lru_cache
def generate_test_sample(device="cpu", apply_grappa=False):
    dataset = RealCESTData()
    kspace, acs, _ = dataset.__getitem__(0)
    # Select first 8 offsets
    kspace = kspace[:, :8]
    kspace = kspace.numpy()
    # kspace = fastmri.fft1c(kspace, dim=-4).numpy()
    # Center crop z dim
    # kspace = kspace[:, :, kspace.shape[2] // 2 - 4:kspace.shape[2] // 2 + 4] * 1e4
    mask = np.abs(kspace[..., 0] + 1j * kspace[..., 1]) > 0

    kspace = kspace[..., 0] + 1j * kspace[..., 1]
    acs = acs[..., 0] + 1j * acs[..., 1]
    kspace = np.swapaxes(kspace, 2, -1)
    acs = np.swapaxes(acs.numpy(), 1, -1)
    if apply_grappa:
        weights = None
        for o in range(kspace.shape[1]):
            if weights is None:
                kspace[:, o], weights = mdgrappa(kspace[:, o].copy(), acs, (5, 5, 5), coil_axis=0,
                                                 ret_weights=True)
            else:
                kspace[:, o] = mdgrappa(kspace[:, o].copy(), acs, (5, 5, 5), coil_axis=0,
                                        weights=weights)

    kspace = np.stack((np.real(kspace), np.imag(kspace)), -1)
    kspace = np.swapaxes(kspace, -2, 2)
    kspace = torch.from_numpy(kspace)
    mask = np.repeat(mask[..., None], 2, -1)
    mask = torch.from_numpy(mask)
    mask = mask[None]
    kspace = kspace[None]
    # kspace, mask, num_low_frequencies
    return kspace.to(device), mask.to(device), 1


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
