import os
from typing import List, Optional

from skimage.transform import resize
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft

from fastmri.data.grappa.kernel_application import apply_kernel
from fastmri.data.grappa.kernel_estimation import kernel_estimation


def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt((data ** 2).sum(dim))


def complex_abs(data):
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data ** 2).sum(dim=-1).sqrt()


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
        x: torch.Tensor,
        shift: List[int],
        dim: List[int],
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.

    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.

    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)


def fft1c_new(data: torch.Tensor, norm: str = "ortho", dim=-2) -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data containing
        norm: Normalization mode. See ``torch.fft.ifft``.
        dim: Dimension to transform

    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = fftshift(data, dim=[dim])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(dim + 1), norm=norm
        )
    )
    data = ifftshift(data, dim=[dim])

    return data


def ifft2c_variant(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.

    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = fftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = ifftshift(data, dim=[-3, -2])

    return data


def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.

    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


def ifft3c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 3-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 4 dimensions:
            dimensions -4, -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.

    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-4, -3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-3, -2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-4, -3, -2])

    return data


def load_data(base_path, fastmri=False, unsat=False, simulate_us=False):
    if not fastmri:
        for file in os.listdir(base_path):
            specifier = "cest_knee_unsat_raw_real" if unsat else "cest_knee_raw_real"
            if specifier in file:
                file_path = os.path.join(base_path, file)
                if not file_path.endswith("mat"):
                    raise NotImplementedError("Can only process .mat files.")
                f = h5py.File(file_path, 'r')
                data = f.get('r')
                data = np.array(data)
                if unsat:
                    data = np.moveaxis(data, np.arange(len(data.shape)), [-1, 1, 2, 3, 4, 5, 0, 6]).squeeze()[:, None]
                else:
                    data = np.moveaxis(data, np.arange(len(data.shape)),
                                       [1, -1, 2, 3, 0, 4])  # maybe switch 3 and 4 ie phase and freq?#
                    npad = ((0, 0), (0, 0), (4, 2), (1, 3), (0, 0), (0, 0))
                    data = np.pad(data, pad_width=npad, mode='constant', constant_values=0)
                print(f"Out of {data.shape[-1]} repetitions, selecting the first one.")
                real = data[..., 0]
                f = h5py.File(file_path.replace("real", "imag"), 'r')
                data = f.get('im')
                data = np.array(data)
                if unsat:
                    data = np.moveaxis(data, np.arange(len(data.shape)), [-1, 1, 2, 3, 4, 5, 0, 6]).squeeze()[:, None]
                else:
                    data = np.moveaxis(data, np.arange(len(data.shape)),
                                       [1, -1, 2, 3, 0, 4])  # maybe switch 3 and 4 ie phase and freq?
                    npad = ((0, 0), (0, 0), (4, 2), (1, 3), (0, 0), (0, 0))
                    data = np.pad(data, pad_width=npad, mode='constant', constant_values=0)
                im = data[..., 0]
                complex_case = np.stack((real, im), -1)
                mask = np.abs(real + 1j * im).sum((0, 1, 4), keepdims=True) > 0.0
                mask = np.repeat(np.repeat(mask, im.shape[1], axis=1)[..., None], 2, axis=-1)
                if simulate_us:
                    us_mask = np.load(r"C:\Users\follels\Documents\fastMRI\fastmri\data\kspace_eliptical_mask_debug.npy")
                    us_mask = us_mask[:, 0, ...][None]
                    us_mask = resize(us_mask, mask.shape) > 0.5
                    cube_size = 8
                    us_mask[:, :, us_mask.shape[2] // 2 - cube_size//2:us_mask.shape[2] // 2 + cube_size//2, us_mask.shape[3] // 2 - cube_size//2:us_mask.shape[3] // 2 + cube_size//2] = True
                    mask = mask * us_mask
                    complex_case = complex_case * us_mask
                # k-space target shape: (coils, offsets, slices, x, y)
                return complex_case, mask
    else:
        with h5py.File(base_path, "r") as hf:
            kspace = np.asarray(hf["kspace"])
            kspace = np.transpose(kspace, (1, 0, 2, 3))
            mask = np.abs(kspace).sum((0, 2), keepdims=True) > 0
            kspace = np.stack((np.real(kspace), np.imag(kspace)), -1)
            if simulate_us:
                us_mask = np.load(r"C:\Users\follels\Documents\fastMRI\fastmri\data\kspace_eliptical_mask_debug.npy")
                us_mask = us_mask[:, 0, ...]
                us_mask = us_mask[:, :, None, ..., 0, 0]
                us_mask = resize(us_mask, mask.shape) > 0.5
                cube_size = 8
                us_mask[:, us_mask.shape[1] // 2 - cube_size // 2:us_mask.shape[1] // 2 + cube_size // 2, :,
                        us_mask.shape[3] // 2 - cube_size // 2:us_mask.shape[3] // 2 + cube_size // 2] = True
                mask = us_mask
                mask = np.repeat(mask[..., None], 2, axis=-1)
                kspace = kspace * mask
                mask = mask[..., 0]
            return kspace, mask


def reconstruct(kspace, fastmri=False, grappa=False):
    if grappa:
        grappa_kernels = kernel_estimation(kspace, mask, ny=3, lamda=1e-6)
        filled_kspace = apply_kernel(kspace, mask, grappa_kernels)

    if not fastmri:
        select_offset = 0
        select_slice = 10
        x = torch.from_numpy(kspace[:, select_offset])
        x = ifft3c_new(x)
        x = x[:, select_slice]
    else:
        select_slice = 10
        x = torch.from_numpy(kspace)
        x = fft1c_new(x, dim=-4)
        x = ifft3c_new(x)
        x = x[:, select_slice]
    x = complex_abs(x)
    x = rss(x, 0)
    return x


def view(volume, title: str):
    if len(volume.shape) > 2:
        volume = volume[10]
    plt.imshow(volume, vmin=volume.min(), vmax=volume.max())
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    fastmri = False
    unsat = False
    simulate_us = False
    if fastmri:
        base_path = r"E:\Lukas\multicoil_train\file1000000.h5"
    else:
        base_path = r"E:\Lukas\cest_data"
    kspace, mask = load_data(base_path=base_path, fastmri=fastmri, unsat=unsat, simulate_us=simulate_us)
    if not fastmri:
        mask = mask[0, 0, ..., 0]
    view(mask.squeeze(), "Mask")
    volume = reconstruct(kspace, fastmri, True)
    view(volume, "Volume")
