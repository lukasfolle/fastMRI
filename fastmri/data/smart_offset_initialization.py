import torch
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


def smart_offset_init(kspace):
    smart_kspace = deepcopy(kspace)
    for offset in range(kspace.shape[1]):
        # TODO: Check if valid entries i.e. non-zero get changed by this code. Looks like they do get changed in matplotlib...
        torch.argmax(kspace[:, offset])
        offset_kspace = kspace[:, offset]
        valid_offset_values = offset_kspace[:, :, 0, :, 0]
        non_zero_mean_all_offsets = kspace[:, :, :, 0, :, 0].sum(1) / ((kspace[:, :, :, 0, :, 0] == 0).sum(1) + 1e-10)
        valid_offset_values[valid_offset_values == 0] = non_zero_mean_all_offsets[valid_offset_values == 0]
        smart_kspace[:, offset] = valid_offset_values[:, :, None, :, None].repeat(1, 1, 160, 1, 2)
    return smart_kspace


def show_init(kspace, smart_kspace):
    plt.subplot(2, 2, 1)
    plt.imshow(kspace[0, 4, :, 80, :, 0])
    plt.subplot(2, 2, 2)
    plt.imshow(kspace[0, 4, :, 80, :, 0] > 0)
    plt.subplot(2, 2, 3)
    plt.imshow(smart_kspace[0, 4, :, 80, :, 0])
    plt.subplot(2, 2, 4)
    plt.imshow(smart_kspace[0, 4, :, 80, :, 0] > 0)
    plt.show()

    plt.subplot(2, 1, 1)
    plt.imshow(smart_kspace[0, 4, :, 80, :, 0] - kspace[0, 4, :, 80, :, 0])
    plt.subplot(2, 1, 2)
    plt.imshow(kspace[0, 4, :, 80, :, 0] > 0)
    plt.show()


if __name__ == "__main__":
    np.random.seed(123)
    from fastmri.data.subsample import create_mask_for_mask_type
    mask = create_mask_for_mask_type("variabledensity3d", [0], [6])
    mask = np.stack([mask((1, 8, 92, 1))[0] for _ in range(8)], 0).squeeze()
    mask = mask[None, :, :, None, :, None]
    mask = np.repeat(np.repeat(np.repeat(mask, 15, axis=0), 160, 3), 2, -1)
    kspace = np.random.rand(15, 8, 8, 160, 92, 2)
    kspace = mask * kspace
    # kspace[kspace < 0.5] = 0
    kspace = torch.from_numpy(kspace)
    smart_kspace = smart_offset_init(kspace)
    show_init(kspace.numpy(), smart_kspace.numpy())
