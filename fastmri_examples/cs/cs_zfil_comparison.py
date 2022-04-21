import numpy as np
import torch
from scipy.io import loadmat
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse

import fastmri
from fastmri.data.transforms import VarNetDataTransformVolume4D
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.mri_data import RealCESTData
import matplotlib.pyplot as plt
from utils.matplotlib_viewer import scroll_slices
from tqdm import trange


def zero_one_norm(data):
    return (data - data.min()) / (data.max() - data.min())

class Dataset:
    def __init__(self):
        self.dataset = None
        self.generate_dataset()
        
    def generate_dataset(self):
        mask = create_mask_for_mask_type("poisson_3d", [0], [6])
        transform = VarNetDataTransformVolume4D(mask_func=mask, use_seed=True)
        cest_ds = RealCESTData(r"E:\Lukas\cest_data\Probanden\Mareike\output\multicoil_train", "multicoil", transform=transform, use_dataset_cache=False,
                                cache_path=r"C:\Users\follels\Documents\fastMRI\cache\cache_val")
        self.dataset = cest_ds
    
    def get(self, i):
        item = self.dataset.__getitem__(i)
        return item

def main():
    ds = Dataset()
    item = ds.get(0)
    
    # Select first offset
    masked_kspace = item.masked_kspace[:, 0, ..., 0] + 1j * item.masked_kspace[:, 0, ..., 1]
    masked_kspace_mps = (item.masked_kspace[..., 0] + 1j * item.masked_kspace[..., 1]).sum(1)
    mask = item.mask[:, 0]
    target = item.target[0]
    target = target.numpy()
    print(f"Kspace shape: ({masked_kspace.shape})")
    
    # Zero-filling reco
    volume = fastmri.ifft3c(torch.view_as_real(masked_kspace))
    volume = fastmri.complex_abs(volume)
    volume = fastmri.rss(volume, dim=0)
    volume = volume[..., 64:192]
    volume = volume.numpy()
    
    # CS reco
    img_tv = loadmat(r"W:\radiologie\data\MR-Physik\Mitarbeiter\Tkotz\tools_Lukas\Test\cs_reco_offsets")["volume_offsets"]
    img_tv = np.abs(np.transpose(img_tv, (2, 0, 1, 3))[..., 0])
    
    # Norm
    target = zero_one_norm(target)
    volume = zero_one_norm(volume)
    img_tv = zero_one_norm(img_tv)

    # Metrics
    for method, pred in zip(["Zero-filling", "CS"], [volume, img_tv]):
        print(f"\n{method}:")
        print(f"PSNR {peak_signal_noise_ratio(target, pred)}")
        print(f"NRMSE", normalized_root_mse(target, pred))
        print(f"SSIM", structural_similarity(target, pred, win_size=3))
    
    # Visualize
    plt.subplot(1,3,1)
    plt.title("Target")
    plt.imshow(target[10], cmap="gray")
    plt.subplot(1,3,2)
    plt.title("Zero-filling")
    plt.imshow(volume[10], cmap="gray")
    plt.subplot(1,3,3)
    plt.title("Total variation")
    plt.imshow(img_tv[10], cmap="gray")
    plt.show()
    print()


if __name__ == "__main__":
    main()

