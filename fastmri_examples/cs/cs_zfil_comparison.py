import numpy as np
import torch
import torch.nn.functional as F
import os
from scipy.io import loadmat, savemat
import nibabel as nib
import json
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse

import fastmri
from fastmri.data.transforms import VarNetDataTransformVolume4DGrappa
from fastmri_examples.cs.hamming import HammingWindowParametrized
from fastmri.pl_modules import VarNetModule
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.mri_data import RealCESTData
import matplotlib.pyplot as plt
from utils.matplotlib_viewer import scroll_slices
from tqdm import trange
from pygrappa.mdgrappa import mdgrappa
from sigpy.mri.app import EspiritCalib


def zero_one_norm(data):
    return (data - data.min()) / (data.max() - data.min())

class Model:
    def __init__(self):
        self.model = VarNetModule(
            num_cascades=6,
            pools=3,
            chans=8,
            sens_pools=3,
            sens_chans=8,
            lr=0.001,
            lr_step_size=10000,
            lr_gamma=1,
            weight_decay=0,
            volume_training=True,
            mask_center=False,
            accelerations=6,
            loss="combined_loss_offsets",
        )
        self.model = self.model.load_from_checkpoint(r"C:\Users\follels\Documents\fastMRI\logs\varnet\varnet_demo\checkpoints\run_128_epoch=54-step=329.ckpt")
        self.model = self.model.to("cuda")
        self.model.eval()
        
    def predict(self, masked_kspace, acs, mask):
        with torch.autocast("cuda"):
            with torch.no_grad():
                masked_kspace = masked_kspace.to("cuda")
                mask = mask.to("cuda")
                acs = torch.stack((torch.real(acs), torch.imag(acs)), -1)
                acs = F.pad(acs, (0, 0, 0, 0, 52, 52))  # Select acs of first offset measurement, pad with zeros
                acs = torch.stack([acs for _ in range(masked_kspace.shape[2])], 2)
                acs = acs.to("cuda")
                preds = []
                for k in range(2):
                    pred = self.model.forward(masked_kspace[:, :, k*8:(k+1)*8], acs[:, :, k*8:(k+1)*8], mask, 0)
                    preds.append(pred)
                pred = torch.cat(preds, 1)
                pred = pred.detach().to("cpu")
                return pred

class Dataset:
    def __init__(self, mask_type="poisson_3d"):
        self.dataset = None
        self.generate_dataset(mask_type)
        
    def generate_dataset(self, mask_type):
        mask = create_mask_for_mask_type(mask_type, [0], [4])
        transform = VarNetDataTransformVolume4DGrappa(mask_func=mask, use_seed=False)
        cest_ds = RealCESTData(r"E:\Lukas\cest_data\Probanden\Mareike\output\multicoil_val", "multicoil", transform=transform, use_dataset_cache=False,
                                cache_path=r"C:\Users\follels\Documents\fastMRI\cache\dense_center\cache_val", number_of_simultaneous_offsets=8)
        self.dataset = cest_ds
    
    def get(self, i):
        item = self.dataset.__getitem__(i)
        return item


def grappa_reco(filled_kspace, do_hamming=True):
    hamming = HammingWindowParametrized(device="cpu")
    grappa_volumes = []
    for o in range(filled_kspace.shape[1]):
        kspace = torch.view_as_real(filled_kspace[:, o])
        if do_hamming:
            kspace = hamming(torch.stack((kspace, kspace), 1))[:, 0]
        volume = fastmri.ifft3c(kspace)
        volume = fastmri.complex_abs(volume)
        volume = fastmri.rss(volume, dim=0)
        volume = volume[..., 64:192]
        grappa_volume = volume.detach().numpy()
        grappa_volumes.append(grappa_volume)
    grappa_volumes = np.stack(grappa_volumes, 0)
    return grappa_volumes

def zero_filling_reco(masked_kspace):
    zfil_volumes = []
    for o in range(masked_kspace.shape[1]):
        volume = fastmri.ifft3c(torch.view_as_real(masked_kspace[:, o]))
        volume = fastmri.complex_abs(volume)
        volume = fastmri.rss(volume, dim=0)
        volume = volume[..., 64:192]
        volume = volume.numpy()
        zfil_volumes.append(volume)
    zfil_volumes = np.stack(zfil_volumes, 0)
    return zfil_volumes

# Comparison:   GRAPPA | Centrum of K-Space | Zero-filling | NN 

def main():
    results = {}
    idx = 0 * 2  # To get 16 offsets
    ds = Dataset("equispaced_fraction_dense_center_3d")
    model = Model()
    item_0 = ds.get(idx)
    item_1 = ds.get(idx + 1)
    offset = 16
    
    masked_kspace_0 = item_0.masked_kspace[..., 0] + 1j * item_0.masked_kspace[..., 1]
    masked_kspace_1 = item_1.masked_kspace[..., 0] + 1j * item_1.masked_kspace[..., 1]
    masked_kspace = torch.cat((masked_kspace_0, masked_kspace_1), 1)
    filled_kspace_0 = item_0.filled_kspace
    filled_kspace_1 = item_1.filled_kspace
    filled_kspace = torch.cat((filled_kspace_0, filled_kspace_1), 1)
    mask = item_0.mask
    acs = item_0.acs
    target_0 = item_0.target
    target_1 = item_1.target
    target = torch.cat((target_0, target_1), 0)
    target = target.numpy()
    target_cest = target / target[0][None]
    print(f"Kspace shape: ({masked_kspace.shape})")
    accel = torch.numel(masked_kspace) / (masked_kspace != 0).sum()
    print(f"Acceleration: {accel}")
        
    # GRAPPA
    grappa_volumes = grappa_reco(filled_kspace[..., 0] + 1j * filled_kspace[..., 1])
    grappa_cest = grappa_volumes / grappa_volumes[0][None]
    results["GRAPPA"] = {"volume": grappa_volumes, "cest": grappa_cest}
    
    # Zero-filling
    zero_filling_volumes = zero_filling_reco(masked_kspace)
    zero_filling_cest = zero_filling_volumes / zero_filling_volumes[0][None]
    results["ZFIL"] = {"volume": zero_filling_volumes, "cest": zero_filling_cest}
    
    # Center of kspace reco
    base = r"E:\Lukas\cest_data\Probanden\Mareike\output\8x_center_kspace"
    files = [os.path.join(base, f) for f in os.listdir(base)]
    center_of_kspace_files = files[idx]
    image = nib.load(center_of_kspace_files)
    kspace_center_volume = np.swapaxes(np.swapaxes(image.get_fdata(), -1, 0), -1, 1)
    kspace_center_volume = np.rot90(kspace_center_volume, -1, (2, 3))
    kspace_center_volume = kspace_center_volume.astype("float32")
    kspace_center_cest = kspace_center_volume / kspace_center_volume[0][None]
    results["CENTERK"] = {"volume": kspace_center_volume, "cest": kspace_center_cest}
    
    # Model reco
    prediction_volume = model.predict(filled_kspace[None], item_0.acs[None], item_0.mask[None])[0]  # Remove batch dimension
    prediction_volume = prediction_volume.numpy()
    prediction_volume = prediction_volume[..., 64:192]
    prediction_cest = prediction_volume / prediction_volume[0][None]
    results["NN"] = {"volume": prediction_volume, "cest": prediction_cest}
    
    # CS reco
    # Do this in Matlab
    # img_tv = loadmat(r"W:\radiologie\data\MR-Physik\Mitarbeiter\Tkotz\tools_Lukas\Test\exchange_recon")["volume_offsets"]
    # img_tv = np.abs(np.transpose(img_tv, (2, 0, 1, 3))).transpose(3, 0, 1, 2)[:offset]
    
    # Norm Should this be done?
    # target = zero_one_norm(target)
    # volume = zero_one_norm(volume)
    # img_tv = zero_one_norm(img_tv)
    # prediction = zero_one_norm(prediction)
    # grappa_volume = zero_one_norm(grappa_volume)

    # Metrics
    metrics = dict.fromkeys(results.keys())
    for method in results.keys():
        pred = results[method]["volume"]
        metrics[method] = {"offset": [], "psnr": [], "nrmse": [], "ssim": []}
        for o in range(offset):
            print(f"\n{method} Offset ({o}):")
            psnr = peak_signal_noise_ratio(target[o], pred[o])
            print(f"PSNR {psnr}")
            nrmse = normalized_root_mse(target[o], pred[o])
            print(f"NRMSE {nrmse}")
            ssim = structural_similarity(target[o], pred[o], win_size=3)
            print(f"SSIM {ssim}")
            metrics[method]["offset"].append(o)
            metrics[method]["psnr"].append(psnr)
            metrics[method]["nrmse"].append(nrmse)
            metrics[method]["ssim"].append(ssim)
    with open('metrics_comparison.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Visualize
    view_offset = 1
    
    plt.figure(figsize=(20, 5))
    plt.subplot(1,5,1)
    plt.title("Target")
    plt.imshow(target[view_offset, 10], cmap="gray")
    
    for k, name in enumerate(results.keys()):
        plt.subplot(1,5,k + 2)
        plt.title(name)
        plt.imshow(results[name]["volume"][view_offset, 10], cmap="gray")
    plt.savefig("Comparison.jpg", dpi=400)
    
    plt.figure(figsize=(20, 5))
    plt.subplot(1,5,1)
    plt.title("Target")
    plt.imshow(target_cest[view_offset, 10], cmap="gray")
    
    for k, name in enumerate(results.keys()):
        plt.subplot(1,5,k + 2)
        plt.title(name)
        plt.imshow(results[name]["cest"][view_offset, 10], cmap="gray")
    plt.savefig("Comparison_CEST.jpg", dpi=400)
    
    print()


if __name__ == "__main__":
    main()


# Zero-filling:
# PSNR 21.243830733880095
# NRMSE 0.31921095178575815
# SSIM 0.7419604616536802

# CS:
# PSNR 19.98388693846141
# NRMSE 0.36904130120424544
# SSIM 0.7800721197718142

# Grappa:
# PSNR 19.97822011085701
# NRMSE 0.36928214881649296
# SSIM 0.7045333140277948


