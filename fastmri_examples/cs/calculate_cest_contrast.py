import os
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_cest(path, mean=None, std=None):
    volumes = nib.load(path).get_fdata()
    volumes_cest = volumes / (volumes[..., 0][..., None] + 1e-10)
    volumes_cest = volumes_cest[..., 1:]  # Remove first offset
    return volumes_cest

target_cest = get_cest(r"E:\Lukas\cest_data\Probanden\Mareike\prediction\val_target_0.nii.gz")
center_kspace_cest = get_cest(r"E:\Lukas\cest_data\Probanden\Mareike\prediction\center_kspace_reco_0.nii.gz")
center_kspace_cest = torch.swapaxes(F.interpolate(torch.swapaxes(torch.from_numpy(center_kspace_cest).unsqueeze(0), 1, -1), scale_factor=2, mode="nearest"), 1, -1).numpy().squeeze()

ref_cest = get_cest(r"W:\radiologie\data\MR-Physik\Mitarbeiter\Tkotz\tools_Lukas\Reference_0806\val_default_reco_0val_default_reco_1.nii")

# Good CEST constrast, less fine details
pred_cest = get_cest(r"E:\Lukas\cest_data\Probanden\Mareike\prediction\version_185_val_prediction_0_epoch_1900.nii.gz")

# Good resolution of fine details, less good fit of CEST curve
# pred_cest = get_cest(r"E:\Lukas\cest_data\Probanden\Mareike\prediction\version_184_val_prediction_0_epoch_190.nii.gz")

pred_cest_center_k = get_cest(r"E:\Lukas\cest_data\Probanden\Mareike\prediction\version_190_val_prediction_0_epoch_1990.nii.gz")

# pred_cest_center_outside = get_cest(r"E:\Lukas\cest_data\Probanden\Mareike\prediction\version_194_val_prediction_0_epoch_1280.nii.gz")

pred_cest_1d = get_cest(r"E:\Lukas\cest_data\Probanden\Mareike\prediction\version_197_val_prediction_0_epoch_1260.nii.gz")


fig = plt.figure(figsize=(10, 6))

volumes = [target_cest, center_kspace_cest, ref_cest, pred_cest, pred_cest_center_k, pred_cest_1d]  # pred_cest_center_outside
names = ["Target", "Low Res", "GRAPPA", "185 NN (GRAPPA)", "190 NN (Center kspace)", "197 NN (1D)"]  # "194 NN (GRAPPA Center fully)"
colors = ["r", "m", "g", "b", "c", "olive"]  # "y"

for k in range(len(volumes)):
    ax = fig.add_subplot(2, round(len(volumes) / 2), k + 1)
    plt.title(names[k])
    plt.axis("off")
    ax.imshow(volumes[k][10, ..., 4], vmin=target_cest[10, ..., 4].min(), vmax=target_cest[10, ..., 4].max())
    rect = patches.Rectangle((60, 25), 10, 10, linewidth=2, edgecolor=colors[k], facecolor='none')
    ax.add_patch(rect)
# plt.show(block=False)
plt.savefig("Slice_comparison.svg")

plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
for k in range(len(volumes)):
    plt.plot(volumes[k][8:12, 25:35, 60:70].reshape(-1, volumes[k].shape[-1]).mean(0), label=names[k], color=colors[k])
plt.legend()

plt.subplot(1,2,2)
target = target_cest[8:12, 25:35, 60:70].reshape(-1, volumes[k].shape[-1]).mean(0)
plt.plot(np.zeros_like(target), "--", label="Baseline", color="gray")
for k in range(1, len(volumes)):
    plt.plot(target - volumes[k][8:12, 25:35, 60:70].reshape(-1, volumes[k].shape[-1]).mean(0), "--", label=f"Error: {names[k]}", color=colors[k])
plt.legend()
# plt.show()
plt.savefig("CEST_curve_comparison.svg")

print()

# Goal: Resolve smaller structures in CEST scans -> Overall similarity (MAE) with target is not necessarily the only goal, but sharpness of small details is important
# How do we "prove" this?
#       - SSIM is better, CEST constrast still present
#       - Segmentation of small structure -> Compare MAE/SSIM over only this structure
