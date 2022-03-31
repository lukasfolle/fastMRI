import os
import torch
import numpy as np
import nibabel as nib
import fastmri
from fastmri.pl_modules import VarNetModule
from fastmri.data.cest_test_data import generate_test_sample


def test(save_path=r"C:\Users\follels\Documents\fastMRI\logs\varnet\outputs"):
    model = VarNetModule(
        num_cascades=6,
        pools=3,
        chans=8,
        sens_pools=3,
        sens_chans=8,
        lr=0,
        volume_training=True
    )
    model = model.load_from_checkpoint(r"C:\Users\follels\Downloads\35732_epoch_550.ckpt")
    model = model.to("cuda:0")

    kspace_us, mask_us, num_low_frequencies, reco_offsets = generate_test_sample("cuda:0")

    # First 8 offsets
    kspace_us = kspace_us[..., :8, :] * 1e2
    mask_us = mask_us[..., :8, :]
    reco_offsets = reco_offsets[..., :8]

    reco_nii = nib.Nifti1Image(reco_offsets.cpu().numpy(), affine=np.eye(4))
    nib.save(reco_nii, os.path.join(save_path, "reco_gt.nii.gz"))

    reco_offsets = []
    for offset in range(kspace_us.shape[-2]):
        reco = fastmri.ifft3c(kspace_us[..., offset, :])
        reco = fastmri.complex_abs(reco)
        reco = fastmri.rss(reco, dim=1).squeeze()
        reco_offsets.append(reco)
    reco_offsets = torch.stack(reco_offsets, -1)
    reco_nii = nib.Nifti1Image(reco_offsets.cpu().numpy(), affine=np.eye(4))
    nib.save(reco_nii, os.path.join(save_path, "reco.nii.gz"))

    print(f"K-space shape {kspace_us.shape}")
    with torch.no_grad():
        kspace_us = kspace_us.permute((0, 1, 5, 2, 3, 4, 6))
        mask_us = mask_us.permute((0, 1, 5, 2, 3, 4, 6))
        # Input: batch x channels x offsets x depth x width x height x complex
        prediction = model.forward(kspace_us.float(), mask_us.bool(), num_low_frequencies).squeeze()
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())
        prediction = torch.permute(prediction, (1, 2, 3, 0))
        prediction_nii = nib.Nifti1Image(prediction.cpu().numpy(), affine=np.eye(4))
        nib.save(prediction_nii, os.path.join(save_path, "prediction.nii.gz"))


if __name__ == "__main__":
    test()
