import os
import torch
import nibabel as nib
import numpy as np

import fastmri
from fastmri.pl_modules import VarNetModule
from fastmri.data.cest_test_data import generate_test_sample


def test(save_path=r"C:\Users\follels\Documents\fastMRI\logs\varnet\outputs"):
    model = VarNetModule(
        num_cascades=4,
        pools=2,
        chans=4,
        sens_pools=3,
        sens_chans=2,
        lr=0,
        volume_training=True
    )
    model = model.load_from_checkpoint(r"C:\Users\follels\Documents\fastMRI\logs\varnet\varnet_demo\checkpoints\epoch=122-step=18449.ckpt")

    kspace, mask, num_low_frequencies = generate_test_sample()

    reco = fastmri.ifft3c_new_offsets(kspace)
    reco = fastmri.complex_abs(reco)
    reco = fastmri.rss(reco, dim=1).squeeze()
    reco = torch.permute(reco, (2, 3, 1, 0))
    reco_nii = nib.Nifti1Image(reco.numpy(), affine=np.eye(4))
    nib.save(reco_nii, os.path.join(save_path, "reco.nii.gz"))

    print(f"K-space shape {kspace.shape}")
    with torch.no_grad():
        prediction = model.forward(kspace, mask, num_low_frequencies).squeeze()
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())
        prediction = torch.permute(prediction, (2, 3, 1, 0))
        prediction_nii = nib.Nifti1Image(prediction.numpy(), affine=np.eye(4))
        nib.save(prediction_nii, os.path.join(save_path, "prediction.nii.gz"))


if __name__ == "__main__":
    test()
