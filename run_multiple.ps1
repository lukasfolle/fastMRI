conda activate fast
Set-Location C:\Users\follels\Documents\fastMRI

# 4 2 4 3 2, gaussian

#python fastmri_examples\varnet\train_varnet_4d.py --accelerations=6 --lr=1e-3 --loss="combined"  # version 12
#python fastmri_examples\varnet\train_varnet_4d.py --accelerations=6 --lr=1e-4 --loss="combined"  # version 13
#python fastmri_examples\varnet\train_varnet_4d.py --accelerations=6 --lr=1e-5 --loss="combined"  # version 14
#
#python fastmri_examples\varnet\train_varnet_4d.py --accelerations=6 --lr=1e-3 --loss="ssim"  # version 15
#python fastmri_examples\varnet\train_varnet_4d.py --accelerations=6 --lr=1e-4 --loss="ssim"  # version 16

# version 17, Gradient checkpointing 8x8x128x128, model 4 2 4 3 2 -> 300k
# version 18, Gradient checkpointing 8x8x128x128, model 8 2 4 3 2 -> 568k, combined loss
# version 19, Gradient checkpointing !!!16x8x128x128, model 4 2 4 3 2, combined loss
# version 20, Gradient checkpointing 8x8x128x128, model 4 2 4 3 2, ssim loss
# version 21, Gradient checkpointing 8x8x128x128, model 4 2 4 3 2, ssim loss, dense center sampling
# version 22, Gradient checkpointing 8xall_slicesx128x128, model 4 2 4 3 2, ssim loss, dense center sampling
# version 23, Gradient checkpointing 8x16x128x128, model 4 2 4 3 2, ssim loss, poisson dense center sampling
# version 24, Gradient checkpointing 8x16x128x128, model 4 2 4 3 2, ssim loss, poisson dense center sampling (Fixed seeds)

python fastmri_examples\varnet\train_varnet_4d.py --accelerations=6 --lr=1e-3 --loss="ssim"
# python fastmri_examples\varnet\train_varnet_4d.py --accelerations=6 --lr=1e-3 --loss="l1"  # version 18
