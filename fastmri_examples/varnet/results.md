# TODO: Try out pretrained network as init
# Run 30: 3 training cases
# Run 32: 1 training case
# Run 33: Overfit one sample
# Test less downsampling influence
# Run 55: +tv, denser outer sampling
# Run 56: dc_weight 1 -> 0.5
# Run 57: 6 3 8 3 8 (7.2M) -> 4 3 8 3 8 (5.2M)
# Run 58: US 4 6 -> 3 5
# Run 59: US 4 6 -> 5 7
# Run 60: US 4 6 -> 2 4
# Run 61/62: US 4 6 -> 1 1
# Run 63: US 4 6 -> 2 4, mse loss
# Run 64: US 4 6 -> 3 5, mse loss
# Run 71: US 3 5, mse loss, all offsets (2x8) --> Prediction an Kathi schicken
# Run 72: US 3 5, mse + ssim loss, all offsets (2x8) (better than mse alone)
# Run 73: US 3 5, mse(l2) + ssim loss, all offsets (2x8)
# Run 74: US 3 5, ssim loss, all offsets (2x8) (best so far)
# Run 75: US 3 5, ssim loss, all offsets (2x8), 4 3 8 3 8 (5.2M)
# Run 76: US 3 5, ssim loss, all offsets (2x8), 2 3 8 3 8 (3.1M)
# Run 77: US 3 5, ssim loss, all offsets (2x8), 8 3 8 3 8 (9.3M)
# Run 78: US 3 5, ssim loss, all offsets (2x8), dense center sampling (better than 74), sampling closer to center more important
# Changed to offset-wise training, US 2 6 (7.2M & 45GB -> 2.5M & 20GB)
# Run 79: US 2 6, ssim loss, single offsets
# Run 80: US 2 6, mse loss, single offset
# Back to (almost) all offset training
# Run 81: US 2 6, ssim loss (2x8 offsets) (best so far)

# Run 82: US 2 6 , Model 6 2 8 2 8 (1.8M), 4 offsets
# Run 83: US 2 6 , Model 6 2 8 2 8, 8 offsets
# Run 84: US 2 6 , Model 6 3 8 3 8, 8 offsets (best so far)
# Run 85: US 2 6 , Model 6 3 8 3 8, 8 offsets (pretrained fastMRI)
# Run 86: US 2 6 , Model 6 3 8 3 8, 8 offsets, ssim loss, no kspace imputation (considerably worse than 84)

# Optimizer channels for 4 offsets
# Run 87: US 26, Model 6 1 8 2 8 (274K), 4 offsets
# Run 88: US 26, Model 6 2 8 2 8, 4 offsets 
# Run 89: US 26, Model 6 4 8 2 8, 4 offsets 
# Run 90: US 26, Model 6 8 8 2 8, 4 offsets 
# Run 91: US 26, Model 6 12 8 2 8, 4 offsets 
# Run 92: US 26, Model 6 16 8 2 8 (6.2M), 4 offsets
# 93:   No normalization, 4 offsets
# 94:   Zero-mean-unit-variance norm, 4 offsets
# 95:   Bigger model, 8 offsets, patch-wise
# 96:   Bigger model, 8 offsets
# 97:   Bigger model, 8 offsets, ssim loss
# Run 98: replicate Run 84
# Run 99: GRAPPA Init


# 100:    KSpace init with Grappa, Unet as denoiser
# 101:    KSpace init with Grappa, Unet as denoiser, mse loss
# 102/3:  KSpace init with Grappa, Hamming Window as denoiser, ssim loss
# 104:    KSpace init with Grappa, Hamming Window as denoiser + hamming init, ssim loss

# Run 105: Grappa init and hamming window layer last
# Run 106: GRAPPA init with alternating masks over offsets
# Run 107: GRAPPA init alternating, R=9 (3x3), removes some artifacts
# Run 108: GRAPPA init alternating, R=9 (3x3), Hamming parametrized as last layer
# Run 109: GRAPPA init alternating, R=9 (3x3), No VarNet, only Hamming parametrized as last layer
# Run 110: Pure VarNet, Poisson undersampling factor 8.6
# Run 111 Try Grappa init for poisson undersampling -> Error: Singular matrix

# Run 112: GRAPPA init alternating, R=16 (4x4)
# Run 113: No Grappa, imputed kspace as input to sens network, masked kspace input to cascades
# Grappa baseline not comparable so far due to using center kspace for all offsets individually, now switched to first offset.
# -> But doesnt change much ...
# Run 114: Grappa Init, 3x3, only first offset as acs 
# Previous grappa results were wrong since center of kspace was kept instead of acs region.

# Run 115 masked_kspace input, Corrected Grappa baseline, acs input to sensmodel, 3x3 
# Run 116 masked_kspace input, Corrected Grappa baseline, acs input to sensmodel, 3x3, ssim + mse
# Run 117 filled_kspace input, acs to SenseNet, ssim + mse 
# Run 118 filled_kspace input, acs to SenseNet, ssim (**Better than GRAPPA, NRMSE - 1, SSIM + 1.4, PSNR + 0.8**)
# Run 119 filled_kspace input, acs to SenseNet, ssim, reduces lr to 0.0001

# Run 120 GRAPPA Init, Esprit sens instead of SensNet, 0.0001 lr (6.2M)
# Run 121 No Init, Esprit sens instead of SensNet
# Run 122 Kspace imputation, Esprit sens instead of SensNet

# Run 123 GRAPPA Init, ACS to SenseNet, Denser center sampling (x8)
# Run 124 Like 123, w/ hamming window parametrized (HWP)
# Run 126 Like 123, w/ hamming window parametrized (prev cascade) (**Best so far** SSIM +3%, PSNR +0.1%, NRMSE +1%)
# Run 127 Like 123, w/ hamming window parametrized (prev cascade), mse
# Run 128 Continue 126

# Run 129: No HWP, added U-Net after to_image_space_transform
# Run 130: No HWP, added U-Net after to_image_space_transform, mse
# Run 131: VarNet3D1D, no image_space_net, 3.1M

# Run 132: VarNet3D1D, poisson sampling
# Run 133: Continue 132 /w mae loss
# Run 134: Use fastmri checkpoint -> train further

# Run 135: Image space net only
# Run 136: Image space net only, mse
# Run 137: Image space net only, combined

# Run 142: VarNet3D1D /w hamming window (compare with 131 & 126)
# Run 143: VarNet3D1D /w hamming window, modified conv (3.8M) (1D Conv along offsets after each 3D Conv)
# Run 144: VarNet3D1D, modified conv (3.8M) (1D Conv along offsets after each 3D Conv)
# Run 145: VarNet3D1D, wnet (3.1M)
# Run 146: VarNet3D1D, wnet + ImageNet(unit init)
# Run 147: VarNet3D1D, wnet + ImageNet residual (unit init)

# Run 148: Shifted cartesian sampling over offsets, VarNet3D1D (grappa weights individually for each offset), filled_kspace
# Run 149: Shifted cartesian sampling over offsets, VarNet3D1D (grappa weights individually for each offset), masked_kspace_imputed (better than 150 & 148)
# Run 150 Shifted cartesian sampling over offsets, VarNet3D1D (grappa weights individually for each offset), masked_kspace

# Run 151: Continue 149

# Run 152 Corrected ACS, shifted cartesian sampling over offsets, filled kspace as input, compare with 148, 149 and 150
# Run 153 Corrected ACS, shifted cartesian sampling over offsets, filled kspace as input, combined loss (mse + ssim)
# Run 154: Hamming window after first varnet block, ssim loss
# Run 155: Ssim + tv loss
# Run 156: 1D Unet only



# Run 158 + 160/165: Imagespace 3D Unet only (***best so far 90 SSIM***, SSIM +6%, PSNR +1%, NRMSE -0.001)
# Run 161: RandSpatialCrops 8 8 64 64 N=8
# Run 162: RandSpatialCrops 8 8 92 92 N=2
# Run 163 Imagespace 3D Unet only, ssim + mse

# Runs 164 - 166: LR 1e-2, 1e-3, 1e-4
# Runs 167 - 170: 8, 16, 64 channels and 1 pool

# Run 171: Kspace NormUnet, loss in image space, /w data consistency
# Run 172: 4D Unet only image space
# Run 173 & 174 & 175 & 176: Like 165, ReduceLRonPlateou
# Run 177: No scheduler, no shifts over us mask over offsets (changed cache path), cp w/ 165
# Run 178: Change back to shift us mask over offsets, removed data *= 100 in mri_data, all 16 offsets
# Run 179: SSIM + 0.1 MSE loss to retain offset-wise characteristics
# Run 180: SSIM + 0.05 MAE loss to retain offset-wise characteristics
# Run 181: As 178, normalize data intensity range (**Best so far**)
# Run 182: Like 181, offset-wise mae loss
# Run 183: Only MAE loss, version_183_val_prediction_0_epoch_120 (**very good CEST contrast in muscle**)
# Run 184: MAE + 0.1 * SSIM loss
# Run 185: Like 183, MAE loss for 2000 epochs (** Close to perfect CEST contrast **)
# Run 186: Like 184, MAE + 0.1 SSIM
# Run 187: SSIM loss (kernel size 5 -> 11)
# Run 188: Like 184, MSE
# Run 189: Like 184, MAE non-offset-wise
# Run 190: Changed to CenterKspaceCEST (data_module & fastmri_dirs.yaml), MAE loss
# Run 191: Changed to CenterKspaceCEST (data_module & fastmri_dirs.yaml), MAE loss + 0.1 SSIM (does not really work well since HF structures cant be resolved)
# Run 192: Back to RealCEST Dataset with cartesian US, MAE + 1e-3 * SSIM loss (equal weights)
# Run 193: Back to RealCEST Dataset with cartesian US, w * MAE + (1-w) * SSIM loss (parametrized)
# Run 194: Center fully sampled, MAE loss
# Run 195: Same US for each offset, since GRAPPA otherwise makes the learning of the NN harder, otherwise same as 185

# Run 196: Offset-only Unet (compare w/ 195) if better -> run with offset-wise different US
# Run 197: Rerun 196 with offset different US (!!!best so far!!!)
# Run 198: Like 197, 3 -> 4 pool layers