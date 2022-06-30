conda activate fast
Set-Location C:\Users\follels\Documents\fastMRI

python fastmri_examples\varnet\train_unet_4d.py --loss="combined_loss_offsets" --chans=8 --num_pool_layers=2 --lr=1e-3
python fastmri_examples\varnet\train_unet_4d.py --loss="combined_loss_offsets" --chans=16 --num_pool_layers=2 --lr=1e-3
python fastmri_examples\varnet\train_unet_4d.py --loss="combined_loss_offsets" --chans=64 --num_pool_layers=2 --lr=1e-3

python fastmri_examples\varnet\train_unet_4d.py --loss="combined_loss_offsets" --chans=32 --num_pool_layers=1 --lr=1e-3
