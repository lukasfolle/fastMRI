conda activate fast
Set-Location C:\Users\follels\Documents\fastMRI

python fastmri_examples\varnet\train_varnet_4d.py --loss="combined_loss_offsets" --max_epochs=30 --number_of_simultaneous_offsets=4 --chans=16
