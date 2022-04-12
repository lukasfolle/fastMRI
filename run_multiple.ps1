conda activate fast
Set-Location C:\Users\follels\Documents\fastMRI

python fastmri_examples\varnet\train_varnet_4d.py --loss="combined_loss_offsets" --number_of_simultaneous_offsets=1
python fastmri_examples\varnet\train_varnet_4d.py --loss="combined_loss_offsets" --number_of_simultaneous_offsets=2
python fastmri_examples\varnet\train_varnet_4d.py --loss="combined_loss_offsets" --number_of_simultaneous_offsets=3
python fastmri_examples\varnet\train_varnet_4d.py --loss="combined_loss_offsets" --number_of_simultaneous_offsets=4
python fastmri_examples\varnet\train_varnet_4d.py --loss="combined_loss_offsets" --number_of_simultaneous_offsets=5
python fastmri_examples\varnet\train_varnet_4d.py --loss="combined_loss_offsets" --number_of_simultaneous_offsets=6
python fastmri_examples\varnet\train_varnet_4d.py --loss="combined_loss_offsets" --number_of_simultaneous_offsets=7
python fastmri_examples\varnet\train_varnet_4d.py --loss="combined_loss_offsets" --number_of_simultaneous_offsets=8

