# IBC-FDR-Thresholding
# Finds the FDR-corrected (two-tailed, q < 0.05) Z-thresholds for the
# IBC RSVP Language Z-maps (Sentence vs Jabberwocky) to use in MRIcroGL
# Data: Individual Brain Charting (IBC), Pinho et al. (2018)

# Step 1: Import all required functions from the library
import numpy as np                                      # works with many numbers at once
import nibabel as nib                                   # opens NIfTI files (.nii or .nii.gz)
from scipy.stats import norm                            # converts Z-values to p-values
from statsmodels.stats.multitest import fdrcorrection   # FDR correction

# Step 2: Load the Z-map (this is the only line to change for each subject)
z_map = nib.load(r'zmap path').get_fdata()   # Change this to your Z-map file path

# Keep only the brain voxels
# The IBC Z-maps are in MNI space and voxels outside the brain are exactly 0,
# so removing the zeros keeps only the voxels inside the brain
# np.isfinite() removes NaN and infinity values that would crash the FDR correction
z_scores = z_map[(z_map != 0) & np.isfinite(z_map)]

# Step 3: Convert ALL Z-values to two-tailed p-values
# Two-tailed = is this voxel different from zero in EITHER direction?
# norm.sf() gives one tail only, so I multiply it by 2
p_values = 2 * norm.sf(np.abs(z_scores))

# Step 4: Apply ONE FDR correction at q < 0.05 to all the brain voxels
# rejected = True/False array showing which voxels survived
rejected, _ = fdrcorrection(p_values, alpha=0.05)

# Step 5: Find the Z-threshold for each direction

# POSITIVE SIDE: Sentence > Jabberwocky
pos = z_scores[rejected & (z_scores > 0)]
if pos.size > 0:
    # Min = weakest effect that survived (the threshold), Max = strongest effect
    print(f"Sentence > Jabberwocky: Min = {pos.min():.4f}, Max = {pos.max():.4f}")
else:
    print("No positive voxels survive FDR q < 0.05")

# NEGATIVE SIDE: Jabberwocky > Sentence
neg = z_scores[rejected & (z_scores < 0)]
if neg.size > 0:
    # For negative values the threshold is the one closest to zero (neg.max())
    print(f"Jabberwocky > Sentence: Min = {neg.min():.4f}, Max = {neg.max():.4f}")
else:
    print("No negative voxels survive FDR q < 0.05")
