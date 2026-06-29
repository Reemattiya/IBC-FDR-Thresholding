# Step 1:Import all required functions from the library 

# numpy helps Python work with many numbers at once, like handling 
# all the voxel Z-values from the brain map
import numpy as np

# NiBabel is used to open neuroimaging files like NIfTI (.nii or .nii.gz)
import nibabel as nib

# import normal distribution from scipy.stats to convert Z-values to p-values
from scipy.stats import norm

# This imports the FDR correction function from Statsmodels
from statsmodels.stats.multitest import fdrcorrection

##################################################################

# Step 2: Loading the NIfTI file and extracting the Z-values from it
# Also includes flattening and filtering the Z-values to prepare for FDR correction

# Loading the NIfTI file
img = nib.load(r'zmap path')  # Change this to your actual file path

# get_fdata() converts the loaded NIfTI Z-map into numerical voxel data
# so that Python can analyse the Z-values and calculate the FDR threshold
voxel_values = img.get_fdata()

# This will tell you the 3D dimensions of the image (e.g. 96, 114, 96)
# confirming the file loaded correctly before doing anything else with it
print("Image shape:", voxel_values.shape)

# np.isfinite() removes NaN and infinity values that can sometimes
# appear in NIfTI files and would crash the FDR correction
z_scores = voxel_values[np.isfinite(voxel_values)]

# Flatten the 3D array of Z-values into a 1D array
# to make it easier to work with statistically
z_scores = z_scores.flatten()
print("After flattening:", z_scores.shape)

# Filtering: keep only the values that are not zero
# Background voxels outside the brain have a value of exactly 0
# Including them would distort the FDR threshold
z_scores = z_scores[z_scores != 0]
print("After filtering:", z_scores.shape)

##################################################################

# ---------- POSITIVE SIDE: Sentence > Jabberwocky ----------

# Step 3a: Keep only positive Z-values
# Positive Z-values represent regions more active during
# meaningful sentences than Jabberwocky (Sentence > Jabberwocky)
pos_z = z_scores[z_scores > 0]

# Step 4a: Convert positive Z-values to one-tailed p-values
# norm.sf() is the survival function — it gives the probability
# of seeing a Z-value this extreme or larger in one tail
# We use one-tailed here because we already separated the directions
pos_p = norm.sf(pos_z)
print("Min positive p-value:", pos_p.min())
print("Max positive p-value:", pos_p.max())

# Step 5a: Apply FDR correction at q < 0.05 to positive voxels
# fdrcorrection() returns two things:
# pos_rejected — True/False array showing which voxels survived
# pos_p_corrected — the adjusted p-values after correction
pos_rejected, pos_p_corrected = fdrcorrection(pos_p, alpha=0.05)
print("Number of positive voxels surviving FDR:", pos_rejected.sum())

# Step 6a: Find the Z-threshold for the positive direction
if np.any(pos_rejected):
    # Minimum positive Z = weakest effect that still survived FDR correction
    # This is the value you enter as minimum in MRIcroGL
    pos_threshold_min = np.min(pos_z[pos_rejected])
    # Maximum positive Z = strongest effect in the surviving voxels
    # This is the value you enter as maximum in MRIcroGL
    pos_threshold_max = np.max(pos_z[pos_rejected])
    print("\nPositive direction: Sentence > Jabberwocky")
    print("FDR q < 0.05 Z-threshold =", round(pos_threshold_min, 4))
    print("Use in MRIcroGL: Min =", round(pos_threshold_min, 4),
          "Max =", round(pos_threshold_max, 4))
else:
    print("No positive voxels survive FDR q < 0.05")
