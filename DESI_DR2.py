import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

# === Load BAO data from file ===
data_file = 'data/desi_gaussian_bao_ALL_GCcomb_mean.txt'
data = np.genfromtxt(data_file, dtype=None, encoding=None)

dmz, dmdata = [], []
dhz, dhdata = [], []
dvz, dvdata = [], []

for z, val, qtype in data:
    if qtype == 'DM_over_rs':
        dmz.append(z)
        dmdata.append(val)
    elif qtype == 'DH_over_rs':
        dhz.append(z)
        dhdata.append(val)
    elif qtype == 'DV_over_rs':
        dvz.append(z)
        dvdata.append(val)

dmz, dmdata = np.array(dmz), np.array(dmdata)
dhz, dhdata = np.array(dhz), np.array(dhdata)
dvz, dvdata = np.array(dvz), np.array(dvdata)

# === Load covariance matrix ===
cov_matrix_file = 'data/desi_gaussian_bao_ALL_GCcomb_cov.txt'
cov_matrix = np.loadtxt(cov_matrix_file)
cov_matrix_inv = np.linalg.inv(cov_matrix)

# === Stack data in the same order as covariance matrix ===
# Order: DV(z1), DM(z2), DH(z2), DM(z3), DH(z3), ..., DM(z7), DH(z7)
combined_data = np.array([
    dvdata[0],        # z = 0.295
    dmdata[0], dhdata[0],  # z = 0.51
    dmdata[1], dhdata[1],  # z = 0.706
    dmdata[2], dhdata[2],  # z = 0.934
    dmdata[3], dhdata[3],  # z = 1.321
    dmdata[4], dhdata[4],  # z = 1.484
    dmdata[5], dhdata[5]   # z = 2.33
])

# === Likelihood function ===
def desidr2_likelihood(z_grid, integral_grid, H_val, params):
     
    Omega0 , Sigma0 , H0 , M , rd = params

    c = 2.99792458e5  # speed of light in km/s

    # Interpolated comoving distance: DM = c/H0 * integral
    integral_dm = np.interp(dmz, z_grid, integral_grid)
    integral_dv = np.interp(dvz, z_grid, integral_grid)

    dl_dm = (c / H0) * integral_dm
    dl_dv = (c / H0) * integral_dv

    # Compute observables
    dmrdval = dl_dm / rd
    dhrdval = c / (rd * H0 * H_val(dhz))
    dvdr_val = ((dvz * dl_dv**2 * c) / (H_val(dvz) * H0))**(1/3) / rd

    # Stack model predictions in matching order

    combined_model = np.array([
        dvdr_val[0],        # z = 0.295
        dmrdval[0], dhrdval[0],  # z = 0.51
        dmrdval[1], dhrdval[1],  # z = 0.706
        dmrdval[2], dhrdval[2],  # z = 0.934
        dmrdval[3], dhrdval[3],  # z = 1.321
        dmrdval[4], dhrdval[4],  # z = 1.484
        dmrdval[5], dhrdval[5]   # z = 2.33
    ])
    
    # Residuals and chi-squared
    residuals = combined_model - combined_data
    chi2 = -0.5 * residuals.T @ cov_matrix_inv @ residuals

    return chi2
