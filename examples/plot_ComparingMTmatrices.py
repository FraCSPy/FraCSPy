r"""
Comparing Moment Tensor Matrices
=======================================
In this example, we will perform a quick and dirty amplitude-based inversion
for the moment tensor.

As this is a plotting example, we will tackle the problem as an inverse-crime.
For a more scientific example of how to perform amplitude-based moment-tensor
inversion, look into the tutorial [XXX] which uses data generated via finite-
difference modelling.
"""
import matplotlib.pyplot as plt
import numpy as np
from pyfrac.mtsolvers.homo_mti import collect_source_angles, pwave_Greens_comp
from pyfrac.mtsolvers.mtutils import get_mt_computation_dict
from pyfrac.mtsolvers.mtai import *
from pyfrac.visualisation.momenttensor_plots import MTMatrix_comparisonplot


plt.close("all")
np.random.seed(0)

###############################################################################
# For this inverse-crime, we will use our inverse operator in the forward mode
# to generate the data we will invert.


# Velocity Model
nx, ny, nz = 47, 51, 75
dx, dy, dz = 4, 4, 4
x, y, z = np.arange(nx)*dx, np.arange(ny)*dy, np.arange(nz)*dz

vp = 1000
vel = vp * np.ones([nx, ny, nz])

# RECEIVERS
grid_rx_locs = np.linspace(dx, (dx*nx)-2*dx, 11)
grid_ry_locs = np.linspace(dy, (dy*ny)-2*dy, 13)
rx, ry, rz = np.meshgrid(grid_rx_locs,
                         grid_ry_locs,
                         dz)
recs = np.vstack((rx.flatten(), ry.flatten(), rz.flatten()))
nr = recs.shape[1]

###############################################################################
# Initialise all the necessary components for the inversion.
#
# To learn more about what these components are and their purpose, see the
# tutorial on amplitude-based moment tensor inversion: [LINK]

# Amplitude terms for inversion
gamma_sourceangles, dist_table = collect_source_angles(x,
                                                       y,
                                                       z,
                                                       reclocs=recs,
                                                       nc=3)
# Source Parameters
omega_p = 30  # Central frequency
sloc_ind = [nx//2, ny//2, nz//2]  # As indices of velocity model
MT_comp_dict = get_mt_computation_dict()  # Moment Tensor Dictionary for house-keeping purposes
# Compute the p-wave Green's functions
Gz = pwave_Greens_comp(gamma_sourceangles,
                       dist_table,
                       sloc_ind,
                       vel,
                       MT_comp_dict,
                       comp_gamma_ind=2,
                       omega_p=omega_p,
                       )

###############################################################################
# Create the forward data based on a chosen moment tensor, in this case the
# forward data is only the p-wave arrival amplitude
mt_xx = 0
mt_yy = 0
mt_zz = 0
mt_xy = -1
mt_xz = 0
mt_yz = 0
mt = [mt_xx, mt_yy, mt_zz, mt_xy, mt_xz, mt_yz]
print('MT for forward modelling: ', mt)
# Forward (Note, this will only give the p-amplitudes of the arrival
p_amps_true = frwrd_mtmodelling(Gz, mt)
# So this is not super boring let's add a tiny bit of noise
p_amps_noisy = p_amps_true + 0.2*((np.random.random(len(p_amps_true))-0.5)*np.mean(abs(p_amps_true)))


###############################################################################
# Perform inverse operation with our least-squares MT solver
mt_est = lsqr_mtsolver(Gz, p_amps_noisy)


###############################################################################
# Now we are ready to plot the comparison between our known MT (mt) and our
# estimate MT (mt_est)

MTMatrix_comparisonplot(mt, mt_est)
plt.show()

###############################################################################
# NB: Just a quick plot to show how shifted the selected arrival amplitudes are

fig = plt.figure()
plt.scatter(x=np.arange(len(p_amps_true)), y=p_amps_true, c='r', label='true')
plt.scatter(x=np.arange(len(p_amps_true)), y=p_amps_noisy, c='b', label='noisy')
plt.legend()
plt.xlabel('Receiver #')
plt.ylabel('P-Amplitude')
plt.title('P-Wave Arrival Amplitudes')