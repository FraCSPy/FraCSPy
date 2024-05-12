r"""
Amplitude-based Moment Tensor Inversion
========================================================
This tutorial illustrates how we can determine the moment tensor of a microseismic source based
on the amplitude of the P-wave arrivals, assuming a known source location.


Theory
--------

Microseismic data can be considered as:
$d=GM
Using the Greens functions as defined by Aki and Richards, the above can be expanded out as

.. math::
    v_i^\theta = j \omega_\theta ( \frac{\gamma_i\gamma_p\gamma_q}{4\pi\rho\alpha^3}  \frac{1}{r} )M_{pq}

where

        :math:`v` is the velocity measurements (seismic data) at the arrival of the wave, in other words the P- or S-wave peak amplitudes

        :math:`M` is the moment tensor

        :math:`\theta` describes whether we are utilising the P- or S-wave information

        :math:`i` describes the component of the data, aligning with the below p,q definitions

        :math:`p` describes the first index of the moment tensor element

        :math:`q` describes the second index of the moment tensor element

        :math:`\omega_\theta` is the peak frequency of the given wave

        :math:`\gamma_i` is the take-off angle in the ith direction (for a ray between the source and receiver)

        :math:`\alpha` is the average velocity (currently we assume a homogeneous velocity)

        :math:`r` is the distance between source and receiver
        
        :math:`\rho` is the average density

For solving with only vertical component data, utilising the P-wave arrival, the above equation can be expanded out as

.. math::
    v_i^P = j \omega_P ( \frac{\gamma_1\gamma_1\gamma_1}{4\pi\rho\alpha^3}  \frac{1}{r} )M_{1,1} +
    j \omega_P ( \frac{\gamma_1\gamma_2\gamma_2}{4\pi\rho\alpha^3}  \frac{1}{r} )M_{2,2} +
    j \omega_P ( \frac{\gamma_1\gamma_3\gamma_3}{4\pi\rho\alpha^3}  \frac{1}{r} )M_{3,3} + \\
    2*j \omega_P ( \frac{\gamma_1\gamma_1\gamma_2}{4\pi\rho\alpha^3}  \frac{1}{r} )M_{1,2} +
    2*j \omega_P ( \frac{\gamma_1\gamma_1\gamma_3}{4\pi\rho\alpha^3}  \frac{1}{r} )M_{1,3} +
    2*j \omega_P ( \frac{\gamma_1\gamma_2\gamma_3}{4\pi\rho\alpha^3}  \frac{1}{r} )M_{2,3}

Here, it is important to recall that the moment tensor matrix is symmetric and can therefore be represented
by only 6 elements. This results in the multiplication by two of the off-diagonal elements in the equation
above. For more information on the creation of the Greens equations, refer to the source code documentation
[LINK TO HOMOMTI DOCS]


"""



import numpy as np
import os
import matplotlib.pyplot as plt
from pylops.utils import dottest
from pylops.utils.wavelets import *

from pyfrac.utils.sofiutils import read_seis
from pyfrac.mtsolvers.homo_mti import collect_source_angles, pwave_Greens_comp
from pyfrac.mtsolvers.mtutils import get_mt_computation_dict
from pyfrac.mtsolvers.mtai import *
from pyfrac.modelling.kirchhoff import Kirchhoff

###############################################################################
# Load seismic data
# ===================================================
# For this example, we will use a toy example of a small homogenous model with a gridded surface receiver
# array. The data are modelled using the SOFI3D Finite Difference package.

data_dir = '../data/pyfrac_SOFIModelling'

# Load receiver geometry
recs_xzy = np.loadtxt(os.path.join(data_dir,'inputs/griddedarray_xzy_20m.dat')).T
nr = recs_xzy.shape[1]
print(nr)

# Load seismic data
expname = 'MT-90-90-180_Homogeneous_griddedarray'
vz = read_seis(os.path.join(data_dir, 'outputs/su/%s_vy.txt'%expname),
               nr=nr)

dt = 1e-3  # SOFI3D Time sampling rate
t_shift = 160  # Time shift required to align FD data to zero for Kirchhoff operators
tdur = 500  # Recording duration
efd_scaler = np.max(abs(vz))    # Scaler to make data more friendly
vz = vz[:, t_shift:t_shift+tdur]*efd_scaler

# Let's just double-check the data is loaded well. Observe the changes in polarity across the
# traces, this is the information that we utilise to determine the Moment Tensor.
fig,ax = plt.subplots(1,1,figsize=[15,5])
ax.imshow(vz.T, aspect='auto',cmap='binary_r')

###############################################################################
# Set up Problem
# ===================================================
# For this simple example, let's use a small homogeneous velocity model with a surface receiver array in a
# gridded formation that fully covers the velocity model.

# Load in the velocity model - this is the model used in the FD modelling to generate the data.
# At the moment it has boundaries that we used for the modelling but we will remove them after
# Loading the model
abs_bounds = 30
dx = dy = dz = 5
nx = 112
ny = 128
nz = 120
mod_w_bounds = np.fromfile(os.path.join(data_dir,'inputs/models/Homogeneous_xyz.vp'),
                           dtype='float32').reshape([nx, ny, nz])

# Remove absorbing boundaries for both the model and receiver coordinates
mod = mod_w_bounds[abs_bounds:-abs_bounds, abs_bounds:-abs_bounds, :-abs_bounds]  # z has free surface
nx, ny, nz = mod.shape
x, y, z = np.arange(nx)*dx, np.arange(ny)*dy, np.arange(nz)*dz
recs = np.array([recs_xzy[0]-(abs_bounds*dx), recs_xzy[2]-(abs_bounds*dx), recs_xzy[1]])

###############################################################################
# Define Source Frequency and Location
# ===================================================
omega_p = 20

sx = nx//2
sy = ny//2
sz = 2*nz//3
sloc_ind = [sx, sy, sz]

###############################################################################
# Get amplitude picks of P-wave arrival
# ===================================================

# TRAVEL TIME TABLE
trav = Kirchhoff._traveltime_table(z,
                                   x,
                                   y=y,
                                   recs=recs,
                                   vel=mod,
                                   mode='eikonal')
TTT_full = trav.reshape(nx,ny,nz,nr).transpose([3,0,1,2])
source_times = np.round(TTT_full[:, sloc_ind[0], sloc_ind[1], sloc_ind[2]]/dt).astype(int)
# Collect Amplitudes at arrival times based on given source location
vz_amps = np.ones(nr)
for i in range(nr):
    vz_amps[i] = vz[i, source_times[i]]
p_amps = vz_amps

plt.figure()
plt.imshow(vz[:, np.min(source_times)-50:150+np.min(source_times)].T, aspect='auto',
           extent=[0, nr, 150+np.min(source_times), np.min(source_times)-50],
           cmap='RdBu',
          )
plt.scatter(range(nr), source_times, marker='o', facecolors='none', edgecolors='k', s=5)

###############################################################################
# Compute Greens functions
# ===================================================

# AMPLITUDE TERMS
gamma_sourceangles, dist_table = collect_source_angles(x,y,z, reclocs=recs, nc=3)

MT_comp_dict = get_mt_computation_dict()  # This keeps everything nice and clean in the later G compute

Gz = pwave_Greens_comp(gamma_sourceangles,
                        dist_table,
                        sloc_ind,
                        mod,
                        MT_comp_dict,
                        comp_gamma_ind=2,
                        omega_p=omega_p,
                        )


###############################################################################
# Solve for the Moment Tensor
# ===================================================

# LSQR Inversion
mt_est = lsqr_mtsolver(Gz, p_amps)
mt_est /= np.max(abs(mt_est))

# Comparison with known MT
mt = np.array([0, 0, 0, 1, 0, 0])
print(mt_est)
print(mt-mt_est)
plt.scatter(range(6), mt, c='k', marker='s', label='True')
plt.scatter(range(6), mt_est, c='r', marker='x', label='Estimated')
plt.legend()
plt.title('MT Amplitude Inversion')