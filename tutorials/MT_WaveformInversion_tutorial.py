r"""
Waveform-based Moment Tensor Inversion - Multicomponent
========================================================
This tutorial illustrates how we can determine the moment tensor of a microseismic source where
we do not know the source location. As such, the workflow can act as a joint location and MT
solution.

Microseismic data can be considered as:
$d=GM
Using the Greens functions as defined by Aki and Richards, the above can be expanded out as

.. math::
    v_i^\theta = j \omega_\theta ( \frac{\gamma_i\gamma_p\gamma_q}{4\pi\rho\alpha^3}  \frac{1}{r} )M_{pq}

where

        :math:`v` is the velocity measurements (seismic data)

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

For more information on the above equation, please refer to [LINK TO MTAI DOCUMENTATION].

In comparison to the Amplitude-Based Moment Tensor Inversion, in this waveform-based approach we do not pick the
amplitudes of the p-wave and therefore, do not need knowledge of the source location. Instead, we solve for the
MT kernels, e.g., :math:`M_{xx}` , :math:`M_{xy}` , :math:`M_{xz}`, etc., across a subsurface area of interest.
The resulting product, as you will see, is six MT kernel images.


The workflow consists of:
    - reading and pre-processing the seismic data
    - setting up the problem (subsurface models)
    - Obtain traveltimes & ray angles
    - Computing Greens functions for the subsurface area of interest
    - Make the combined Kirchhoff-MT operator
    - Jointly solve for the location and MT with a least squares solver
    - _[Bonus]_ Jointly solve for the location and MT with a FISTA solver



*Assumptions*: for now, the MTWI procedure assumes a homogeneous velocity model.

Keywords: Greens functions -- Kirchhoff
"""



import numpy as np
import os
import matplotlib.pyplot as plt
from pylops.utils import dottest
from pylops.utils.wavelets import *

from pyfrac.utils.sofiutils import read_seis

from pyfrac.modelling.kirchhoff import Kirchhoff
from pyfrac.modelling.trueamp_kirchhoff import Kirchhoff as TAKirchhoff

from pyfrac.locationsolvers.localisationutils import get_max_locs
from pyfrac.mtsolvers.mtwi import *
from pyfrac.mtsolvers.homo_mti import collect_source_angles, multicomp_Greens_Pwave
from pyfrac.mtsolvers.mtutils import get_mt_computation_dict, get_mt_at_loc, expected_sloc_from_mtwi
from pyfrac.visualisation.eventimages import locimage3d

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

# Vz is Vy cause SOFI ¯\_(ツ)_/¯
vx = read_seis(os.path.join(data_dir, 'outputs/su/%s_vx.txt'%expname),
               nr=nr)
vy = read_seis(os.path.join(data_dir, 'outputs/su/%s_vz.txt'%expname),
               nr=nr)
vz = read_seis(os.path.join(data_dir, 'outputs/su/%s_vy.txt'%expname),
               nr=nr)

sdt = 1e-3  # SOFI3D Time sampling rate
t_shift = 167  # Time shift required to align FD data to zero for Kirchhoff operators
tdur = 500  # Recording duration
efd_scaler = np.max(abs(vz))    # Scaler to make data more friendly

vx = vx[:, t_shift:t_shift+tdur]*efd_scaler
vy = vy[:, t_shift:t_shift+tdur]*efd_scaler
vz = vz[:, t_shift:t_shift+tdur]*efd_scaler
# Combine into a single array
FD_data = np.array([vx, vy, vz])

# Let's just double-check the data is loaded well. Observe the changes in polarity across the
# traces, this is the information that we utilise to determine the Moment Tensor.
fig,axs = plt.subplots(1,3,figsize=[15,5])
axs[0].imshow(vx.T, aspect='auto',cmap='binary_r')
axs[1].imshow(vy.T, aspect='auto',cmap='binary_r')
axs[2].imshow(vz.T, aspect='auto',cmap='binary_r')

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
mod_w_bounds = np.fromfile(os.path.join(data_dir,'inputs/models/Homogeneous_xyz.vp'),dtype='float32').reshape([nx,ny,nz])

# Remove absorbing boundaries for both the model and receiver coordinates
mod = mod_w_bounds[abs_bounds:-abs_bounds, abs_bounds:-abs_bounds, :-abs_bounds] # z has free surface
nx, ny, nz = mod.shape
x, y, z = np.arange(nx)*dx, np.arange(ny)*dy, np.arange(nz)*dz
recs = np.array([recs_xzy[0]-(abs_bounds*dx), recs_xzy[2]-(abs_bounds*dx), recs_xzy[1]])

###############################################################################
# Obtain traveltimes & ray angles, Estimate Wavelet
# ===================================================
# These are required for the Green's functions


# TRAVEL TIME TABLE
trav = Kirchhoff._traveltime_table(z,
                                   x,
                                   y=y,
                                   recs=recs,
                                   vel=mod,
                                   mode='eikonal')

TTT_full = trav.reshape(nx,ny,nz,nr).transpose([3,0,1,2])
# AMPLITUDE TERMS
gamma_sourceangles, dist_table = collect_source_angles(x,y,z, reclocs=recs, nc=3)

dt = sdt
nt = vz.shape[1]
t = np.arange(nt)*dt
wav, wavt, wavc = ricker(t[:81], f0=20)
plt.plot(wav)

###############################################################################
# Computing Greens functions for the subsurface area of interest
# ================================================================


###############################################################################
# Make cube area of interest as can't consider the full subsurface body
hwin_nx_aoi, hwin_ny_aoi, hwin_nz_aoi = 15, 13, 11  # half window lengths in x, y, z
winc_x, winc_y, winc_z = nx//2, ny//2, 2*nz//3  # Center points of the area of interest
# Defining area of interest
xsi, xfi = winc_x-hwin_nx_aoi, winc_x+hwin_nx_aoi+1   # start/end index of x-region of interest
ysi, yfi = winc_y-hwin_ny_aoi, winc_y+hwin_ny_aoi+1   # start/end index of y-region of interest
zsi, zfi = winc_z-hwin_nz_aoi, winc_z+hwin_nz_aoi+1   # start/end index of z-region of interest


# Parameters only for the area of interest
gamma_sourceangles_aoi = gamma_sourceangles[:, :, xsi:xfi, ysi:yfi, zsi:zfi]
dist_table_aoi = dist_table[:, xsi:xfi, ysi:yfi, zsi:zfi]
tt_table_aoi = TTT_full[:, xsi:xfi, ysi:yfi, zsi:zfi]

nr, nx_aoi, ny_aoi, nz_aoi = tt_table_aoi.shape
print(nr, nx_aoi, ny_aoi, nz_aoi)

MT_comp_dict = get_mt_computation_dict()  # This keeps everything nice and clean in the later G compute
# Computing Greens functions for AoI
Gx, Gy, Gz = multicomp_Greens_Pwave(nxyz=[nx_aoi, ny_aoi, nz_aoi],
                                    nr=nr,
                                    gamma_sourceangles=gamma_sourceangles_aoi,
                                    dist_table=dist_table_aoi,
                                    vel=mod,
                                    MT_comp_dict=MT_comp_dict,
                                    omega_p=1,
                                    )


###############################################################################
# Make the combined Kirchhoff-MT operator
# ===================================================

Mstack_Op = multicomp_pwave_mtioperator(x=x[xsi:xfi],
                                        y=y[ysi:yfi],
                                        z=z[zsi:zfi],
                                        recs=recs,
                                        t=t,
                                        wav=wav,
                                        wavc=wavc,
                                        tt_table=tt_table_aoi,
                                        Gx=Gx,
                                        Gy=Gy,
                                        Gz=Gz,
                                        Ms_scaling = 1e7,
                                        engine='numba'
                                        )

###############################################################################
# Jointly solve for the location and MT with a least squares solver
# ===================================================================


nxyz=[nx_aoi, ny_aoi, nz_aoi]
# ADJOINT
mt_adj = adjoint_mtmodelling(FD_data, Mstack_Op, nxyz)


# LSQR
mt_inv = lsqr_mtsolver(FD_data, Mstack_Op, nxyz)

exp_sloc, _ = expected_sloc_from_mtwi(mt_inv)
print('Expected Source Location (AOI coord. ref.): \n', exp_sloc)
mt_at_loc = get_mt_at_loc(mt_inv, [int(exp_sloc[0]), int(exp_sloc[1]), int(exp_sloc[2])])
print('MT at expected Source Location (full): \n', mt_at_loc)
print('MT at expected Source Location (rounded): \n', np.round(mt_at_loc, decimals=2))

clim = 1e-4
locimage3d(mt_inv[0], int(exp_sloc[0]), int(exp_sloc[1]), int(exp_sloc[2]), clipval=[-clim, clim])
locimage3d(mt_inv[1], int(exp_sloc[0]), int(exp_sloc[1]), int(exp_sloc[2]), clipval=[-clim, clim])
locimage3d(mt_inv[2], int(exp_sloc[0]), int(exp_sloc[1]), int(exp_sloc[2]), clipval=[-clim, clim])
locimage3d(mt_inv[3], int(exp_sloc[0]), int(exp_sloc[1]), int(exp_sloc[2]), clipval=[-clim, clim])
locimage3d(mt_inv[4], int(exp_sloc[0]), int(exp_sloc[1]), int(exp_sloc[2]), clipval=[-clim, clim])
locimage3d(mt_inv[5], int(exp_sloc[0]), int(exp_sloc[1]), int(exp_sloc[2]), clipval=[-clim, clim]);