r"""
Waveform-based Moment Tensor Inversion - Multicomponent
========================================================
This is a follow-up of the Amplitude-based Moment Tensor Inversion tutorial. In this tutorial, we will extend the
MT inversion algorithm to work directly with waveforms instead of picked amplitudes. By avoiding any picking,
this method can determine the moment tensor of a microseismic source when the source location is not known a-prior.
As such, it can be considered a joint location and MT inversion algorithm.

We will start once again from the far-field particle velocity expression from a moment tensor source in a homogeneous full space
(from eq. 4.29, Aki and Richards) - see the Amplitude-based Moment Tensor Inversion tutorial for more details.

However, in comparison to the Amplitude-Based Moment Tensor Inversion tutorial, in this waveform-based approach we
assume a distributed source within a subsurface area of interest and use the following integral relation to reconstruct
the surface data:

.. math::
        v_1^P(\mathbf{x_r}, t) =
        w(t) * \int\limits_V G_{pq}(\mathbf{x_r}, \mathbf{x}, t) M_{pq}(\mathbf{x})\,\mathrm{d}\mathbf{x}

where :math:`M_{pq}` with :math:`p,q=1,2,3` are the so-called MT kernels and :math:`G_{pq}` are the so-called
Green's functions, whose high-frequency approximation can be written as:

.. math::
    G_{pq}(\mathbf{x_r}, \mathbf{x}, \omega) = a_{pq}(\mathbf{x_r}, \mathbf{x})
        e^{j \omega t(\mathbf{x_r}, \mathbf{x})}

Here :math:`a_{pq}` with :math:`p,q=1,2,3` represent the same coefficients used in the Amplitude-based Moment Tensor
Inversion tutorial.

To summarize, we will apply the following workflow:

    - Load model and data;
    - Compute the traveltimes & ray angles
    - Compute the Greens functions for the subsurface area of interest
    - Define the Kirchhoff-MT operator
    - Jointly solve for the location and MT with a least-squares solver

*Assumptions*: for now, the MTWI procedure assumes a homogeneous velocity model.

"""

import os
import matplotlib.pyplot as plt

import pyfrac

from pylops.utils.wavelets import ricker
from pyfrac.utils.sofiutils import read_seis
from pyfrac.modelling.kirchhoff import Kirchhoff
from pyfrac.mtsolvers.mtwi import *
from pyfrac.mtsolvers.homo_mti import collect_source_angles, multicomp_Greens_Pwave
from pyfrac.mtsolvers.mtutils import get_mt_computation_dict, get_mt_at_loc, expected_sloc_from_mtwi


###############################################################################
# Load model and seismic data
# ---------------------------
# For this example, we will use a toy homogenous model with a gridded surface
# receiver array. The data are modelled using the
# `SOFI3D <https://gitlab.kit.edu/kit/gpi/ag/software/sofi3d>`_
# Finite Difference modelling software. The model is the same that we have used
# in the FD modelling to generate the data. As such, it contains additional
# boundaries, which we need to remove prior to performing localisation.

# Directory containing input data
input_dir = '../data/pyfrac_SOFIModelling'

# Model parameters
abs_bounds = 30
dx = dy = dz = 5
nx = 112
ny = 128
nz = 120

# Modelling parameters
dt = 1e-3  # SOFI3D Time sampling rate
t_shift = 167  # Time shift required to align FD data to zero time for Kirchhoff operators
tdur = 500  # Recording duration

# Load model
mod_w_bounds = np.fromfile(os.path.join(input_dir,'inputs',
                                        'models',
                                        'Homogeneous_xyz.vp'),
                           dtype='float32').reshape([nx, ny, nz])

# Load receiver geometry
recs_xzy = np.loadtxt(os.path.join(input_dir,'inputs',
                                   'griddedarray_xzy_20m.dat')).T
nr = recs_xzy.shape[1]


# Load seismic data (note that Vz is Vy given the SOFI convention)
expname = 'MT-90-90-180_Homogeneous_griddedarray'

vx = read_seis(os.path.join(input_dir, 'outputs', 'su', '%s_vx.txt' % expname), nr=nr)
vy = read_seis(os.path.join(input_dir, 'outputs', 'su', '%s_vz.txt' % expname), nr=nr)
vz = read_seis(os.path.join(input_dir, 'outputs', 'su', '%s_vy.txt' % expname), nr=nr)

efd_scaler = np.max(abs(vz))    # Scaler to make data more friendly
vx = vx[:, t_shift:t_shift + tdur] / efd_scaler
vy = vy[:, t_shift:t_shift + tdur] / efd_scaler
vz = vz[:, t_shift:t_shift + tdur] / efd_scaler

# Combine into a single array
FD_data = np.array([vx, vy, vz])

# Remove absorbing boundaries for both the model and receiver coordinates
mod = mod_w_bounds[abs_bounds:-abs_bounds, abs_bounds:-abs_bounds, :-abs_bounds] # z has free surface
nx, ny, nz = mod.shape
x, y, z = np.arange(nx)*dx, np.arange(ny)*dy, np.arange(nz)*dz
recs = np.array([recs_xzy[0]-(abs_bounds*dx), recs_xzy[2]-(abs_bounds*dx), recs_xzy[1]])

###############################################################################
# Let's now double-check that the data has been loaded correctly. Observe the
# changes in polarity across the traces, this is the information that we
# utilise to determine the Moment Tensor.

fig, axs = plt.subplots(3, 1,figsize=[10, 8])
axs[0].imshow(vx.T, aspect='auto',cmap='binary_r')
axs[1].imshow(vy.T, aspect='auto',cmap='binary_r')
axs[2].imshow(vz.T, aspect='auto',cmap='binary_r')
plt.tight_layout()

###############################################################################
# Create modelling operator
# -------------------------
# First, we will define a Ricker wavelet with
# peak frequency of 20Hz. This is the same wavelet that we used in modelling;
# in real applications, this will need to be estimated from the data.

nt = vz.shape[1]
t = np.arange(nt)*dt
wav, wavt, wavc = ricker(t[:81], f0=20)

fig, ax = plt.subplots(figsize=(10, 2))
ax.plot(wavt, wav, 'k', lw=2)
ax.set_xlabel('t [s]')
ax.set_title('Wavelet')
ax.grid('on')
plt.tight_layout()

###############################################################################
# Second, we compute the traveltimes and ray angles to be used to create the
# Green's functions.

# Traveltime terms
trav = Kirchhoff._traveltime_table(z,
                                   x,
                                   y=y,
                                   recs=recs,
                                   vel=mod,
                                   mode='eikonal')

TTT_full = trav.reshape(nx,ny,nz,nr).transpose([3, 0, 1, 2])

# Amplitude terms
gamma_sourceangles, dist_table = collect_source_angles(x,y,z, reclocs=recs, nc=3)

###############################################################################
# And we can now compute the Green's functions. We will define an area of
# interest where we expect the source to be located. In fact, whilst in practice
# one could consider the entire subsurface, this comes with a computational and
# storage burden for the Green's functions.

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

# This keeps everything nice and clean in the later G compute
MT_comp_dict = get_mt_computation_dict()

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
# Finally we can create our Kirchhoff-MT operator

Mstack_Op = multicomp_pwave_mtioperator(
    x=x[xsi:xfi],
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
    Ms_scaling = 1e6,
    engine='numba'
    )

###############################################################################
# Joint localisation and MT inversion
# -----------------------------------
# Finally, we are ready to invert our waveform data for the 6 MT kernels.

# Dimensions of area of interest
nxyz = [nx_aoi, ny_aoi, nz_aoi]

# Adjoint
mt_adj = adjoint_mtmodelling(FD_data, Mstack_Op, nxyz)

# Least-squares inversion
mt_inv = lsqr_mtsolver(FD_data, Mstack_Op, nxyz)

###############################################################################
# Let's now extract both the expected location and MT source parameters

exp_sloc, _ = expected_sloc_from_mtwi(mt_inv)
print('Expected Source Location (AOI coord. ref.): \n', exp_sloc)
mt_at_loc = get_mt_at_loc(mt_inv, [int(exp_sloc[0]), int(exp_sloc[1]), int(exp_sloc[2])])
print('MT at expected Source Location (full): \n', mt_at_loc)
print('MT at expected Source Location (rounded): \n', np.round(mt_at_loc, decimals=2))


###############################################################################
# And finally we visualize the estimated kernels both from the adjoint and
# inverse approaches.

clim = 1
pyfrac.visualisation.eventimages.locimage3d(mt_adj[0], int(exp_sloc[0]), int(exp_sloc[1]), int(exp_sloc[2]), clipval=[-clim, clim])
pyfrac.visualisation.eventimages.locimage3d(mt_adj[1], int(exp_sloc[0]), int(exp_sloc[1]), int(exp_sloc[2]), clipval=[-clim, clim])
pyfrac.visualisation.eventimages.locimage3d(mt_adj[2], int(exp_sloc[0]), int(exp_sloc[1]), int(exp_sloc[2]), clipval=[-clim, clim])
pyfrac.visualisation.eventimages.locimage3d(mt_adj[3], int(exp_sloc[0]), int(exp_sloc[1]), int(exp_sloc[2]), clipval=[-clim, clim])
pyfrac.visualisation.eventimages.locimage3d(mt_adj[4], int(exp_sloc[0]), int(exp_sloc[1]), int(exp_sloc[2]), clipval=[-clim, clim])
pyfrac.visualisation.eventimages.locimage3d(mt_adj[5], int(exp_sloc[0]), int(exp_sloc[1]), int(exp_sloc[2]), clipval=[-clim, clim])

pyfrac.visualisation.eventimages.locimage3d(mt_inv[0], int(exp_sloc[0]), int(exp_sloc[1]), int(exp_sloc[2]), clipval=[-clim, clim])
pyfrac.visualisation.eventimages.locimage3d(mt_inv[1], int(exp_sloc[0]), int(exp_sloc[1]), int(exp_sloc[2]), clipval=[-clim, clim])
pyfrac.visualisation.eventimages.locimage3d(mt_inv[2], int(exp_sloc[0]), int(exp_sloc[1]), int(exp_sloc[2]), clipval=[-clim, clim])
pyfrac.visualisation.eventimages.locimage3d(mt_inv[3], int(exp_sloc[0]), int(exp_sloc[1]), int(exp_sloc[2]), clipval=[-clim, clim])
pyfrac.visualisation.eventimages.locimage3d(mt_inv[4], int(exp_sloc[0]), int(exp_sloc[1]), int(exp_sloc[2]), clipval=[-clim, clim])
pyfrac.visualisation.eventimages.locimage3d(mt_inv[5], int(exp_sloc[0]), int(exp_sloc[1]), int(exp_sloc[2]), clipval=[-clim, clim])
