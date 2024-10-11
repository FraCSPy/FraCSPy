r"""
3.2 Waveform-based Moment Tensor Inversion
==========================================
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
import numpy as np
import matplotlib.pyplot as plt

import fracspy

from pylops.utils.wavelets import ricker
from fracspy.utils.sofiutils import read_seis
from fracspy.mtinversion.utils import get_mt_max_locs, get_mt_at_loc
from fracspy.mtinversion.mtwi import *
from fracspy.visualisation.momenttensor_plots import MTMatrix_comparisonplot
# sphinx_gallery_thumbnail_number = 11

###############################################################################
# Load model and seismic data
# ---------------------------
# For this example, we will use a toy homogenous model with a gridded surface
# receiver array. The data are modelled using the
# `SOFI3D <https://gitlab.kit.edu/kit/gpi/ag/software/sofi3d>`_
# Finite Difference modelling software. The model is the same that we have used
# in the FD modelling to generate the data. As such, it contains additional
# boundaries, which we need to remove prior to performing localisation.

default_cmap = 'seismic'

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
vz = read_seis(os.path.join(input_dir, 'outputs', 'su', 
                            '%s_vy.txt' % expname), 
               nr=nr)
vz = vz[:, t_shift: t_shift + tdur]
efd_scaler = np.max(abs(vz))  # Scaler to make data more friendly
vz /= efd_scaler

# Remove absorbing boundaries for both the model and receiver coordinates
mod = mod_w_bounds[abs_bounds:-abs_bounds, abs_bounds:-abs_bounds, :-abs_bounds] # z has free surface
nx, ny, nz = mod.shape
x, y, z = np.arange(nx)*dx, np.arange(ny)*dy, np.arange(nz)*dz
recs = np.array([recs_xzy[0]-(abs_bounds*dx), recs_xzy[2]-(abs_bounds*dx), recs_xzy[1]])

###############################################################################
# Let's now double-check that the data has been loaded correctly. Observe the
# changes in polarity across the traces, this is the information that we
# utilise to determine the Moment Tensor.

fig, ax = fracspy.visualisation.traceviz.traceimage(vz, climQ=99.99, figsize=(10, 4))

###############################################################################
# Create modelling operator
# -------------------------
# First, we will define a Ricker wavelet with
# peak frequency of 20Hz. This is the same wavelet that we used in modelling;
# in real applications, this will need to be estimated from the data.

omega_p = 20
nt = vz.shape[1]
t = np.arange(nt)*dt
wav, wavt, wavc = ricker(t[:81], f0=omega_p)

fig, ax = plt.subplots(figsize=(10, 2))
ax.plot(wavt, wav, 'k', lw=2)
ax.set_xlabel('t [s]')
ax.set_title('Wavelet')
ax.grid('on')
plt.tight_layout()

###############################################################################
# Second we define an area of interest where we expect the source to be located.
# In fact, whilst in practice one could consider the entire subsurface,
# this comes with a computational and storage burden for the Green's functions.

sx = nx // 2
sy = ny // 2
sz = 2 * nz // 3
sloc_ind = [sx, sy, sz]

hwin_nx_aoi, hwin_ny_aoi, hwin_nz_aoi = 15, 13, 11  # half window lengths in x, y, z
winc_x, winc_y, winc_z = nx // 2, ny // 2, 2 * nz // 3  # Center points of the area of interest

# Defining area of interest
xsi, xfi = winc_x-hwin_nx_aoi, winc_x+hwin_nx_aoi+1   # start/end index of x-region of interest
ysi, yfi = winc_y-hwin_ny_aoi, winc_y+hwin_ny_aoi+1   # start/end index of y-region of interest
zsi, zfi = winc_z-hwin_nz_aoi, winc_z+hwin_nz_aoi+1   # start/end index of z-region of interest
nx_aoi = xfi - xsi
ny_aoi = yfi - ysi
nz_aoi = zfi - zsi

# MT in area of interest
MT_aoi = np.zeros([6, nx_aoi, ny_aoi, nz_aoi])  # MT components as images
MT_selected = -1 * np.array([0, 0, 0, 1, 0, 0])
MT_aoi[:, nx_aoi//2, ny_aoi//2, nz_aoi//2] = MT_selected

###############################################################################
# Next, we create our Kirchhoff-MT operator

Ms_scaling = 1.92e10
mtw = MTW(x, y, z, recs, mod, sloc_ind,
          2, omega_p, (hwin_nx_aoi, hwin_ny_aoi, hwin_nz_aoi),
          t, wav, wavc, multicomp=False,
          Ms_scaling=Ms_scaling,
          engine='numba')
data = mtw.model(MT_aoi)

###############################################################################
# Joint localisation and MT inversion
# -----------------------------------
# Finally, we are ready to invert our waveform data for the 6 MT kernels.

# Adjoint
mt_adj = mtw.adjoint(vz)

# Inversion
mt_inv = mtw.lsi(vz, niter=100, verbose=True)

###############################################################################
# Let's now compare the expected and true MT source parameters at the true 
# location. Note that for the expected MT parameters, we display their 
# normalized version since the modelling operator is only accurate up to relative
# amplitudes.

mt = np.array([0, 0, 0, 1, 0, 0])
mt_at_loc = get_mt_at_loc(mt_inv, [sloc_ind[0]-xsi, sloc_ind[1]-ysi, sloc_ind[2]-zsi])

MTMatrix_comparisonplot(mt, mt_at_loc / np.abs(np.array(mt_at_loc)).max())

###############################################################################
# Finally we visualize the estimated kernels from the inversion.

clim = 1e-4

fracspy.visualisation.eventimages.locimage3d(mt_inv[0], sloc_ind[0]-xsi, sloc_ind[1]-ysi, sloc_ind[2]-zsi, 
                                             clipval=[-clim, clim], title='Mxx')
fracspy.visualisation.eventimages.locimage3d(mt_inv[1], sloc_ind[0]-xsi, sloc_ind[1]-ysi, sloc_ind[2]-zsi, 
                                             clipval=[-clim, clim], title='Myy')
fracspy.visualisation.eventimages.locimage3d(mt_inv[2], sloc_ind[0]-xsi, sloc_ind[1]-ysi, sloc_ind[2]-zsi, 
                                             clipval=[-clim, clim], title='Mzz')
fracspy.visualisation.eventimages.locimage3d(mt_inv[3], sloc_ind[0]-xsi, sloc_ind[1]-ysi, sloc_ind[2]-zsi, 
                                             clipval=[-clim, clim], title='Mxy')
fracspy.visualisation.eventimages.locimage3d(mt_inv[4], sloc_ind[0]-xsi, sloc_ind[1]-ysi, sloc_ind[2]-zsi, 
                                             clipval=[-clim, clim], title='Mxz')
fracspy.visualisation.eventimages.locimage3d(mt_inv[5], sloc_ind[0]-xsi, sloc_ind[1]-ysi, sloc_ind[2]-zsi, 
                                             clipval=[-clim, clim], title='Mzz')
