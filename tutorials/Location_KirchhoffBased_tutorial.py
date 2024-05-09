r"""
Kirchhoff-Based Localisation - Single component
========================================================
This tutorial illustrates how the source location can be determined

"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pylops.utils import dottest
from pylops.utils.wavelets import *

from pyfrac.utils.sofiutils import read_seis
from pyfrac.modelling.kirchhoff import Kirchhoff
from pyfrac.visualisation.traceviz import traceimage
from pyfrac.visualisation.eventimages import locimage3d
from pyfrac.locationsolvers.imaging import *
from pyfrac.locationsolvers.crosscorr_imaging import xcorr_imaging


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
expname = 'explosive_Homogeneous_griddedarray'
vz = read_seis(os.path.join(data_dir, 'outputs/su/%s_vy.txt'%expname),
               nr=nr)

dt = 1e-3  # SOFI3D Time sampling rate
t_shift = 167  # Time shift required to align FD data to zero for Kirchhoff operators
tdur = 500  # Recording duration
vz = vz[:, t_shift:]  # To account for selected ignition time
vz /= np.max(abs(vz))  # Normalise to get at reasonable amplitudes

# Let's just double-check the data is loaded well.
ax = traceimage(vz, climQ=99.99)
ax.set_title('SOFI FD data - Vertical Component')

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


# Make wavelet
nt = vz.shape[1]
t = np.arange(nt)*dt
wav, wavt, wavc = ricker(t[:81], f0=20)
plt.plot(wav)

# INITIALISE OPERATOR
Op = Kirchhoff(z=z,
               x=x,
               y=y,
               t=t,
               srcs=recs[:, :1],
               recs=recs,
               vel=1000*np.ones_like(mod),
               wav=wav,
               wavcenter=wavc,
               mode='eikonal',
               engine='numba')

# check operator with dottest
_ = dottest(Op, verb=True)


###############################################################################
# ASIDE: Checking the operator is optimal
# ===================================================
# Here we will just forward model the data to check that it looks similar to
# the loaded data. This is a slight cheat as we are working with synthetic data
# so we can `perfectly` match the forward data with the seismic data. However,
# even when working with field data, this is a good step to look at the source
# wavelet and an estimation of the ignition time. [More commentary on ignition
# time can be found in the tutorial on rolling detection.]

# Choose a Microseismic source location as function of grid points
sx = nx//2
sy = ny//2
sz = 2*nz//3
print(sx,sy,sz)
microseismic = np.zeros((nx, ny, nz))
microseismic[sx, sy, sz] = 1.

# PERFORM FORWARD (MODEL)
frwddata_1d = Op @ microseismic.flatten().squeeze()
frwddata  = frwddata_1d.reshape(nr,nt)

# Compare across full array
fig, axs = plt.subplots(1,2,figsize=[15,5])
axs[0].imshow(vz.T,aspect='auto', cmap='seismic', vmin=-1,vmax=1)
axs[1].imshow(frwddata.T,aspect='auto', cmap='seismic', vmin=-1,vmax=1)

# Compare trace level
plt.figure(figsize=[15,3])
plt.plot(-1*vz[10,250:],'k')
plt.plot(frwddata[10,250:])

###############################################################################
# Perform Adjoint
# ===================================================
# The adjoint is a good first step to looking at if energy pools in a certain subsurface location.
# However, it is the pure inverse of the forward operator so cannot handle noise or errors/uncertainties
# in the velocity model. In addition, the source location is likely to be a smoothed product, as opposed
# to the desired single location.
#
# In our setup, the adjoint is fast to perform. Therefore, we recommend computing the adjoint to look at the
# product, this could give an initial idea on possible source locations. However, we do not recommend it as
# the method for determining the final source location. We will cover those options further below.

migrated, mig_hc = migration(Op,
                             vz,
                             [nx,ny,nz],
                             nforhc=10)
print('True Hypo-Center:', [sx,sy,sz])
print('Migration Hypo-Centers:', mig_hc)

fig,axs = locimage3d(migrated,
                     x0=int(np.round(mig_hc[0])),
                     y0=int(np.round(mig_hc[1])),
                     z0=int(np.round(mig_hc[2])),
                     p=100)

###############################################################################
# Location Method 1: Least-Squares Inversion
# ===================================================
# The first method we will consider is simple least squares inversion

inv, inv_hc = lsqr_migration(Op,
                             vz,
                             [nx,ny,nz],
                             nforhc=10,
                             verbose=False)
print('True Hypo-Center:', [sx,sy,sz])
print('LSQR Inversion Hypo-Centers:', inv_hc)
fig,axs = locimage3d(inv,
                     x0=int(np.round(inv_hc[0])),
                     y0=int(np.round(inv_hc[1])),
                     z0=int(np.round(inv_hc[2])),
                     p=100)

###############################################################################
# Location Method 2: FISTA-based Inversion
# ===================================================
# FISTA INVERSION
fista, fista_hc = fista_migration(Op, vz, [nx,ny,nz], nforhc=10, verbose=False, fista_eps=1e1)
print('True Hypo-Center:', [sx,sy,sz])
print('FISTA Inversion Hypo-Centers:', fista_hc)
fig,axs = locimage3d(fista,
                     x0=int(np.round(fista_hc[0])),
                     y0=int(np.round(fista_hc[1])),
                     z0=int(np.round(fista_hc[2])),
                     p=100)

