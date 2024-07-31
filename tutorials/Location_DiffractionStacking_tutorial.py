r"""
Diffraction Stacking Localisation - Simple scenario
===================================================

This tutorial illustrates how to perform source localisation using a 
diffraction stacking based on semblance.

In this tutorial we will consider a simple scenario where the subsurface is 
homogenous, and traveltimes are computed analytically as

.. math::
        t(\mathbf{x_r},\mathbf{x_s}) = \frac{d(\mathbf{x_r},\mathbf{x_s})}{v}

where :math:`d(\mathbf{x_r},\mathbf{x_s})` is the distance between the source 
and receiver, and :math:`v` is velocity (e.g. P-wave velocity :math:`v_p`).

The waveforms are computed using the FD modelling.

"""

#%%

###############################################################################
# Load all necessary packages
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

import numpy as np
import matplotlib.pyplot as plt

from pylops.utils import dottest
from pylops.utils.wavelets import ricker

# Import modelling utils
from fracspy.modelling.kirchhoff import Kirchhoff

# Import diffraction stacking utils
from fracspy.location.utils import dist2rec
from fracspy.location.migration import semblancediffstack

# Import visualization utils
from fracspy.visualisation.traceviz import traceimage
from fracspy.visualisation.eventimages import locimage3d

# Deal with warnings (for a cleaner code)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#%%

###############################################################################
# Setup
# ^^^^^
# Here we setup the parameters of the velocity model, geometry of receivers and 
# microseismic source for forward modelling

###############################################################################
# Velocity Model
# """"""""""""""

nx, ny, nz = 50, 50, 50
dx, dy, dz = 4, 4, 4
x, y, z = np.arange(nx)*dx, np.arange(ny)*dy, np.arange(nz)*dz

v0 = 1000 # initial velocity
vel = np.ones([nx,ny,nz])*v0

print(f"Velocity model shape: {vel.shape}")

###############################################################################
# Receivers
# """""""""

dr_xyz = 4*dx

grid_rx_locs = np.arange(dx, (dx*nx)-dx, dr_xyz)
grid_ry_locs = np.arange(dy, (dy*ny)-dy, dr_xyz)

rx, ry, rz = np.meshgrid(grid_rx_locs,
                         grid_ry_locs,
                         dz) 
recs = np.vstack((rx.flatten(), ry.flatten(), rz.flatten()))
nr = recs.shape[1]

print(f"Receiver array shape: {recs.shape}")

###############################################################################
# Microseismic sources
# """"""""""""""""""""

sx, sy, sz = [nx//4, ny//2, nz//2]
microseismic = np.zeros((nx, ny, nz))
microseismic[sx, sy, sz] = 1.


#%%

###############################################################################
# Generate synthetic data
# ^^^^^^^^^^^^^^^^^^^^^^^
# 

nt = 101 # number of time steps
dt = 0.004 # time step
f0 = 20 # Central frequency
t = np.arange(nt) * dt # time vector

###############################################################################
# Create signal wavelet
# """""""""""""""""""""
wav, wavt, wavc = ricker(t[:41], f0=f0)

###############################################################################
# Initialize operator
# """""""""""""""""""

Op = Kirchhoff(z=z, 
               x=x, 
               y=y, 
               t=t, 
               recs=recs, 
               vel=vel, 
               wav=wav, 
               wavcenter=wavc, 
               mode='eikonal', 
               engine='numba')

###############################################################################
# Check operator with dottest
# """""""""""""""""""""""""""

_ = dottest(Op, verb=True)

###############################################################################
# Forward modelling
# """""""""""""""""

frwddata_1d = Op @ microseismic.flatten().squeeze()
frwddata = frwddata_1d.reshape(nr,nt)


#%%

###############################################################################
# Diffraction stacking
# ^^^^^^^^^^^^^^^^^^^^
# Here we apply diffraction stakcing algorithm based on semblance to get the
# image volume and determine location from the maximum of this volume

###############################################################################
# Define search grid
# """"""""""""""""""

gx = x
gy = y
gz = z

###############################################################################
# Prepare traveltimes
# """""""""""""""""""

tt = 1 / v0*dist2rec(recs,gx,gy,gz)
print(f"Traveltime array shape: {tt.shape}")

###############################################################################
# Perform standard semblance-based diffraction stack
# """"""""""""""""""""""""""""""""""""""""""""""""""

dstacked, hc = semblancediffstack(data=frwddata, 
                                  n_xyz=[len(gx),len(gy),len(gz)], 
                                  tt=tt, dt=dt, nforhc=10)

#%%

###############################################################################
# Visualization of results and data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here we visualize the slices of the resulting image volume as well as 
# modelled input data and receiver geometry

###############################################################################
# Plot resulting image volume
# """""""""""""""""""""""""""

fig,axs = locimage3d(dstacked, x0=sx, y0=sy, z0=sz)

###############################################################################
# Plot modelled data
# """"""""""""""""""

fig, ax = traceimage(frwddata, climQ=99.99)
ax.set_title('Point Receivers')
ax.set_ylabel('Time steps')
fig = ax.get_figure()
fig.set_size_inches(10, 3)  # set size in inches

#%%
###############################################################################
# Plot wavelet
# """"""""""""
fig, ax = plt.subplots(1, 1)
ax.plot(wav)
fig.set_size_inches(10, 3)  # set size in inches

#%%
###############################################################################
# Plot receiver geometry
# ^^^^^^^^^^^^^^^^^^^^^^

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 8)  # set size in inches
ax.set_aspect('equal')
ax.scatter(recs[0],recs[1])
ax.scatter(sx*dx,sy*dy, marker='*')
ax.set_title('Receiver Geometry: map view')
ax.legend(['Receivers', 'Source'],loc='upper right')
ax.set_xlabel('x')
ax.set_ylabel('y')

