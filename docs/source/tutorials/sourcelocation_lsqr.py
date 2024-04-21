r"""
LSQR Kirchhoff-based Inversion for Source Location
===================================================
This example shows how to create pre-stack angle gathers using
the `pyfrac.locationsolvers.imaging.lsqr_migration`.
"""

import numpy as np
import matplotlib.pyplot as plt
from pylops.utils import dottest
from pylops.utils.wavelets import *

from pyfrac.modelling.kirchhoff import Kirchhoff
from pyfrac.visualisation.traceviz import traceimage
from pyfrac.visualisation.eventimages import locimage3d

r"""
Set up Problem - Velocity Model + Receiver Geometry
===================================================
For this simple example, let's use a small homogeneous velocity model with a surface receiver array in a 
gridded formation that fully covers the velocity model.
"""
# Model Definition
nx, ny, nz = 50, 50, 50  # Number of dimensions of model
dx, dy, dz = 4, 4, 4  # Grid spacing of model
x, y, z = np.arange(nx)*dx, np.arange(ny)*dy, np.arange(nz)*dz  # Model coordinates

# Velocity Model
vp = 1000  # P-wave velocity
vel = np.ones([nx,ny,nz])*vp  # 3D Velocity Model

# Receiver array
dr_xyz = 4*dx  # Spacing of receivers as a function of the model step size
grid_rx_locs = np.arange(dx, (dx*nx)-dx, dr_xyz)  # Get receiver x-coords
grid_ry_locs = np.arange(dy, (dy*ny)-dy, dr_xyz)  # Get receiver y-coords
rx, ry, rz = np.meshgrid(grid_rx_locs,
                         grid_ry_locs,
                         dz)
recs = np.vstack((rx.flatten(), ry.flatten(), rz.flatten()))  # Flatten grid
nr = recs.shape[1]  # Number of receivers (useful parameter for later)

# Plot receiver array
plt.figure()
plt.scatter(recs[0], recs[1])
r"""
Make Kirchhoff imaging operator
=================================================
The Kirchhoff imaging operator 
"""
# Define time parameters
nt = 251
dt = 0.004
t = np.arange(nt)*dt

# Make source wavelet
fc = 20  # expected central frequency
wav, wavt, wavc = ricker(t[:41], f0=20)

# INITIALISE OPERATOR
Op = Kirchhoff(z=z,
               x=x,
               y=y,
               t=t,
               srcs=recs[:, :1],
               recs=recs,
               vel=vel,
               wav=wav,
               wavcenter=wavc,
               mode='eikonal',
               engine='numba')

# check operator with dottest
_ = dottest(Op, verb=True)