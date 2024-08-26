r"""

Diffraction Stacking Localisation With Moment Tensor Inversion
==============================================================
This tutorial illustrates how to perform source localisation using 
diffraction stacking with moment tensor inversion. 

We consider here a homogeneous subsurface model and 
a point Double-Couple (DC) microseismic source.
We also consider only P-waves for simplicity here.

Traveltimes
^^^^^^^^^^^
In a homogeneous medium traveltimes are computed analytically as

.. math::
        t(\mathbf{x_r},\mathbf{x_s}) = \frac{d(\mathbf{x_r},\mathbf{x_s})}{v}

where :math:`d(\mathbf{x_r},\mathbf{x_s})` is the distance between a source 
at :math:`\mathbf{x_s}` and a receiver at :math:`\mathbf{x_r}`, and 
:math:`v` is medium wave velocity (e.g. P-wave velocity :math:`v_p`).

Waveforms
^^^^^^^^^
The input data waveforms are computed with the help of PyLops operator which
involves finite-difference (FD) modelling.

See more:
https://pylops.readthedocs.io

Diffraction stacking with moment tensor inversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Basics of the simple diffraction stacking are explained in
:ref:`sphx_glr_tutorials_Location_DiffractionStacking_tutorial.py`.

References
^^^^^^^^^^
Anikiev, D. (2015). Joint detection, location and source mechanism 
determination of microseismic events (Doctoral dissertation). 
St. Petersburg State University. St. Petersburg. 
https://disser.spbu.ru/files/phdspsu2015/Anikiev_PhD_web_final.pdf

Anikiev, D., Valenta, J., Staněk, F. & Eisner, L. (2014). Joint location and 
source mechanism inversion of microseismic events: Benchmarking on seismicity 
induced by hydraulic fracturing. Geophysical Journal International, 198(1), 
249–258. https://doi.org/10.1093/gji/ggu126

Neidell, N. S., & Taner, M. T. (1971). SEMBLANCE AND OTHER COHERENCY MEASURES 
FOR MULTICHANNEL DATA. Geophysics, 36(3), 482–497. 
https://doi.org/10.1190/1.1440186

Trojanowski, J., & Eisner, L. (2016). Comparison of migration‐based location 
and detection methods for microseismic events. Geophysical Prospecting, 65(1), 
47–63. https://doi.org/10.1111/1365-2478.12366
"""

#%%

###############################################################################
# Load all necessary packages
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

import os
import fracspy
import numpy as np
import matplotlib.pyplot as plt

# Import modelling utils
from fracspy.modelling.kirchhoff import Kirchhoff

# Import data utils
from fracspy.utils.sofiutils import read_seis

# Import diffraction stacking utils
from fracspy.location import Location

# Import visualisation utils
from fracspy.visualisation.traceviz import traceimage
from fracspy.visualisation.eventimages import locimage3d


# Deal with warnings (for a cleaner code)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Track computation time
from time import time


#%%
###############################################################################
# Load model and seismic data
# ---------------------------
# For this example, we will use a toy example of a small homogenous model with a gridded surface receiver
# array, same as in :ref:`sphx_glr_tutorials_MT_AmplitudeINversion_tutorial.py`.
#The data are modelled using the SOFI3D Finite Difference package.

# Directory containing input data
input_dir = '../data/pyfrac_SOFIModelling'

# Loading the model
abs_bounds = 30
dx = dy = dz = 5
mnx = 112
mny = 128
mnz = 120

# Load source parameters
source = np.loadtxt(os.path.join(input_dir,'inputs/centralsource.dat')).T
sf = source[3] # source frequency

# Modelling parameters
dt = 1e-3  # SOFI3D Time sampling rate
t_shift = 160  # Time shift required to align FD data to zero
tdur = 500  # Recording duration

# Load model
mod_w_bounds = np.fromfile(os.path.join(input_dir,'inputs',
                                        'models',
                                        'Homogeneous_xyz.vp'),
                           dtype='float32').reshape([mnx, mny, mnz])

# Get velocity assuming it is homogeneous (to speed up traveltimes computations)
vp = float(mod_w_bounds[0][0][0])

# Load receiver geometry
recs_xzy = np.loadtxt(os.path.join(input_dir,'inputs/griddedarray_xzy_20m.dat')).T
#recs_xzy = np.loadtxt(os.path.join(input_dir,'inputs/walkaway8arms_xzy.dat')).T
nr = recs_xzy.shape[1]

# Load seismic data
#expname = 'MT-90-90-180_Homogeneous_griddedarray'
expname = 'explosive_Homogeneous_griddedarray'
# expname = 'MT-90-90-180_Homogeneous_walkaway8arms'
data_vz = read_seis(os.path.join(input_dir, 'outputs','su', f'{expname}_vy.txt'),
                    nr=nr)

# Define scaler and make data more friendly
efd_scaler = np.max(abs(data_vz)) 
data_vz = data_vz[:, t_shift: t_shift + tdur] * efd_scaler

# Remove absorbing boundaries from the model, source and receiver coordinates
mod = mod_w_bounds[abs_bounds:-abs_bounds, abs_bounds:-abs_bounds, :-abs_bounds] # z has free surface
nx, ny, nz = mod.shape
x, y, z = np.arange(nx) * dx, np.arange(ny) * dy, np.arange(nz) * dz
sx, sy, sz = source[0]-(abs_bounds*dx), source[2]-(abs_bounds*dy), source[1] 
recs = np.array([recs_xzy[0]-(abs_bounds*dx), recs_xzy[2]-(abs_bounds*dy), recs_xzy[1]])
# Get indices of source
isx, isy, isz = int(sx/dx), int(sy/dy), int(sz/dz)


###############################################################################
# Plot input data
# ^^^^^^^^^^^^^^^
# Let's now double-check that the data has been loaded correctly. Observe the
# changes in polarity across the  traces; this is the information that we utilise
# to determine the Moment Tensor.

fig, ax = traceimage(data_vz, climQ=99.99, figsize=(10, 4))
ax.set_title('SOFI FD data - Vertical Component')
plt.tight_layout()


###############################################################################
# Plot source location and receiver geometry
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 8)  # set size in inches
ax.set_aspect('equal')
ax.scatter(recs[0],recs[1])
ax.scatter(sx,sy, marker='*')
ax.set_title('Receiver Geometry: map view')
ax.legend(['Receivers', 'Source'],loc='upper right')
ax.set_xlabel('x')
ax.set_ylabel('y')

#%%

###############################################################################
# Prepare for location
# ^^^^^^^^^^^^^^^^^^^^

###############################################################################
# Define location class using grid vectors
# """"""""""""""""""""""""""""""""""""""""
# Use the original velocity model grid for location (the grid can be different)

gx = x
gy = y
gz = z

# Set up the location class

L = Location(gx, gy, gz)

###############################################################################
# Prepare traveltimes
# """""""""""""""""""
from fracspy.location.utils import dist2rec
tt = 1 / vp*dist2rec(recs,gx,gy,gz)
print(f"Traveltime array shape: {tt.shape}")


# ###############################################################################
# # Compute traveltimes
# # """""""""""""""""""
# # We can now model the traveltimes from the grid points to each of the receivers
# # Here, unlike :ref:`sphx_glr_tutorials_Location_DiffractionStacking_tutorial.py, 
# # we use :py:class:`fracspy.modelling.kirchhoff.Kirchhoff._traveltime_table` 
# # function to compute traveltimes analytically

# start_time = time()
# print("Computing traveltimes...")
# # Traveltime table
# tt = Kirchhoff._traveltime_table(z=gz,
#                                  x=gx,
#                                  y=gy,
#                                  recs=recs,
#                                  vel=vp,
#                                  mode='analytic')
# tt = tt.reshape(nx,ny,nz,nr).transpose([3,0,1,2])
# # Show consumed time
# end_time = time()
# print(f"Computation time: {end_time - start_time} seconds")
# print(tt.shape)

#%%

###############################################################################
# Apply diffraction stacking to clean data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here we apply various diffraction stacking algorithms to clean noise-free 
# data, get the image volume and determine location from the maximum of this 
# volume.

###############################################################################
# Perform absolute-value diffraction stacking
# """""""""""""""""""""""""""""""""""""""""""

start_time = time()
print("Absolute-value diffraction stacking...")
dstacked_abs, hc_abs = L.apply(data_vz,
                               kind="absdiffstack",
                               tt=tt, dt=dt, nforhc=10)
end_time = time()
print(f"Computation time: {end_time - start_time} seconds")


#%%

###############################################################################
# Visualisation of results
# ^^^^^^^^^^^^^^^^^^^^^^^^
# Here we visualise the slices of the resulting image volume

###############################################################################
# Plot resulting image volumes from absolute-value diffraction stacking
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Results of application to clean data:
fig,axs = locimage3d(dstacked_abs,
                      title='Location with absolute-value diffraction stacking:\nclean data',
                      x0=isx, y0=isy, z0=isz)
print('True event hypocenter:', [isx, isy, isz])
print('Event hypocenter from absolute diffraction stacking:', hc_abs.tolist())
print('Location error:', [x - y for x, y in zip([isx, isy, isz], hc_abs.tolist())])
