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

# Import location utils
from fracspy.location import Location
from fracspy.location.utils import *
from fracspy.location.migration import *

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
mnx, mny, mnz = 112, 128, 120

# Load source parameters
source = np.loadtxt(os.path.join(input_dir,'inputs/centralsource.dat')).T
f0 = source[4] # source frequency

# Modelling parameters
dt = 1e-3  # SOFI3D Time sampling rate
t_shift = 360  # Time shift to reduce the time steps
tdur = 250  # Recording duration

# Load model
mod_w_bounds = np.fromfile(os.path.join(input_dir,'inputs',
                                        'models',
                                        'Homogeneous_xyz.vp'),
                           dtype='float32').reshape([mnx, mny, mnz])

# Get velocity value considering that the model is homogeneous
vp = float(mod_w_bounds[0][0][0])

# Load receiver geometry
recs_xzy = np.loadtxt(os.path.join(input_dir,'inputs/griddedarray_xzy_20m.dat')).T
#recs_xzy = np.loadtxt(os.path.join(input_dir,'inputs/walkaway8arms_xzy.dat')).T
nr = recs_xzy.shape[1]

# Load seismic data
expname = 'MT-90-90-180_Homogeneous_griddedarray'
#expname = 'explosive_Homogeneous_griddedarray'
# expname = 'MT-90-90-180_Homogeneous_walkaway8arms'
data_vz = read_seis(os.path.join(input_dir, 'outputs','su', f'{expname}_vy.txt'),
                    nr=nr)

# Define scaler and make data more friendly computation-wise
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
# Test basic functions
# ^^^^^^^^^^^^^^^^^^^^
# Here we test basic elements of diffraction stacking like moveout and polarity
# correction

###############################################################################
# Test moveout correction and plot corrected data
# """""""""""""""""""""""""""""""""""""""""""""""

# Compute traveltimes to receivers from the true location
tt_true = 1 / vp * np.squeeze(dist2rec(recs,sx,sy,sz))
itshifts_true = np.round((tt_true - tt_true.min(axis=0))/dt)
data_mc = moveout_correction(data=data_vz,
                             itshifts=itshifts_true)

# Plot data with corrected moveout
fig, ax = traceimage(data_mc, climQ=99.99, figsize=(10, 4))
ax.set_title('Data with corrected event moveout')
plt.tight_layout()

# Restore moveout (use negative time index tshifts)
data_mr = moveout_correction(data=data_mc,
                             itshifts=-itshifts_true)

# Plot data with restored moveout
fig, ax = traceimage(data_mr, climQ=99.99, figsize=(10, 4))
ax.set_title('Data with restored event moveout')
plt.tight_layout()

###############################################################################
# Test polarity correction and plot corrected data
# """"""""""""""""""""""""""""""""""""""""""""""""

# Compute compute vectorized Green tensor derivatives for the true location
vgtd_true = vgtd(x=sx,y=sy,z=sz,recs=recs)
# Compuite the GTG matrix
gtg_inv_true = mgtdinv(g=vgtd_true)

data_pc = polarity_correction(data = data_mc,
                              polcor_type = "mti",
                              g = vgtd_true,
                              gtg_inv = gtg_inv_true
                              )

# Plot data with corrected moveout and polarity
fig, ax = traceimage(data_pc, climQ=99.99, figsize=(10, 4))
ax.set_title('Data with corrected event moveout and polarity')
plt.tight_layout()

# Restore moveout for polarity corrected data
data_pc_mr = moveout_correction(data=data_pc,
                                itshifts=-itshifts_true)

# Plot data with restored moveout
fig, ax = traceimage(data_pc_mr, climQ=99.99, figsize=(10, 4))
ax.set_title('Data with restored event moveout and corrected polarity')
plt.tight_layout()

#%%

###############################################################################
# Prepare for location
# ^^^^^^^^^^^^^^^^^^^^
# Set up the location class and grid, compute traveltimes

###############################################################################
# Define location class using grid vectors
# """"""""""""""""""""""""""""""""""""""""
# We can use the original velocity model grid for location,
# but for the sake of having more representative image we shift the grid deeper
# Moreover, we reduce the grid step twice for efficiency

gdx = dx*2
gdy = dy*2
gdz = dz*2
gx = x[::2]
gy = y[::2]
gz = np.arange(150, 460, gdz)



# Set up the location class

L = Location(gx, gy, gz)

###############################################################################
# Prepare traveltimes
# """""""""""""""""""
from fracspy.location.utils import dist2rec
tt = 1 / vp*dist2rec(recs,gx,gy,gz)
print(f"Traveltime array shape: {tt.shape}")


#%%

##############################################################################
# Apply diffraction stacking without polarity correction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here we apply a diffraction stacking algorithm without 
# any polarity correction, for reference

###############################################################################
# Perform squared-value diffraction stacking without polarity correction
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

start_time = time()
print("Squared-value diffraction stacking without polarity correction...")
dstacked_sqd, hc_sqd = L.apply(data_vz,
                                kind="diffstack",
                                tt=tt, dt=dt, nforhc=10,
                                stack_type="squared",
                                output_type = "mean")
end_time = time()
print(f"Computation time: {end_time - start_time} seconds")

#%%

##############################################################################
# Apply diffraction stacking witth polarity correction using MTI
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here we apply various diffraction stacking algorithms with 
# polarity correction using moment tensor inversion

###############################################################################
# Perform squared-value diffraction stacking with polarity correction with MTI
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

start_time = time()
print("Squared-value diffraction stacking with polarity correction based on MTI...")
dstacked_sqd_mti, hc_sqd_mti = L.apply(data_vz,
                                       kind="diffstack",
                                       tt=tt, dt=dt, nforhc=10,
                                       stack_type="squared",
                                       output_type = "mean",
                                       polcor_type="mti",recs=recs)
end_time = time()
print(f"Computation time: {end_time - start_time} seconds")

###############################################################################
# Perform semblance-based diffraction stacking with polarity correction with MTI
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Define sliding window as two periods of the signal
swsize = int(2/f0/dt)
print(f"Sliding window size in samples: {swsize}")

start_time = time()
print("Semblance-based diffraction stacking with polarity correction based on MTI...")
dstacked_sem_mti, hc_sem_mti = L.apply(data_vz,
                                kind="diffstack",
                                tt=tt, dt=dt, nforhc=10,                                
                                stack_type="semblance", swsize = swsize,
                                output_type = "mean",
                                polcor_type="mti",recs=recs)
end_time = time()
print(f"Computation time: {end_time - start_time} seconds")


#%%

###############################################################################
# Visualisation of results
# ^^^^^^^^^^^^^^^^^^^^^^^^
# Here we visualise the slices of the resulting image volumes.
# Clearly, the result of application of stacking without polarity correction  
# shows an unfocused image, whereas for stacking with polarity correction the ]
# resulting images are focused in the correct location which fluctuates in 
# depth, reflecting the time-depth tradeoff.
# Uncertainty in depth direction is much higher due to surface acquisition.


###############################################################################
# Plot resulting image volumes from squared-value diffraction stacking
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Get the spatial limits for plotting
xlim = (min(gx),max(gx))
ylim = (min(gy),max(gy))
zlim = (min(gz),max(gz))

# Print true location
print('True event hypocenter:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*[sx, sy, sz]))

# Results of application:
fig,axs = locimage3d(dstacked_sqd, 
                      title='Location with squared-value diffraction stacking\nwithout polarity correction:',
                      x0=int(isx/2), y0=int(isy/2), z0=int(isz/2)-25,
                      xlim=xlim,ylim=ylim,zlim=zlim)

print('-------------------------------------------------------')
print('Event hypocenter from squared-value diffraction stacking without polarity correction:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*L.putongrid(hc_sqd)))
print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([sx, sy, sz],  L.putongrid(hc_sqd))))

fig,axs = locimage3d(dstacked_sqd_mti, 
                     title='Location with squared-value diffraction stacking\nwith polarity correction based on MTI:',
                     x0=int(isx/2), y0=int(isy/2), z0=int(isz/2)-25,
                     xlim=xlim,ylim=ylim,zlim=zlim)

print('-------------------------------------------------------')
print('Event hypocenter from squared-value diffraction stacking with polarity correction:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*L.putongrid(hc_sqd_mti)))
print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([sx, sy, sz], L.putongrid(hc_sqd_mti))))

fig,axs = locimage3d(dstacked_sem_mti, 
                     title='Location with semblance-based diffraction stacking\nwith polarity correction based on MTI:',
                     x0=int(isx/2), y0=int(isy/2), z0=int(isz/2)-25,
                     xlim=xlim,ylim=ylim,zlim=zlim)

print('-------------------------------------------------------')
print('Event hypocenter from semblance-based diffraction stacking with polarity correction:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*L.putongrid(hc_sem_mti)))
print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([sx, sy, sz],  L.putongrid(hc_sem_mti))))
