r"""
1.2 Diffraction Stacking: Detection
===================================
This tutorial illustrates how to perform microseismic event detection using diffraction stacking. 

We consider here a simple case of a homogeneous subsurface model and a set of point microseismic sources with various epicenters and various origin times.
We consider only P-waves and single-component receivers for simplicity.

Traveltimes
^^^^^^^^^^^
In a homogeneous medium traveltimes are computed analytically as

.. math::
        t(\mathbf{x_r},\mathbf{x_s}) = \frac{d(\mathbf{x_r},\mathbf{x_s})}{v}

where :math:`d(\mathbf{x_r},\mathbf{x_s})` is the distance between a source at 
:math:`\mathbf{x_s}` and a receiver at :math:`\mathbf{x_r}`, 
and :math:`v` is medium wave velocity (e.g. P-wave velocity :math:`v_p`).

Waveforms
^^^^^^^^^
The input data waveforms are computed with the help of PyLops Kirchhoff operator which uses Kirchhoff integral relation with high-frequency Green's functions.

See more information here:
https://pylops.readthedocs.io

Detection by diffraction stacking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Basics of simple diffraction stacking are explained in
:ref:`sphx_glr_tutorials_Location_DiffractionStacking_tutorial.py`.

Microseismic data always contain scattered energy and noise which may result in multiple
local maxima of the 4D imaging function. Therefore, a certain criterion is required to identify these local maxima
as detections of microseismic events.
For this purpose a joint detection algorithm was proposed in Anikiev at al. (2014).
For every time :math:`t` the maximum of the image function over all potential locations is
evaluated (thus only one event at any given time :math:`t` is assumed):

.. math::
        F_t(t) = \max_{\mathbf{r}} F(\mathbf{r},t)

The leading local maxima of the function :math:`F_t(t)` (later on referred to it as the
maximum stack function or MSF) occur at the origin times of microseismic events (Anikiev 2015).
These local maxima can be found by triggering algorithms, usually used for automatic picking of
seismic signal, for instance, the STA/LTA (Short Term Average / Long Term Average) method
 (e.g., Withers et al., 1998; Trnkoczy, 2012). Local maxima are detected by
measuring the ratio of average stack values in short and long sliding time windows and
comparing this ratio with the pre-defined STA/LTA threshold. As seismic waves scatter at
the near-surface or along the path, there can be multiple extrema in the MSF
corresponding to later arrivals. Therefore, it is necessary to make sure that only the first
(leading) maximum in a group is identified for event detection and location (Anikiev 2015).


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

Trnkoczy, A. (2012). Understanding and parameter setting of STA/LTA trigger algorithm. 
In: Bormann, P. (Ed.), New Manual of Seismological Observatory Practice 2 (NMSOP-2), Potsdam: Deutsches GeoForschungsZentrum GFZ, 1-20.
https://doi.org/10.2312/GFZ.NMSOP-2_IS_8.1

Withers, M., Aster, R., Young, C., Beiriger, J., Harris, M., Moore, S., & Trujillo, J. (1998). 
A comparison of select trigger algorithms for automated global seismic phase and event detection. 
Bulletin of the Seismological Society of America, 88(1), 95–106. https://doi.org/10.1785/bssa0880010095

"""
#%%

###############################################################################
# Load all necessary packages
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

# sphinx_gallery_thumbnail_number = 6
import numpy as np
import matplotlib.pyplot as plt

from pylops.utils import dottest
from pylops.utils.wavelets import ricker

# Import modelling utils
from fracspy.modelling.kirchhoff import Kirchhoff
from fracspy.utils.synthutils import add_noise

# Import location utils
from fracspy.location import Location
from fracspy.location.utils import *

# Import visualisation utils
from fracspy.visualisation.traceviz import traceimage
from fracspy.visualisation.eventimages import locimage3d

# Deal with warnings (for a cleaner code)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Track computation time
from time import time 
# sphinx_gallery_thumbnail_number = 4
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
# Here we set up multiple sources at different locations 
# and with different origin times

# number of sources
nsrc=3

# Declare empty arrays
isx, isy, isz = [np.zeros(nsrc).astype(int), np.zeros(nsrc).astype(int), np.zeros(nsrc).astype(int)]
sx, sy, sz = [np.zeros(nsrc), np.zeros(nsrc), np.zeros(nsrc)] 

# Fill grid locations of events
isx[0], isy[0], isz[0] = [nx//4, ny//2, nz//2]
isx[1], isy[1], isz[1] = [(3*nx)//4, ny//2, (3*nz)//4]
isx[2], isy[2], isz[2] = [nx//2, ny//4, nz//4]

# Get real locations
for isrc in np.arange(nsrc):
    sx[isrc], sy[isrc], sz[isrc] = isx[isrc]*dx, isy[isrc]*dy, isz[isrc]*dz 

# Origin times in sec
ort = [0, 0.15, 0.40]

# Define different magnitudes
microseismic0 = np.zeros((nx, ny, nz))
microseismic1 = np.zeros((nx, ny, nz))
microseismic2 = np.zeros((nx, ny, nz))
microseismic0[isx[0], isy[0], isz[0]] = 1.
microseismic1[isx[1], isy[1], isz[1]] = 0.3
microseismic2[isx[2], isy[2], isz[2]] = 0.8


#%%

###############################################################################
# Generate synthetic data
# ^^^^^^^^^^^^^^^^^^^^^^^
# 
start_time = time()
print("Generating synthetic data...")
nt = 161 # number of time steps
dt = 0.004 # time step
f0 = 20 # Central frequency
t = np.arange(nt) * dt # time vector

###############################################################################
# Create signal wavelet
# """""""""""""""""""""
wav, wavt, wavc = ricker(t[:41], f0=f0)

###############################################################################
# Initialise operator
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
# This test can help to detect errors in the operator implementation.

_ = dottest(Op, verb=True)

#%%

###############################################################################
# Forward modelling
# """""""""""""""""
# Apply operator to model data for each event
frwddata0 = (Op @ microseismic0.flatten().squeeze()).reshape(nr,nt)
frwddata1 = (Op @ microseismic1.flatten().squeeze()).reshape(nr,nt)
frwddata2 = (Op @ microseismic2.flatten().squeeze()).reshape(nr,nt)

# Combine the data
frwddata = np.roll(frwddata0,shift=int(ort[0]/dt), axis=1) + np.roll(frwddata1,shift=int(ort[1]/dt), axis=1) + np.roll(frwddata2,shift=int(ort[2]/dt), axis=1)

#frwddata = frwddata0 + frwddata1,shift=int(ort[0]/dt), axis=1) 
#+ np.roll(frwddata2,shift=int(ort[2]/dt), axis=1)

# Contaminate data with white noise
# """""""""""""""""""""""""""""""""

# Fix the seed for reproducibility
seed=1

# Fix SNR levels
snr_wn=1
snr_sn=1/10
snr_rn=1/5

# Fix traces for ringy noise
trind_rn = np.arange(1,nr,11)

# Add white noise of defined SNR
frwddata_wn = add_noise(frwddata,noise_type="white",snr=snr_wn,seed=seed)

# Contaminate data with spiky noise
# """""""""""""""""""""""""""""""""

# Add noise spikes with twice as bigger SNR
frwddata_sn = add_noise(frwddata,noise_type="spiky",snr=snr_sn,seed=seed)

# Contaminate data with ringy noise
# """""""""""""""""""""""""""""""""

# Add ringy noise on some traces
frwddata_rn = add_noise(frwddata,noise_type="ringy",snr=snr_rn,
                        trind=trind_rn,seed=seed)

# Show consumed time
end_time = time()
print(f"Computation time: {end_time - start_time} seconds")

#%%

###############################################################################
# Plot input data
# ^^^^^^^^^^^^^^^

# Plot modelled data
# """"""""""""""""""

fig, ax = traceimage(frwddata, climQ=99.99)
ax.set_title('Noise-free modelled data')
ax.set_ylabel('Time steps')
fig = ax.get_figure()
fig.set_size_inches(10, 3)  # set size in inches

###############################################################################
# Plot modelled data contaminated with white noise
# """"""""""""""""""""""""""""""""""""""""""""""""

fig, ax = traceimage(frwddata_wn, climQ=99.99)
ax.set_title(f"Modelled data contaminated with white noise of SNR={snr_wn}")
ax.set_ylabel('Time steps')
fig = ax.get_figure()
fig.set_size_inches(10, 3)  # set size in inches

# ###############################################################################
# # Plot modelled data contaminated with spiky noise
# # """"""""""""""""""""""""""""""""""""""""""""""""

# fig, ax = traceimage(frwddata_sn, climQ=99.99)
# ax.set_title(f"Modelled data contaminated with spiky noise of SNR={snr_sn}")
# ax.set_ylabel('Time steps')
# fig = ax.get_figure()
# fig.set_size_inches(10, 3)  # set size in inches

# ###############################################################################
# # Plot modelled data contaminated with ringy noise
# # """"""""""""""""""""""""""""""""""""""""""""""""

# fig, ax = traceimage(frwddata_rn, climQ=99.99)
# ax.set_title(f"Modelled data contaminated with ringy noise of SNR={snr_rn}")
# ax.set_ylabel('Time steps')
# fig = ax.get_figure()
# fig.set_size_inches(10, 3)  # set size in inches

###############################################################################
# Plot receiver geometry
# """"""""""""""""""""""

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 8)  # set size in inches
ax.set_aspect('equal')
ax.scatter(recs[0],recs[1])
for isrc in np.arange(nsrc):
    ax.scatter(sx[isrc],sy[isrc], marker='*')

ax.set_title('Receiver Geometry: map view')
ax.legend(['Receivers', 'Source 1', 'Source 2', 'Source 3'],loc='upper right')
_ = ax.set_xlabel('x')
_ = ax.set_ylabel('y')

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

tt = 1 / v0*dist2rec(recs,gx,gy,gz)
print(f"Traveltime array shape: {tt.shape}")


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
dstacked_abs, hc_abs = L.apply(frwddata, 
                      kind="diffstack",
                      x=gx, y=gy, z=gz,
                      tt=tt, dt=dt, nforhc=10,
                      stack_type="absolute",
                      output_type="mean")
end_time = time()
print(f"Computation time: {end_time - start_time} seconds")

###############################################################################
# Perform squared-value diffraction stacking
# """""""""""""""""""""""""""""""""""""""""""

start_time = time()
print("Squared-value diffraction stacking...")
dstacked_sqd, hc_sqd = L.apply(frwddata, 
                      kind="diffstack",
                      x=gx, y=gy, z=gz,
                      tt=tt, dt=dt, nforhc=10,
                      stack_type="squared")
end_time = time()
print(f"Computation time: {end_time - start_time} seconds")

# ###############################################################################
# # Perform semblance-based diffraction stacking without sliding time window
# # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# start_time = time()
# print("Semblance-based diffraction stacking...")
# # Run the stacking using Location class
# dstacked_semb, hc_semb = L.apply(frwddata, 
#                       kind="diffstack",
#                       x=gx, y=gy, z=gz,
#                       tt=tt, dt=dt, nforhc=10,
#                       stack_type="semblance")
# end_time = time()
# print(f"Computation time: {end_time - start_time} seconds")

# ###############################################################################
# # Perform semblance-based diffraction stacking with sliding time window
# # """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# # Define sliding window as two periods of the signal
# swsize = int(2/f0/dt)
# print(f"Sliding window size in samples: {swsize}")

# start_time = time()
# print("Semblance-based diffraction stacking...")
# # Run the stacking using Location class
# dstacked_semb_swin, hc_semb_swin = L.apply(frwddata, 
#                       kind="diffstack",
#                       x=gx, y=gy, z=gz,
#                       tt=tt, dt=dt, nforhc=10,
#                       stack_type="semblance", swsize=swsize)
# end_time = time()
# print(f"Computation time: {end_time - start_time} seconds")


# #%%

# ###############################################################################
# # Apply diffraction stacking to noise-contaminated data
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# # Here we apply diffraction stacking algorithms to data contaminated 
# # with noise

# ###############################################################################
# # Perform absolute-value diffraction stacking
# # """""""""""""""""""""""""""""""""""""""""""

# start_time = time()
# print("Absolute-value diffraction stacking...")
# dstacked_abs_wn, hc_abs_wn = L.apply(frwddata_wn, 
#                             kind="diffstack",
#                             x=gx, y=gy, z=gz,
#                             tt=tt, dt=dt, nforhc=10,
#                             stack_type="absolute")
# dstacked_abs_sn, hc_abs_sn = L.apply(frwddata_sn, 
#                             kind="diffstack",
#                             x=gx, y=gy, z=gz,
#                             tt=tt, dt=dt, nforhc=10,
#                             stack_type="absolute")
# dstacked_abs_rn, hc_abs_rn = L.apply(frwddata_rn, 
#                             kind="diffstack",
#                             x=gx, y=gy, z=gz,
#                             tt=tt, dt=dt, nforhc=10,
#                             stack_type="absolute")
# end_time = time()
# print(f"Computation time: {end_time - start_time} seconds")

# ###############################################################################
# # Perform squared-value diffraction stacking
# # """""""""""""""""""""""""""""""""""""""""""

# start_time = time()
# print("Squared-value diffraction stacking...")
# dstacked_sqd_wn, hc_sqd_wn = L.apply(frwddata_wn, 
#                             kind="diffstack",
#                             x=gx, y=gy, z=gz,
#                             tt=tt, dt=dt, nforhc=10,
#                             stack_type="squared")
# dstacked_sqd_sn, hc_sqd_sn = L.apply(frwddata_sn, 
#                             kind="diffstack",
#                             x=gx, y=gy, z=gz,
#                             tt=tt, dt=dt, nforhc=10,
#                             stack_type="squared")
# dstacked_sqd_rn, hc_sqd_rn = L.apply(frwddata_rn, 
#                             kind="diffstack",
#                             x=gx, y=gy, z=gz,
#                             tt=tt, dt=dt, nforhc=10,
#                             stack_type="squared")
# end_time = time()
# print(f"Computation time: {end_time - start_time} seconds")

# ###############################################################################
# # Perform semblance-based diffraction stacking with sliding time window
# # """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# # Define sliding window as two periods of the signal
# print(f"Sliding window size in samples: {swsize}")
# start_time = time()
# print("Semblance-based diffraction stacking...")
# dstacked_semb_swin_wn, hc_semb_swin_wn = L.apply(frwddata_wn, 
#                                         kind="diffstack",
#                                         x=gx, y=gy, z=gz,
#                                         tt=tt, dt=dt, nforhc=10,
#                                         stack_type="semblance", swsize=swsize)
# dstacked_semb_swin_sn, hc_semb_swin_sn = L.apply(frwddata_sn, 
#                                         kind="diffstack",
#                                         x=gx, y=gy, z=gz,
#                                         tt=tt, dt=dt, nforhc=10,
#                                         stack_type="semblance", swsize=swsize)
# dstacked_semb_swin_rn, hc_semb_swin_rn = L.apply(frwddata_rn, 
#                                         kind="diffstack",
#                                         x=gx, y=gy, z=gz,
#                                         tt=tt, dt=dt, nforhc=10,
#                                         stack_type="semblance", swsize=swsize)
# end_time = time()
# print(f"Computation time: {end_time - start_time} seconds")


#%%

###############################################################################
# Visualisation of results
# ^^^^^^^^^^^^^^^^^^^^^^^^
# Here we visualise the slices of the resulting image volume

###############################################################################
# Plot resulting image volumes from absolute-value diffraction stacking
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# The form and inclination of the location spot reflect the 
# receiver geometry, whereas focusing is related to the selected imaging 
# condition (absolute value).
# You can see how noise of different kind affects the result.

# Get the spatial limits for plotting
xlim = (min(gx),max(gx))
ylim = (min(gy),max(gy))
zlim = (min(gz),max(gz))

# Define colormap
cmap='cmc.bilbao_r'

# Define legend
crosslegend=('Intersect plane (True location)','Determined location')


# Print true locations
for isrc in np.arange(nsrc):
    print('True event {:d} hypocenter:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(isrc,sx[isrc], sy[isrc], sz[isrc]))

# Results of application to clean data:
fig,axs = locimage3d(dstacked_abs,                
                      cmap=cmap,
                      title='Location with absolute-value diffraction stacking:\nclean data',
                      x0=isx[0], y0=isy[0], z0=isz[0],
                      secondcrossloc=hc_abs,
                      crosslegend=crosslegend,
                      xlim=xlim,ylim=ylim,zlim=zlim)

print('-------------------------------------------------------')
print('Event hypocenter from absolute-value diffraction stacking for clean data:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*np.multiply(hc_abs,[dx, dy, dz])))
print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx[0], isy[0], isz[0]], hc_abs, [dx, dy, dz])))

# # Results of application to data contaminated with white noise:
# fig,axs = locimage3d(dstacked_abs_wn, 
#                      cmap=cmap,
#                      title=f"Location with absolute-value diffraction stacking:\ndata contaminated with white noise of SNR={snr_wn}",
#                      x0=isx, y0=isy, z0=isz,
#                      secondcrossloc=hc_abs_wn,
#                      crosslegend=crosslegend,
#                      xlim=xlim,ylim=ylim,zlim=zlim)
# print('-------------------------------------------------------')
# print('Event hypocenter from absolute-value diffraction stacking for data contaminated with white noise of SNR={:.1f}:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(snr_wn,*np.multiply(hc_abs_wn,[dx, dy, dz])))
# print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_abs_wn, [dx, dy, dz])))

# # Results of application to data contaminated with spiky noise:
# fig,axs = locimage3d(dstacked_abs_sn, 
#                      cmap=cmap,
#                      title=f"Location with absolute-value diffraction stacking:\ndata contaminated with spiky noise of SNR={snr_sn}",
#                      x0=isx, y0=isy, z0=isz,
#                      secondcrossloc=hc_abs_sn,
#                      crosslegend=crosslegend,
#                      xlim=xlim,ylim=ylim,zlim=zlim)
# print('-------------------------------------------------------')
# print('Event hypocenter from absolute-value diffraction stacking for data contaminated with spiky noise of SNR={:.1f}:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(snr_sn,*np.multiply(hc_abs_sn,[dx, dy, dz])))
# print('Location error: [{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_abs_sn, [dx, dy, dz])))

# # Results of application to data contaminated with ringy noise:
# fig,axs = locimage3d(dstacked_abs_rn, 
#                      cmap=cmap,
#                      title=f"Location with absolute-value diffraction stacking:\ndata contaminated with ringy noise of SNR={snr_rn}",
#                      x0=isx, y0=isy, z0=isz,
#                      secondcrossloc=hc_abs_rn,
#                      crosslegend=crosslegend,
#                      xlim=xlim,ylim=ylim,zlim=zlim)
# print('-------------------------------------------------------')
# print('Event hypocenter from absolute-value diffraction stacking for data contaminated with ringy noise of SNR={:.1f}:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(snr_rn,*np.multiply(hc_abs_rn,[dx, dy, dz])))
# print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_abs_rn, [dx, dy, dz])))

###############################################################################
# Plot resulting image volumes from squared-value diffraction stacking
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# You can see that the focusing is better when using squared values

# Print true locations
for isrc in np.arange(nsrc):
    print('True event {:d} hypocenter:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(isrc,sx[isrc], sy[isrc], sz[isrc]))
    
# Results of application to clean data:
fig,axs = locimage3d(dstacked_sqd, 
                      cmap=cmap,
                      title='Location with squared-value diffraction stacking:\nclean data',
                      x0=isx[0], y0=isy[0], z0=isz[0],
                      secondcrossloc=hc_sqd,
                      crosslegend=crosslegend,
                      xlim=xlim,ylim=ylim,zlim=zlim)

print('-------------------------------------------------------')
print('Event hypocenter from squared-value diffraction stacking for clean data:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*np.multiply(hc_sqd,[dx, dy, dz])))
print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx[0], isy[0], isz[0]], hc_sqd, [dx, dy, dz])))

# # Results of application to data contaminated with white noise:
# fig,axs = locimage3d(dstacked_sqd_wn, 
#                      cmap=cmap,
#                      title=f"Location with squared-value diffraction stacking:\ndata contaminated with white noise of SNR={snr_wn}",
#                      x0=isx, y0=isy, z0=isz,
#                      secondcrossloc=hc_sqd_wn,
#                      crosslegend=crosslegend,
#                      xlim=xlim,ylim=ylim,zlim=zlim)
# print('-------------------------------------------------------')
# print('Event hypocenter from squared-value diffraction stacking for data contaminated with white noise of SNR={:.1f}:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(snr_wn,*np.multiply(hc_sqd_wn,[dx, dy, dz])))
# print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_abs_wn, [dx, dy, dz])))

# # Results of application to data contaminated with spiky noise:
# fig,axs = locimage3d(dstacked_sqd_sn, 
#                      cmap=cmap,
#                      title=f"Location with squared-value diffraction stacking:\ndata contaminated with spiky noise of SNR={snr_sn}",
#                      x0=isx, y0=isy, z0=isz,
#                      secondcrossloc=hc_sqd_sn,
#                      crosslegend=crosslegend,
#                      xlim=xlim,ylim=ylim,zlim=zlim)
# print('-------------------------------------------------------')
# print('Event hypocenter from squared-value diffraction stacking for data contaminated with spiky noise of SNR={:.1f}:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(snr_sn,*np.multiply(hc_sqd_sn,[dx, dy, dz])))
# print('Location error: [{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_abs_sn, [dx, dy, dz])))

# # Results of application to data contaminated with ringy noise:
# fig,axs = locimage3d(dstacked_sqd_rn, 
#                      cmap=cmap,
#                      title=f"Location with squared-value diffraction stacking:\ndata contaminated with ringy noise of SNR={snr_rn}",
#                      x0=isx, y0=isy, z0=isz,
#                      secondcrossloc=hc_sqd_rn,
#                      crosslegend=crosslegend,
#                      xlim=xlim,ylim=ylim,zlim=zlim)
# print('-------------------------------------------------------')
# print('Event hypocenter from squared-value diffraction stacking for data contaminated with ringy noise of SNR={:.1f}:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(snr_rn,*np.multiply(hc_sqd_rn,[dx, dy, dz])))
# print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_sqd_rn, [dx, dy, dz])))


# ###############################################################################
# # Plot resulting image volume from semblance-based diffraction stacking
# # """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# # First result show application of semblance-based diffraction stacking without
# # sliding time window. The result has several numerical artifacts.
# # Involving sliding window helps to reduce the artifacts and improve focusing
# # but slightly increases the location error.
# # Semblance-based stacking is generally acting in the presence of noise better 
# # than absolute-based stacking.

# # Print true location
# print('True event hypocenter:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*[sx, sy, sz]))

# # Results of application to clean data:
# fig,axs = locimage3d(dstacked_semb,                       
#                      cmap=cmap,
#                      x0=isx, y0=isy, z0=isz,
#                      secondcrossloc=hc_semb,
#                      crosslegend=crosslegend,
#                      xlim=xlim,ylim=ylim,zlim=zlim)
# fig.suptitle("Location with semblance-based diffraction stacking:\nclean data")
# print('-------------------------------------------------------')
# print('Event hypocenter from semblance-based diffraction stacking for clean data:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*np.multiply(hc_semb,[dx, dy, dz])))
# print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_semb, [dx, dy, dz])))


# # Results of application to clean data:
# fig,axs = locimage3d(dstacked_semb_swin,                       
#                      cmap=cmap,
#                      x0=isx, y0=isy, z0=isz,
#                      secondcrossloc=hc_semb_swin,
#                      crosslegend=crosslegend,
#                      xlim=xlim,ylim=ylim,zlim=zlim)
# fig.suptitle(f"Location with semblance-based diffraction stacking:\nsliding window of {swsize} samples,\nclean data")
# print('-------------------------------------------------------')
# print('Event hypocenter from semblance-based diffraction stacking with sliding window of {:d} samples for clean data:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(swsize,*np.multiply(hc_semb_swin,[dx, dy, dz])))
# print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_semb_swin, [dx, dy, dz])))

# # Results of application to data contaminated with white noise:
# fig,axs = locimage3d(dstacked_semb_swin_wn,                     
#                      cmap=cmap,
#                      x0=isx, y0=isy, z0=isz,
#                      secondcrossloc=hc_semb_swin_wn,
#                      crosslegend=crosslegend,
#                      xlim=xlim,ylim=ylim,zlim=zlim)
# fig.suptitle(f"Location with semblance-based diffraction stacking:\nsliding window of {swsize} samples,\ndata contaminated with white noise of SNR={snr_wn}")
# print('-------------------------------------------------------')
# print('Event hypocenter from semblance-based diffraction stacking with sliding window of {:d} samples for data contaminated with white noise of SNR={:.1f}:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(swsize,snr_wn,*np.multiply(hc_semb_swin_wn,[dx, dy, dz])))
# print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_semb_swin_wn, [dx, dy, dz])))

# # Results of application to data contaminated with spiky noise:
# fig,axs = locimage3d(dstacked_semb_swin_sn,                     
#                      cmap=cmap,
#                      x0=isx, y0=isy, z0=isz,
#                      secondcrossloc=hc_semb_swin_sn,
#                      crosslegend=crosslegend,
#                      xlim=xlim,ylim=ylim,zlim=zlim)
# fig.suptitle(f"Location with semblance-based diffraction stacking:\nsliding window of {swsize} samples,\ndata contaminated with spiky noise of SNR={snr_sn}")
# print('-------------------------------------------------------')
# print('Event hypocenter from semblance-based diffraction stacking with sliding window of {:d} samples for data contaminated with spiky noise of SNR={:.1f}:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(swsize,snr_sn,*np.multiply(hc_semb_swin_sn,[dx, dy, dz])))
# print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_semb_swin_sn, [dx, dy, dz])))

# # Results of application to data contaminated with ringy noise:
# fig,axs = locimage3d(dstacked_semb_swin_rn,                     
#                      cmap=cmap,
#                      x0=isx, y0=isy, z0=isz,
#                      secondcrossloc=hc_semb_swin_rn,
#                      crosslegend=crosslegend,
#                      xlim=xlim,ylim=ylim,zlim=zlim)
# fig.suptitle(f"Location with semblance-based diffraction stacking:\nsliding window of {swsize} samples,\ndata contaminated with ringy noise of SNR={snr_rn}")
# print('-------------------------------------------------------')
# print('Event hypocenter from semblance-based diffraction stacking with sliding window of {:d} samples for data contaminated with ringy noise of SNR={:.1f}:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(swsize,snr_rn,*np.multiply(hc_semb_swin_rn,[dx, dy, dz])))
# print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_semb_swin_rn, [dx, dy, dz])))
