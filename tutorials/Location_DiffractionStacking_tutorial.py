r"""

Diffraction Stacking Localisation - Simple scenario
===================================================
This tutorial illustrates how to perform source localisation using diffraction stacking based on semblance. 

We consider here a simple scenario of a homogeneous subsurface model and a point microseismic source with a uniform radiation pattern (explosion-like).
We also consider only P-waves for simplicity here, and single-component receivers.

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
The input data waveforms are computed with the help of PyLops operator which involves finite-difference (FD) modelling.

See more information here:
https://pylops.readthedocs.io

Diffraction stacking
^^^^^^^^^^^^^^^^^^^^
The subsurface volume is discretised and each grid node is considered to be a potential source position or a so-called image point.
In other words, each image point represents a possible diffraction point from which seismic energy radiates. 
The term ‘diffraction stacking’ dates back to the works of Claerbout (1971, 1985) and Timoshin (1972). 
The concept was initially related to seismic migration and imaging of reflectors as a set of diffraction points (in the context of the exploding reflector principle). 
Here, the diffraction stacking is related to imaging of real excitation sources.

Waveforms from all traces are stacked along the calculated moveouts. 
In other words, one should apply event moveout (EMO) correction (e.g., Trojanowski and Eisner, 2016) to microseismic data and stack the traces.
As the source origin time of a seismic event is unknown, in order to get the image function value :math:`F(\mathbf{r},t)`, the stacking must be performed for all possible origin times :math:`t`:   
    
.. math::
        F(\mathbf{r},t) = \left|\sum_{R=1}^{N_R} A_R 
        \left(t + T_R(\mathbf{r})\right)\right|,

where :math:`\mathbf{r}` is a vector that defines a spatial position :math:`(x, y, z)` of the image point, 
:math:`T_R(\mathbf{r})` is the P-wave traveltime from the image point :math:`\mathbf{r}` to a receiver :math:`R`,
:math:`N_R` is a number of receivers, and :math:`A_R` is the observed waveform at the receiver :math:`R` (e.g., Anikiev, 2015).

The term 

.. math::
        A_R^{EMO}(t,\mathbf{r}) = A_R \left(t + T_R(\mathbf{r})\right)

represents the EMO-corrected data.

An absolute value of the stack of the EMO-corrected data is used in order to treat equally positive and negative phases of the stacked amplitudes 
and let the positive maximum of the resulting 4D image function  :math:`\mathbf{F}_0(\mathbf{r},t)` work as an indicator of a correct location 
and origin time of an event (Anikiev et al. 2014).

Another option would be to stack squared values:
        
.. math::
        E(\mathbf{r},t) = \left(\sum_{R=1}^{N_R} A_R 
        \left(t + T_R(\mathbf{r})\right)\right)^2,
        
This simple modification comes from idea of summation of signal energy rather than the amplitudes.

However, simple stacking of the absolute or squared values can be further improved, e.g., using a semblance-based approach. 
The semblance is a coherency or similarity measure and can be understood as the ratio of the total energy (the square of sum of amplitudes) 
to the energy of individual traces (the sum of squares) (Neidell and Taner, 1971).

For the EMO-corrected data, for image point :math:`\mathbf{r}` and a given time step :math:`t` the semblance-based image function :math:`S(\mathbf{r},t)` can be calculated by:
        
.. math::
        S(\mathbf{r},t) = \frac{\left[\sum_{R=1}^{N_R} A_R^{EMO}(t,\mathbf{r})\right]^2}
        {N_R \sum_{R=1}^{N_R} \left[A_R^{EMO}(t,\mathbf{r})\right]^2}    

The main advantage of the semblance-based imaging is its ability to identify and suppress high stack values that result from high noise on only a few receivers, 
which is a common problem for surface monitoring (Trojanowski and Eisner, 2016). 
The semblance reaches its maximum value of 1 for identical signals on all traces and the minimum value of 0 for a zero sum of the samples. 
High stacks resulting from high noise on individual receivers have a low semblance value and in contrast to microseismic events 
that have consistent amplitude arrivals across the array (provided that EMO correction is done with a suitable velocity model).

In order to suppress the effect of noise even better, it is possible to extend 
the semblance-based approach by introducing a sliding time window :math:`W` over which the energy measures are summed:

.. math::
        S_W(\mathbf{r},t,W) = \frac{\sum_{k=it-W}^{it+W}\left[\sum_{R=1}^{N_R} A_R^{EMO}(t,\mathbf{r})\right]^2}
        {N_R \sum_{k=it-W}^{it+W}\sum_{R=1}^{N_R} \left[A_R^{EMO}(t,\mathbf{r})\right]^2}

where :math:`k` an index of the time-discretised signal within a sliding time interval 
consisting of the :math:`2W + 1` samples, and :math:`it` is the index of time :math:`t` (Trojanowski and Eisner, 2016).

Finally, assuming that there is only one event in the given time window, 
the actual 3D volume of the image function can obtained by computing a maximum of the 4D image function.

All approaches listed above are implemented in :py:class:`fracspy.location.migration.diffstack`.

Note that neither of the approaches described here take into account the potential polarity changes of the signal.
Therefore, seismograms generated by shear source mechanisms (with positive and negative P-wave and S-wave polarizations) 
cause these methods to fail to produce high stack values at the true origin time 
and location because of the destructive interference of the signal (e.g. Anikiev et al., 2014, Trojanowski and Eisner, 2016).

We discuss this issue and stacking with polarity correction in :ref:`sphx_glr_tutorials_Location_DSMTI_tutorial.py`.

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

Claerbout, J. F. (1971). Toward a unified theory of reflector mapping. 
Geophysics, 36(3), 467-481. https://doi.org/10.1190/1.1440185

Claerbout, J. (1985). Imaging the earth's interior. Oxford, England: Blackwell 
Scientific Publications. https://sepwww.stanford.edu/sep/prof/iei2/

Neidell, N. S., & Taner, M. T. (1971). SEMBLANCE AND OTHER COHERENCY MEASURES 
FOR MULTICHANNEL DATA. Geophysics, 36(3), 482–497. 
https://doi.org/10.1190/1.1440186

Timoshin Yu. V. (1972). Fundamentals of diffraction conversion of seismic 
recordings. Moscow: Nedra. [In Russian].

Trojanowski, J., & Eisner, L. (2016). Comparison of migration‐based location 
and detection methods for microseismic events. Geophysical Prospecting, 65(1), 
47–63. https://doi.org/10.1111/1365-2478.12366
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

isx, isy, isz = [nx//4, ny//2, nz//2]
microseismic = np.zeros((nx, ny, nz))
microseismic[isx, isy, isz] = 1.
sx, sy, sz = isx*dx, isx*dy, isz*dz 

#%%

###############################################################################
# Generate synthetic data
# ^^^^^^^^^^^^^^^^^^^^^^^
# 
start_time = time()
print("Generating synthetic data...")
nt = 81 # number of time steps
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

###############################################################################
# Forward modelling
# """""""""""""""""
# Apply operator to model data
frwddata_1d = Op @ microseismic.flatten().squeeze()
frwddata = frwddata_1d.reshape(nr,nt)

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

###############################################################################
# Plot modelled data contaminated with spiky noise
# """"""""""""""""""""""""""""""""""""""""""""""""

fig, ax = traceimage(frwddata_sn, climQ=99.99)
ax.set_title(f"Modelled data contaminated with spiky noise of SNR={snr_sn}")
ax.set_ylabel('Time steps')
fig = ax.get_figure()
fig.set_size_inches(10, 3)  # set size in inches

###############################################################################
# Plot modelled data contaminated with ringy noise
# """"""""""""""""""""""""""""""""""""""""""""""""

fig, ax = traceimage(frwddata_rn, climQ=99.99)
ax.set_title(f"Modelled data contaminated with ringy noise of SNR={snr_rn}")
ax.set_ylabel('Time steps')
fig = ax.get_figure()
fig.set_size_inches(10, 3)  # set size in inches

###############################################################################
# Plot receiver geometry
# """"""""""""""""""""""

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
                      tt=tt, dt=dt, nforhc=10,
                      stack_type="absolute")
end_time = time()
print(f"Computation time: {end_time - start_time} seconds")

###############################################################################
# Perform squared-value diffraction stacking
# """""""""""""""""""""""""""""""""""""""""""

start_time = time()
print("Squared-value diffraction stacking...")
dstacked_sqd, hc_sqd = L.apply(frwddata, 
                      kind="diffstack",                            
                      tt=tt, dt=dt, nforhc=10,
                      stack_type="squared")
end_time = time()
print(f"Computation time: {end_time - start_time} seconds")

###############################################################################
# Perform semblance-based diffraction stacking without sliding time window
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

start_time = time()
print("Semblance-based diffraction stacking...")
# Run the stacking using Location class
dstacked_semb, hc_semb = L.apply(frwddata, 
                      kind="diffstack",                       
                      tt=tt, dt=dt, nforhc=10,
                      stack_type="semblance")
end_time = time()
print(f"Computation time: {end_time - start_time} seconds")

###############################################################################
# Perform semblance-based diffraction stacking with sliding time window
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Define sliding window as two periods of the signal
swsize = int(2/f0/dt)
print(f"Sliding window size in samples: {swsize}")

start_time = time()
print("Semblance-based diffraction stacking...")
# Run the stacking using Location class
dstacked_semb_swin, hc_semb_swin = L.apply(frwddata, 
                      kind="diffstack",   
                      tt=tt, dt=dt, nforhc=10,
                      stack_type="semblance", swsize=swsize)
end_time = time()
print(f"Computation time: {end_time - start_time} seconds")


#%%

###############################################################################
# Apply diffraction stacking to noise-contaminated data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here we apply diffraction stacking algorithms to data contaminated 
# with noise

###############################################################################
# Perform absolute-value diffraction stacking
# """""""""""""""""""""""""""""""""""""""""""

start_time = time()
print("Absolute-value diffraction stacking...")
dstacked_abs_wn, hc_abs_wn = L.apply(frwddata_wn, 
                            kind="diffstack", 
                            tt=tt, dt=dt, nforhc=10,
                            stack_type="absolute")
dstacked_abs_sn, hc_abs_sn = L.apply(frwddata_sn, 
                            kind="diffstack", 
                            tt=tt, dt=dt, nforhc=10,
                            stack_type="absolute")
dstacked_abs_rn, hc_abs_rn = L.apply(frwddata_rn, 
                            kind="diffstack", 
                            tt=tt, dt=dt, nforhc=10,
                            stack_type="absolute")
end_time = time()
print(f"Computation time: {end_time - start_time} seconds")

###############################################################################
# Perform squared-value diffraction stacking
# """""""""""""""""""""""""""""""""""""""""""

start_time = time()
print("Squared-value diffraction stacking...")
dstacked_sqd_wn, hc_sqd_wn = L.apply(frwddata_wn, 
                            kind="diffstack", 
                            tt=tt, dt=dt, nforhc=10,
                            stack_type="squared")
dstacked_sqd_sn, hc_sqd_sn = L.apply(frwddata_sn, 
                            kind="diffstack", 
                            tt=tt, dt=dt, nforhc=10,
                            stack_type="squared")
dstacked_sqd_rn, hc_sqd_rn = L.apply(frwddata_rn, 
                            kind="diffstack", 
                            tt=tt, dt=dt, nforhc=10,
                            stack_type="squared")
end_time = time()
print(f"Computation time: {end_time - start_time} seconds")

###############################################################################
# Perform semblance-based diffraction stacking with sliding time window
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Define sliding window as two periods of the signal
print(f"Sliding window size in samples: {swsize}")
start_time = time()
print("Semblance-based diffraction stacking...")
dstacked_semb_swin_wn, hc_semb_swin_wn = L.apply(frwddata_wn, 
                                        kind="diffstack", 
                                        tt=tt, dt=dt, nforhc=10,
                                        stack_type="semblance", swsize=swsize)
dstacked_semb_swin_sn, hc_semb_swin_sn = L.apply(frwddata_sn, 
                                        kind="diffstack", 
                                        tt=tt, dt=dt, nforhc=10,
                                        stack_type="semblance", swsize=swsize)
dstacked_semb_swin_rn, hc_semb_swin_rn = L.apply(frwddata_rn, 
                                        kind="diffstack", 
                                        tt=tt, dt=dt, nforhc=10,
                                        stack_type="semblance", swsize=swsize)
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
# The form and inclination of the location spot reflect the 
# receiver geometry, whereas focusing is related to the selected imaging 
# condition (absolute value).
# You can see how noise of different kind affects the result.

# Get the spatial limits for plotting
xlim = (min(gx),max(gx))
ylim = (min(gy),max(gy))
zlim = (min(gz),max(gz))

# Print true location
print('True event hypocenter:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*[sx, sy, sz]))

# Results of application to clean data:
fig,axs = locimage3d(dstacked_abs, 
                      title='Location with absolute-value diffraction stacking:\nclean data',
                      x0=isx, y0=isy, z0=isz,
                      xlim=xlim,ylim=ylim,zlim=zlim)

print('-------------------------------------------------------')
print('Event hypocenter from absolute-value diffraction stacking for clean data:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*np.multiply(hc_abs,[dx, dy, dz])))
print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_abs, [dx, dy, dz])))

# Results of application to data contaminated with white noise:
fig,axs = locimage3d(dstacked_abs_wn, 
                     title=f"Location with absolute-value diffraction stacking:\ndata contaminated with white noise of SNR={snr_wn}",
                     x0=isx, y0=isy, z0=isz,
                     xlim=xlim,ylim=ylim,zlim=zlim)
print('-------------------------------------------------------')
print('Event hypocenter from absolute-value diffraction stacking for data contaminated with white noise of SNR={:.1f}:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(snr_wn,*np.multiply(hc_abs_wn,[dx, dy, dz])))
print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_abs_wn, [dx, dy, dz])))

# Results of application to data contaminated with spiky noise:
fig,axs = locimage3d(dstacked_abs_sn, 
                     title=f"Location with absolute-value diffraction stacking:\ndata contaminated with spiky noise of SNR={snr_sn}",
                     x0=isx, y0=isy, z0=isz,
                     xlim=xlim,ylim=ylim,zlim=zlim)
print('-------------------------------------------------------')
print('Event hypocenter from absolute-value diffraction stacking for data contaminated with spiky noise of SNR={:.1f}:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(snr_sn,*np.multiply(hc_abs_sn,[dx, dy, dz])))
print('Location error: [{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_abs_sn, [dx, dy, dz])))

# Results of application to data contaminated with ringy noise:
fig,axs = locimage3d(dstacked_abs_rn, 
                     title=f"Location with absolute-value diffraction stacking:\ndata contaminated with ringy noise of SNR={snr_rn}",
                     x0=isx, y0=isy, z0=isz,
                     xlim=xlim,ylim=ylim,zlim=zlim)
print('-------------------------------------------------------')
print('Event hypocenter from absolute-value diffraction stacking for data contaminated with ringy noise of SNR={:.1f}:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(snr_rn,*np.multiply(hc_abs_rn,[dx, dy, dz])))
print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_abs_rn, [dx, dy, dz])))

###############################################################################
# Plot resulting image volumes from squared-value diffraction stacking
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# You can see that the focusing is better when using squared values

# Print true location
print('True event hypocenter:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*[sx, sy, sz]))

# Results of application to clean data:
fig,axs = locimage3d(dstacked_sqd, 
                      title='Location with squared-value diffraction stacking:\nclean data',
                      x0=isx, y0=isy, z0=isz,
                      xlim=xlim,ylim=ylim,zlim=zlim)

print('-------------------------------------------------------')
print('Event hypocenter from squared-value diffraction stacking for clean data:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*np.multiply(hc_sqd,[dx, dy, dz])))
print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_sqd, [dx, dy, dz])))

# Results of application to data contaminated with white noise:
fig,axs = locimage3d(dstacked_sqd_wn, 
                     title=f"Location with squared-value diffraction stacking:\ndata contaminated with white noise of SNR={snr_wn}",
                     x0=isx, y0=isy, z0=isz,
                     xlim=xlim,ylim=ylim,zlim=zlim)
print('-------------------------------------------------------')
print('Event hypocenter from squared-value diffraction stacking for data contaminated with white noise of SNR={:.1f}:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(snr_wn,*np.multiply(hc_sqd_wn,[dx, dy, dz])))
print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_abs_wn, [dx, dy, dz])))

# Results of application to data contaminated with spiky noise:
fig,axs = locimage3d(dstacked_sqd_sn, 
                     title=f"Location with squared-value diffraction stacking:\ndata contaminated with spiky noise of SNR={snr_sn}",
                     x0=isx, y0=isy, z0=isz,
                     xlim=xlim,ylim=ylim,zlim=zlim)
print('-------------------------------------------------------')
print('Event hypocenter from squared-value diffraction stacking for data contaminated with spiky noise of SNR={:.1f}:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(snr_sn,*np.multiply(hc_sqd_sn,[dx, dy, dz])))
print('Location error: [{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_abs_sn, [dx, dy, dz])))

# Results of application to data contaminated with ringy noise:
fig,axs = locimage3d(dstacked_sqd_rn, 
                     title=f"Location with squared-value diffraction stacking:\ndata contaminated with ringy noise of SNR={snr_rn}",
                     x0=isx, y0=isy, z0=isz,
                     xlim=xlim,ylim=ylim,zlim=zlim)
print('-------------------------------------------------------')
print('Event hypocenter from squared-value diffraction stacking for data contaminated with ringy noise of SNR={:.1f}:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(snr_rn,*np.multiply(hc_sqd_rn,[dx, dy, dz])))
print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_sqd_rn, [dx, dy, dz])))


###############################################################################
# Plot resulting image volume from semblance-based diffraction stacking
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# First result show application of semblance-based diffraction stacking without
# sliding time window. The result has several numerical artifacts.
# Involving sliding window helps to reduce the artifacts and improve focusing
# but slightly increases the location error.
# Semblance-based stacking is generally acting in the presence of noise better 
# than absolute-based stacking.

# Print true location
print('True event hypocenter:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*[sx, sy, sz]))

# Results of application to clean data:
fig,axs = locimage3d(dstacked_semb,                       
                      x0=isx, y0=isy, z0=isz,
                      xlim=xlim,ylim=ylim,zlim=zlim)
fig.suptitle(f"Location with semblance-based diffraction stacking:\nclean data")
print('-------------------------------------------------------')
print('Event hypocenter from semblance-based diffraction stacking for clean data:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*np.multiply(hc_semb,[dx, dy, dz])))
print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_semb, [dx, dy, dz])))


# Results of application to clean data:
fig,axs = locimage3d(dstacked_semb_swin,                       
                      x0=isx, y0=isy, z0=isz,
                      xlim=xlim,ylim=ylim,zlim=zlim)
fig.suptitle(f"Location with semblance-based diffraction stacking:\nsliding window of {swsize} samples,\nclean data")
print('-------------------------------------------------------')
print('Event hypocenter from semblance-based diffraction stacking with sliding window of {:d} samples for clean data:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(swsize,*np.multiply(hc_semb_swin,[dx, dy, dz])))
print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_semb_swin, [dx, dy, dz])))

# Results of application to data contaminated with white noise:
fig,axs = locimage3d(dstacked_semb_swin_wn,                     
                     x0=isx, y0=isy, z0=isz,
                     xlim=xlim,ylim=ylim,zlim=zlim)
fig.suptitle(f"Location with semblance-based diffraction stacking:\nsliding window of {swsize} samples,\ndata contaminated with white noise of SNR={snr_wn}")
print('-------------------------------------------------------')
print('Event hypocenter from semblance-based diffraction stacking with sliding window of {:d} samples for data contaminated with white noise of SNR={:.1f}:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(swsize,snr_wn,*np.multiply(hc_semb_swin_wn,[dx, dy, dz])))
print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_semb_swin_wn, [dx, dy, dz])))

# Results of application to data contaminated with spiky noise:
fig,axs = locimage3d(dstacked_semb_swin_sn,                     
                     x0=isx, y0=isy, z0=isz,
                     xlim=xlim,ylim=ylim,zlim=zlim)
fig.suptitle(f"Location with semblance-based diffraction stacking:\nsliding window of {swsize} samples,\ndata contaminated with spiky noise of SNR={snr_sn}")
print('-------------------------------------------------------')
print('Event hypocenter from semblance-based diffraction stacking with sliding window of {:d} samples for data contaminated with spiky noise of SNR={:.1f}:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(swsize,snr_sn,*np.multiply(hc_semb_swin_sn,[dx, dy, dz])))
print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_semb_swin_sn, [dx, dy, dz])))

# Results of application to data contaminated with ringy noise:
fig,axs = locimage3d(dstacked_semb_swin_rn,                     
                     x0=isx, y0=isy, z0=isz,
                     xlim=xlim,ylim=ylim,zlim=zlim)
fig.suptitle(f"Location with semblance-based diffraction stacking:\nsliding window of {swsize} samples,\ndata contaminated with ringy noise of SNR={snr_rn}")
print('-------------------------------------------------------')
print('Event hypocenter from semblance-based diffraction stacking with sliding window of {:d} samples for data contaminated with ringy noise of SNR={:.1f}:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(swsize,snr_rn,*np.multiply(hc_semb_swin_rn,[dx, dy, dz])))
print('Location error:\n[{:.2f} m, {:.2f} m, {:.2f} m]'.format(*get_location_misfit([isx, isy, isz], hc_semb_swin_rn, [dx, dy, dz])))
