r"""

Diffraction Stacking Localisation - Simple scenario.
====================================================
This tutorial illustrates how to perform source localisation using 
diffraction stacking based on semblance. 

We consider here a simple scenario of a homogeneous subsurface model and 
a point microseismic source with a uniform radiation pattern (explosion-like).
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

Diffraction stacking
^^^^^^^^^^^^^^^^^^^^
The subsurface volume is discretised and each grid node is considered to be a
potential source position or a so-called image point.
In other words, each image point represents a possible diffraction point from 
which seismic energy radiates. The term ‘diffraction stacking’ dates back to 
the works of Claerbout (1971, 1985) and Timoshin (1972). 
The concept was initially related to seismic migration and imaging of 
reflectors as a set of diffraction points (in the context of the exploding 
reflector principle). Here, the diffraction stacking is related to imaging of 
real excitation sources.

Waveforms from all traces are stacked along the calculated moveouts. In other 
words, one should apply event moveout (EMO) correction 
(e.g., Trojanowski and Eisner, 2016) to microseismic data and stack the traces.
As the source origin time of a seismic event is unknown, in order to get the 
image function value :math:`\mathbf{F}(\mathbf{r},t)`, the stacking must be 
performed for all possible origin times :math:`t`:   
    
.. math::
        \mathbf{F}(\mathbf{r},t) = \left|\sum_{R=1}^{N_R} A_R 
        \left(t + T_R(\mathbf{r})\right)\right|,

where :math:`\mathbf{r}` is a vector that defines a spatial position 
:math:`(x, y, z)` of the image point, :math:`T_R(\mathbf{r})` is the P-wave 
traveltime from the image point :math:`\mathbf{r}` to a receiver :math:`R`,
:math:`N_R` is a number of receivers, and :math:`A_R` is the observed waveform 
at the receiver :math:`R` (e.g., Anikiev, 2015).
The term 

.. math::
        A_R^{EMO}(t,\mathbf{r}) = A_R \left(t + T_R(\mathbf{r})\right)

represents the EMO-corrected data.

Stacking of the modulus of the EMO-corrected data is needed as it treats 
equally positive and negative phases of the stacked amplitudes and let the 
positive maximum of the resulting image function :math:`\mathbf{F}_0(\mathbf{r},t)`
work as an indicator of a correct location and origin time of an event.

However, simple stacking of the absolute values to get 
:math:`\mathbf{F}(\mathbf{r},t)` can be further improved by using a 
semblance-based approach. The semblance is a coherency measure and can be 
understood as the ratio of the signal energy to the energy of the signal and 
noise in the selected window (Neidell and Taner, 1971).


For the EMO-corrected data, for image point :math:`\mathbf{r}` and a given time
step :math:`t` the semblance-based image function :math:`S(\mathbf{r},t)` 
can be calculated by:
    
.. math::
        S(\mathbf{r},t) = \frac{\left[\sum_{R=1}^{N_R} A_R^{EMO}(t,\mathbf{r})\right]^2}
        {N_R \sum_{R=1}^{N_R} \left[A_R^{EMO}(t,\mathbf{r})\right]^2}

The main advantage of the semblance-based imaging is its ability to identify 
and suppress high stack values that result from high noise on only a few 
receivers, which is a common problem for surface monitoring 
(Trojanowski and Eisner, 2016). The semblance reaches its maximum value of 1 
for identical signals on all traces and the minimum value of 0 for a zero sum 
of the samples. High stacks resulting from high noise on individual receivers
have a low semblance value and in contrast to microseismic events that have 
consistent amplitude arrivals across the array (provided that EMO correction is 
done with a suitable velocity model).


        
.. math::
        S_w(\mathbf{r},t) = \frac{\sum_{R=1}^{N_R} \left[\sum_{i=1}^{M} u_i^{\text{EMO}(k)}\right]^2}
        {M \sum_{k=t-W}^{t+W} \left[\sum_{i=1}^{M} u_i^{\text{EMO}(k)}\right]^2}

Note that neither stack of absolute values, nor semblance-based stack take 
into account the potential polarity changes of the signal, i.e. they both are 
best suitable for localisation of events with single polarity (e.g. an 
explosion or an implosion).


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

# #%%

# ###############################################################################
# # Load all necessary packages
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^

# import numpy as np
# import matplotlib.pyplot as plt

# from pylops.utils import dottest
# from pylops.utils.wavelets import ricker

# # Import modelling utils
# from fracspy.modelling.kirchhoff import Kirchhoff

# # Import diffraction stacking utils
# from fracspy.location import Location
# from fracspy.location.utils import dist2rec

# # Import visualisation utils
# from fracspy.visualisation.traceviz import traceimage
# from fracspy.visualisation.eventimages import locimage3d

# # Deal with warnings (for a cleaner code)
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning)

# # Track computation time
# from time import time 

# #%%

# ###############################################################################
# # Setup
# # ^^^^^
# # Here we setup the parameters of the velocity model, geometry of receivers and 
# # microseismic source for forward modelling

# ###############################################################################
# # Velocity Model
# # """"""""""""""

# nx, ny, nz = 50, 50, 50
# dx, dy, dz = 4, 4, 4
# x, y, z = np.arange(nx)*dx, np.arange(ny)*dy, np.arange(nz)*dz

# v0 = 1000 # initial velocity
# vel = np.ones([nx,ny,nz])*v0

# print(f"Velocity model shape: {vel.shape}")

# ###############################################################################
# # Receivers
# # """""""""

# dr_xyz = 4*dx

# grid_rx_locs = np.arange(dx, (dx*nx)-dx, dr_xyz)
# grid_ry_locs = np.arange(dy, (dy*ny)-dy, dr_xyz)

# rx, ry, rz = np.meshgrid(grid_rx_locs,
#                          grid_ry_locs,
#                          dz) 
# recs = np.vstack((rx.flatten(), ry.flatten(), rz.flatten()))
# nr = recs.shape[1]

# print(f"Receiver array shape: {recs.shape}")

# ###############################################################################
# # Microseismic sources
# # """"""""""""""""""""

# sx, sy, sz = [nx//4, ny//2, nz//2]
# microseismic = np.zeros((nx, ny, nz))
# microseismic[sx, sy, sz] = 1.


# #%%

# ###############################################################################
# # Generate synthetic data
# # ^^^^^^^^^^^^^^^^^^^^^^^
# # 

# nt = 81 # number of time steps
# dt = 0.004 # time step
# f0 = 20 # Central frequency
# t = np.arange(nt) * dt # time vector

# ###############################################################################
# # Create signal wavelet
# # """""""""""""""""""""
# wav, wavt, wavc = ricker(t[:41], f0=f0)

# ###############################################################################
# # Initialise operator
# # """""""""""""""""""

# Op = Kirchhoff(z=z, 
#                x=x, 
#                y=y, 
#                t=t, 
#                recs=recs, 
#                vel=vel, 
#                wav=wav, 
#                wavcenter=wavc, 
#                mode='eikonal', 
#                engine='numba')

# ###############################################################################
# # Check operator with dottest
# # """""""""""""""""""""""""""
# # This test can help to detect errors in the operator implementation.

# _ = dottest(Op, verb=True)

# ###############################################################################
# # Forward modelling
# # """""""""""""""""

# frwddata_1d = Op @ microseismic.flatten().squeeze()
# frwddata = frwddata_1d.reshape(nr,nt)


# #%%

# ###############################################################################
# # Apply diffraction stacking
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^
# # Here we apply diffraction stacking algorithm based on semblance to get the
# # image volume and determine location from the maximum of this volume.

# ###############################################################################
# # Define location class using grid vectors
# # """"""""""""""""""""""""""""""""""""""""
# # Use the original velocity model grid for location (the grid can be different)

# gx = x
# gy = y
# gz = z

# # Set up the location class

# L = Location(gx, gy, gz)

# ###############################################################################
# # Prepare traveltimes
# # """""""""""""""""""

# tt = 1 / v0*dist2rec(recs,gx,gy,gz)
# print(f"Traveltime array shape: {tt.shape}")

# ###############################################################################
# # Perform standard semblance-based diffraction stacking
# # """""""""""""""""""""""""""""""""""""""""""""""""""""
# start_time = time()
# print("Diffraction stacking...")
# # Run the stacking using Location class
# dstacked, hc = L.apply(frwddata, 
#                       kind="semblancediffstack", 
#                       tt=tt, dt=dt, nforhc=10)

# # One can also run it like that:
# # dstacked, hc = fracspy.location.migration.semblancediffstack(frwddata,
# #                                                              n_xyz=[len(gx),len(gy),len(gz)], 
# #                                                              tt=tt, 
# #                                                              dt=dt, 
# #                                                              nforhc=10)

# end_time = time()
# print(f"Computation time: {end_time - start_time} seconds")

# print('True event hypocenter:', [sx, sy, sz])
# print('Event hypocenter from diffraction stacking:', hc.tolist())
# print('Location error:', [x - y for x, y in zip([sx, sy, sz], hc.tolist())])

# #%%

# ###############################################################################
# # Visualisation of results and data
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# # Here we visualise the slices of the resulting image volume as well as 
# # modelled input data and receiver geometry

# ###############################################################################
# # Plot resulting image volume
# # """""""""""""""""""""""""""

# fig,axs = locimage3d(dstacked, x0=sx, y0=sy, z0=sz)

# ###############################################################################
# # Plot modelled data
# # """"""""""""""""""

# fig, ax = traceimage(frwddata, climQ=99.99)
# ax.set_title('Point Receivers')
# ax.set_ylabel('Time steps')
# fig = ax.get_figure()
# fig.set_size_inches(10, 3)  # set size in inches

# ###############################################################################
# # Plot wavelet
# # """"""""""""

# fig, ax = plt.subplots(1, 1)
# ax.plot(wav)
# ax.set_xlabel('Time steps')
# ax.set_ylabel('Amplitude')
# fig.set_size_inches(10, 3)  # set size in inches

# ###############################################################################
# # Plot receiver geometry
# # ^^^^^^^^^^^^^^^^^^^^^^

# fig, ax = plt.subplots(1, 1)
# fig.set_size_inches(8, 8)  # set size in inches
# ax.set_aspect('equal')
# ax.scatter(recs[0],recs[1])
# ax.scatter(sx*dx,sy*dy, marker='*')
# ax.set_title('Receiver Geometry: map view')
# ax.legend(['Receivers', 'Source'],loc='upper right')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
