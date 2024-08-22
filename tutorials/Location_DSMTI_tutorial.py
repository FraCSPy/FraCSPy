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
