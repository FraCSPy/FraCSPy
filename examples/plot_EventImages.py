r"""
Event Volume Plotting 
=====================
This example shows how to visualise microseismic event characteristics, i.e., semblence,
as a volume.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from fracspy.visualisation.eventimages import locimage3d
from fracspy.location.utils import get_max_locs


###############################################################################
# Setting up model
# ^^^^^^^^^^^^^^^^

# Define subsurface
nx, ny, nz = 60, 65, 75
dx = dy = dz = 5
x = np.arange(0,nx)*dx
y = np.arange(0,ny)*dy
z = np.arange(0,nz)*dz

source_location = 25, 30, 50  # x,y,z location 

###############################################################################
# Simple Example - Clean source location
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


background_noise = gaussian_filter(100*np.random.rand(nx,ny,nz), sigma=1, radius=2)

microseismic_image = np.zeros_like(background_noise)
microseismic_image[source_location] = 40
microseismic_image = gaussian_filter(microseismic_image, sigma=2, radius=5)


###############################################################################
# Perform plotting, here we are going to intersect the volume at the known 
# source location. We also provide the limits for the model space using 
# the xlim, ylim, and zlim parameters. 

fig, axs = locimage3d(microseismic_image,
                      x0=int(np.round(source_location[0])),
                      y0=int(np.round(source_location[1])),
                      z0=int(np.round(source_location[2])),
                      xlim=[x[0],x[-1]],
                      ylim=[y[0],y[-1]],
                      zlim=[z[0],z[-1]],
                      clipval=[0,1])
plt.tight_layout()


###############################################################################
# Noisy Example - Artifact present in image 
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

artifact_location_center = 25, 35, 25  # x,y,z location 
artifact_image = np.zeros_like(microseismic_image)
artifact_image[artifact_location_center] = 125
artifact_image = gaussian_filter(artifact_image, sigma=3, radius=4)

microseismic_image_noisy = microseismic_image.copy() + artifact_image

###############################################################################
# If we perform slicing at the location of the maxima of the image volume we
# will now be slicing over the artifact location, as opposed to the true
# source location

max_loc, _ = get_max_locs(microseismic_image_noisy, n_max=5)

fig, axs = locimage3d(microseismic_image_noisy,
                      x0=int(np.round(max_loc[0])),
                      y0=int(np.round(max_loc[1])),
                      z0=int(np.round(max_loc[2])),
                      xlim=[x[0],x[-1]],
                      ylim=[y[0],y[-1]],
                      zlim=[z[0],z[-1]],
                      clipval=[0,1])
plt.tight_layout()
