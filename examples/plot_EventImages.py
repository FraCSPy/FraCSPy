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

###############################################################################
# Simple Example - Clean source location
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
nx, ny, nz = 60, 65, 75
background_noise = gaussian_filter(10*np.random.random([nx,ny,nz]), sigma=1, radius=3)

source_location = 25, 30, 40  # x,y,z location 
microseismic_image = np.zeros_like(background_noise)
microseismic_image[source_location] = 25
microseismic_image = gaussian_filter(microseismic_image, sigma=2, radius=5)

sx = int(np.round(source_location[0]))
sy = int(np.round(source_location[1]))
sz = int(np.round(source_location[2]))

fig, axs = locimage3d(microseismic_image,
                      x0=int(np.round(source_location[0])),
                      y0=int(np.round(source_location[1])),
                      z0=int(np.round(source_location[2])),
                      clipval=[0,1])
plt.tight_layout()


###############################################################################
# Noisy Example - Artifact present in image 
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

artifact_location_center = 25, 35, 25  # x,y,z location 
artifact_image = np.zeros_like(microseismic_image)
artifact_image[artifact_location_center] = 50
artifact_image = gaussian_filter(artifact_image, sigma=3, radius=4)

microseismic_image_noisy = microseismic_image.copy() + artifact_image

###############################################################################
# If we perform slicing at the location of the maxima of the image volume we
# will now be slicing over the artifact location, as opposed to the true
# source location

fig, axs = locimage3d(microseismic_image_noisy,
                      x0=int(np.round(artifact_location_center[0])),
                      y0=int(np.round(artifact_location_center[1])),
                      z0=int(np.round(artifact_location_center[2])),
                      clipval=[0,1])
plt.tight_layout()

###############################################################################
# Instead, if we know the area where we expect the microseismicity to occur we
# can crop the image around this location using the xlim, ylim and zlim parameters.
#
# Note, this is a good solution to allow zooming in on specific areas, however 
# if you have an area of interest defined it is preferable to use that in the 
# localisation/MTI procedures and then your resulting inverted volume will already 
# focus on that area. For more information, see the MTWI tutorial.

sx = int(np.round(source_location[0]))
sy = int(np.round(source_location[1]))
sz = int(np.round(source_location[2]))

fig, axs = locimage3d(microseismic_image_noisy,
                      x0=sx,
                      y0=sy,
                      z0=sz,
                      xlim=[sx-10,sx+10],
                      ylim=[sy-10,sy+10],
                      zlim=[sz-10,sz+10],
                      clipval=[0,1])
plt.tight_layout()