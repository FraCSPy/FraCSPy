r"""
Moment Tensor Plotting - Matrix Style
=======================================
This example shows how to create a heatmap of the moment tensor matrix
"""
import matplotlib.pyplot as plt
import numpy as np
from fracspy.visualisation.momenttensor_plots import MTBeachball


plt.close("all")
np.random.seed(0)

###############################################################################
# Let's start by creating a moment tensor, as the moment tensor is a symmetric
# matrix we only need to define 6 of the 9 components
mt_xx = 0
mt_yy = 0
mt_zz = 0
mt_xy = -1
mt_xz = 0
mt_yz = 0
mt = [mt_xx, mt_yy, mt_zz, mt_xy, mt_xz, mt_yz]
print(mt)


###############################################################################
# We are going to visualise the moment tensor in the common form of a beachball
# defined by the microseismic event's focal mechanism. The plotting function
# handles conversion of the six-component moment tensor to the focal mechanism
# form desired by the ObSpy python package. After which, the beachball is drawn
# and attached to the provided axis.
# Here we utilise the function:
# pyfrac.visualisation.momenttensor_plots.MTBeachball

fig, ax = plt.subplots(1,1,figsize=[4.8,4.8])
MTBeachball(mt,ax)
plt.show()
