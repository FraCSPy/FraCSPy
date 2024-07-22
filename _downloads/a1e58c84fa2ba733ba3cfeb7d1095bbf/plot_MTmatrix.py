r"""
Moment Tensor Plotting - Matrix Style
=======================================
This example shows how to create a heatmap of the moment tensor matrix
"""
import matplotlib.pyplot as plt
import numpy as np
from fracspy.visualisation.momenttensor_plots import MTMatrixplot


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
# We are going to visualise the moment tensor in its original 3-by-3 format,
# as a heatmap with the MT component values overlaid on top of their respective
# coordinate. Here we utilise the function:
# pyfrac.visualisation.momenttensor_plots.MTMatrixplot

fig, ax = plt.subplots(1,1,figsize=[4.8,4.8])
MTMatrixplot(mt,ax)
plt.show()
