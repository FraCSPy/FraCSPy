r"""
Comparing Moment Tensor Matrices
=======================================
In this example, we will perform a quick and dirty amplitude-based inversion
for the moment tensor.

As this is a plotting example, we will tackle the problem as an inverse-crime.
For a more scientific example of how to perform amplitude-based moment-tensor
inversion, look into the tutorial [XXX] which uses data generated via finite-
difference modelling.
"""
import matplotlib.pyplot as plt
import numpy as np
from fracspy.visualisation.momenttensor_plots import MTMatrix_comparisonplot


plt.close("all")
np.random.seed(0)

###############################################################################
# Let's begin by creating a certain MT - we consider this the True MT
mt_xx = 0
mt_yy = 0
mt_zz = 0
mt_xy = -1
mt_xz = 0
mt_yz = 0
mt = [mt_xx, mt_yy, mt_zz, mt_xy, mt_xz, mt_yz]
print('MT for forward modelling: ', mt)


###############################################################################
# Let's make a second MT which is a noisy version of our True MT - we can consider
# this the Estimated MT
mt_noise = 0.75 * (np.random.random(6)-0.5)
mt_est = mt + mt_noise


###############################################################################
# Now we are ready to plot the comparison between our known MT (mt) and our
# estimate MT (mt_est)

MTMatrix_comparisonplot(mt, mt_est)
plt.show()

