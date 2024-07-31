import numpy as np
from scipy.linalg import lstsq


def mtamplitude_modelling(Gz, mt):
    return np.matmul(Gz.T, mt)


def mtamplitude_inversion(Gz, u_zp):
    mt_inv = lstsq(Gz.T, u_zp)[0]
    return mt_inv
