import numpy as np
from scipy.linalg import lstsq


def frwrd_mtmodelling(Gz, mt):
    return np.matmul(Gz.T, mt)


def lsqr_mtsolver(Gz, u_zp):
    mt_lsqr, res, rnk, s = lstsq(Gz.T, u_zp)
    return mt_lsqr
