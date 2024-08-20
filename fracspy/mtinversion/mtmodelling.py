__all__ = [
    "MTA",
    "MTW",
]

from scipy.linalg import lstsq

from fracspy.mtinversion.greensfunction import *


class MTA():
    """Moment-Tensor Amplitude modelling and inversion

    This class acts as an abstract interface for users to perform
    moment-tensor modelling of far-field amplitudes

    Parameters
    ----------
    G : :obj:`numpy.ndarray`
        Single component Green's functions of size :math:`6 \times n_r \times n_x \times n_y \times n_z`


    """
    def __init__(self, G):
        self.G = G

    def model(self, mt):
        """Modelling

        This method models the amplitudes resulting from a given moment-tensor source

        Parameters
        ----------
        mt : :obj:`numpy.ndarray`
            Moment tensor of size :math`6`

        Results
        -------
        d : :obj:`numpy.ndarray`
            Modelled amplitudes of size :math`n_r`

        """
        d = np.matmul(self.G.T, mt)
        return d

    def invert(self, d):
        """Inversion

        This method inverts the amplitudes for the corresponding given moment-tensor source

        Parameters
        ----------
        d : :obj:`numpy.ndarray`
            Amplitudes of size :math`n_r`

        Results
        -------
        mt : :obj:`numpy.ndarray`
            Moment tensor of size :math`6`

        """
        mt = lstsq(self.G.T, d)[0]
        return mt


class MTW():
    """Moment-Tensor Waveform modelling and inversion

    This class acts as an abstract interface for users to perform
    moment-tensor modelling of waveforms

    Parameters
    ----------
    G : :obj:`numpy.ndarray`
        Single component Green's functions of size :math:`6 \times n_r \times n_x \times n_y \times n_z`

    """

    def __init__(self, G):
        self.G = G

    def model(self, mt):
        """Modelling

        This method models the amplitudes resulting from a given moment-tensor source

        Parameters
        ----------
        mt : :obj:`numpy.ndarray`
            Moment tensor of size :math`6`

        Results
        -------
        d : :obj:`numpy.ndarray`
            Modelled amplitudes of size :math`n_r`

        """
        d = np.matmul(self.G.T, mt)
        return d

    def invert(self, d):
        """Inversion

        This method inverts the amplitudes for the corresponding given moment-tensor source

        Parameters
        ----------
        d : :obj:`numpy.ndarray`
            Amplitudes of size :math`n_r`

        Results
        -------
        mt : :obj:`numpy.ndarray`
            Moment tensor of size :math`6`

        """
        mt = lstsq(self.G.T, d)[0]
        return mt
