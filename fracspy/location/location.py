__all__ = [
    "Location",
]


from fracspy.location.migration import diffstack, semblancediffstack
from fracspy.location.imaging import lsi, sparselsi, xcorri


_location_kind = {"diffstack": diffstack,
                  "semblancediffstack": semblancediffstack,
                  "lsi": lsi,
                  "sparselsi": sparselsi,
                  "xcorri": xcorri,
                  }


class Location():
    """Event location

    This class acts as an abstract interface for users to perform
    event location on a microseismic dataset

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        X-axis
    y : :obj:`numpy.ndarray`
        Y-axis
    z : :obj:`numpy.ndarray`
        Z-axis

    """
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        self.n_xyz = x.size, y.size, z.size

    def apply(self, data, kind="diffstack", **kwargs):
        """Perform event location

        This method performs event location for the provided dataset using
        the pre-defined acquisition geometry using one of the available imaging techniques.

        .. note:: This method can be called multiple times using different input datasets
          and/or imaging methods as the internal parameters are not modified during the
          location procedure.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            Data of shape :math`n_r \times n_t`
        kind : :obj:`str`, optional
            Algorithm kind (`diffstack`, `semblancediffstack`,
            `lsi`, `sparselsi`, or `xcorri`
        kwargs : :obj:`dict`, optional
            Keyword arguments to pass to the location algorithm

        Returns
        -------
        im : :obj:`numpy.ndarray`
            Migrated volume
        hc : :obj:`numpy.ndarray`
            Estimated hypocentral location

        """
        im, hc = _location_kind[kind](data, self.n_xyz, **kwargs)[:2]
        return im, hc