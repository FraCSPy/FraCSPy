__all__ = [
    "Location",
]


from fracspy.location.migration import kmigration, diffstack
from fracspy.location.imaging import lsi, sparselsi
from fracspy.location.xcorri import xcorri


_location_kind = {"kmigration": kmigration,
                  "diffstack": diffstack,                    
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

    def apply(self, data, kind="kmigration", **kwargs):
        """Perform event location

        This method performs event location for the provided dataset using
        the pre-defined acquisition geometry using one of the available imaging technique.

        .. note:: This method can be called multiple times using different input datasets
          and/or imaging methods as the internal parameters are not modified during the
          location procedure.

        """
        if kind == "diffstack":
            return _location_kind[kind](data, self.x, self.y, self.z, **kwargs)
        else:
            return _location_kind[kind](data, self.n_xyz, **kwargs)
