__all__ = [
    "Location",
]

import numpy as np
from fracspy.location.migration import diffstack, kmigration
from fracspy.location.imaging import lsi, sparselsi, xcorri


_location_kind = {"kmigration": kmigration,
                  "diffstack": diffstack,                    
                  "lsi": lsi,
                  "sparselsi": sparselsi,
                  "xcorri": xcorri,
                  }


class Location():
    """Event location

    This class acts as an abstract interface for users to perform
    event location on a microseismic dataset.
    It assumes that grid vectors are regularly spaced.

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
        self.dx = self.x[1]-self.x[0]
        self.dy = self.y[1]-self.y[0]
        self.dz = self.z[1]-self.z[0]
        self.n_xyz = x.size, y.size, z.size

    def apply(self, data, kind="kmigration", **kwargs):
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
        
    def grid(self):
        """Construct the grid array from the internal grid vectors
        
        This method constructs the grid array of size (3, self.n_xyz) 
        from the internal grid vectors.

        .. note:: This method can be called multiple times as the internal parameters are not modified.

        """
        # Create a meshgrid
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        # Stack the arrays into a (3, self.n_xyz) array
        return np.vstack((X.flatten(), Y.flatten(), Z.flatten()))

    def indtogrid(self, points:np.ndarray):
        """Return the grid coordinates for points provided as grid indices

        This method computes the spatial grid coordinates of points with coordinates provided as grid indices.
        Points have shape (3,npoints) where `npoints` is number of points.

        .. note:: This method can be called multiple times as the internal parameters are not modified.

        """
        return np.array([
            self.x[0] + points[0] * self.dx,
            self.y[0] + points[1] * self.dy,
            self.z[0] + points[2] * self.dz
        ])
    
    def gridtoind(self, points:np.ndarray):
        """Return the grid indices for points with coordinates on a grid

        This method computes the grid indices for point with provided spatial grid coordinates
        Points have shape (3,npoints) where `npoints` is number of points.

        .. note:: This method can be called multiple times as the internal parameters are not modified.

        """
        return np.array([
            ((points[0] - self.x[0]) / self.dx).astype(int),
            ((points[1] - self.y[0]) / self.dy).astype(int),
            ((points[2] - self.z[0]) / self.dz).astype(int)
        ])
