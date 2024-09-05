import matplotlib.pyplot as plt
from .plotting_support import explode_volume


def locimage3d(image, 
               x0, y0, z0, 
               intersect_linecol='k',
               title='', 
               p=99.9, 
               clipval=None, 
               cmap='hot_r',
               xlim=None,
               ylim=None, 
               zlim=None, 
               labels=('[m]', '[m]', '[m]')):    
    """
    Plots x-y, y-z, and x-z slices of a 3D volume for visualizing microseismic source images and moment tensor (MT) kernel images.

    This function generates unrolled 2D slices of a given 3D numpy array (`image`) at specified coordinates (x0, y0, z0). 
    It is particularly useful for visualizing microseismic source images or MT kernels over a subsurface area of interest. 
    The appearance of the resulting plots can be customized using various parameters.

    Parameters:
    -----------
    image : numpy.ndarray
        A 3D numpy array representing the volume to be sliced and visualized. x,y,z orientation.
    x0, y0, z0 : int
        The x, y, and z indices where the intersections will be made in the 3D volume.
    intersect_linecol : str, optional
        Color of the intersecting lines in the plot that indicate the slicing locations. Default is 'k' (black).
    title : str, optional
        Title for the entire figure. Default is an empty string.
    p : float, optional
        Percentile value for intensity scaling of the image. Default is 99.9.
    clipval : tuple or None, optional
        Clip value for the intensity scaling of the image. If None, it is not applied. Default is None.
    cmap : str, optional
        Colormap used for displaying the slices. Default is 'hot_r'.
    xlim, ylim, zlim : tuple or None, optional
        Limits for the x, y, and z axes, respectively. Each should be a tuple of (min, max). If None, the limits are set automatically. Default is None.
    labels : tuple of str, optional
        Labels for the x, y, and z axes. Default is ('[m]', '[m]', '[m]').

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The Figure object containing the plotted slices.
    axs : numpy.ndarray of matplotlib.axes.Axes
        Array of Axes objects for the different slices.

    Notes:
    ------
    The function utilizes the `explode_volume` method for handling the 3D volume slicing and plotting. 
    Slices are generated for the x-y, y-z, and x-z planes at the specified coordinates, with options 
    for adjusting color maps, intensity scaling, and axis limits.
    """

    
    linespec = dict(ls='-', lw=1.5, color=intersect_linecol)

    fig, axs = explode_volume(image.transpose(2, 0, 1),
                              p=p, clipval=clipval,
                              x=x0, y=y0, t=z0,
                              tlabel='z',
                              cmap=cmap,
                              linespec=linespec,
                              xlim=xlim,
                              ylim=ylim, 
                              tlim=zlim, 
                              labels=labels,
                              )
    fig.suptitle(title, fontsize=18)
    fig.tight_layout()

    return fig,axs
