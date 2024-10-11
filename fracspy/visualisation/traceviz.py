import matplotlib.pyplot as plt
import numpy as np
import warnings


default_cmap = 'seismic'


def wiggleplot(data, dt=0.004, norm_indiv=True, figsize=[12, 6], rec_label=True, xhline=True):
    """
    Plots seismic traces as wiggles, with options for normalization and customization of the plot appearance.

    This function visualizes a 2D array of seismic data as a series of wiggle plots (one plot per trace). 
    Each trace is plotted as a line that represents seismic amplitude variations over time. The function 
    supports individual normalization of traces and provides customization options for labels, grid lines, 
    and figure size.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D numpy array where each row represents a seismic trace, and each column represents a time sample.
    dt : float, optional
        Time step between samples in seconds. Default is 0.004.
    norm_indiv : bool, optional
        If True, each trace is individually normalized to its maximum absolute amplitude for better visualization. 
        Default is True.
    figsize : list, optional
        List specifying the width and height of the figure in inches. Default is [12, 6].
    rec_label : bool, optional
        If True, labels indicating the receiver (trace) number are added to the y-axis of a subset of traces. 
        Default is True.
    xhline : bool, optional
        If True, a horizontal line is drawn at zero amplitude for each trace. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure object containing the plotted wiggles.
    axs : numpy.ndarray of matplotlib.axes.Axes
        Array of Axes objects for the individual wiggle plots.

    Notes
    -----
    - If the number of traces (rows in `data`) exceeds 40, a warning is issued as this may affect plot readability.
    - For datasets with more than 10 traces, only a subset of receiver labels is shown to avoid overcrowding.
    - The function adjusts the x and y axis limits based on the normalization option and data range.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> data = np.random.randn(25, 1000)  # 25 traces, 1000 samples each
    >>> fig, axs = wiggleplot(data, dt=0.002)
    >>> plt.show()
    """


    t = np.arange(0,data.shape[1])*dt
    nr = data.shape[0]

    if nr > 40:
        warnings.warn("The number of traces (nr) is greater than 40. This may affect the readability of the plot.", 
                      UserWarning)

    # Put a reasonable number of receiver labels
    if nr>10: reclab_ind = np.linspace(1,nr+2,10).astype('int')
    else: reclab_ind = np.arange(1,nr+1,step=1)

    fig, axs = plt.subplots(nr, 1, figsize=figsize, sharex=True)
    for i, ax in enumerate(axs):
        if norm_indiv:
            ax.plot(t, data[i] / np.max(abs(data[i])), 'k');
        else:
            ax.plot(t, data[i], 'k');
        if rec_label and (i+1) in reclab_ind: ax.set_ylabel(i+1)

    for ax in axs:
        ax.set_xlim([0, t[-1]]);
        if norm_indiv: ax.set_ylim([-1.1, 1.1]); 
        else:  ax.set_ylim([-1*np.max(abs(data)), np.max(abs(data))]); 
        if xhline: ax.hlines(0, 0, t[-1], color='gray', linestyles=':')
        # ax.set_xticks(np.arange(len(data[0])));
        ax.set_yticks([]);
        ax.spines['top'].set_visible(False);
        ax.spines['right'].set_visible(False);
        ax.spines['bottom'].set_visible(False);
        ax.axis('tight')
        # ax.grid(axis="x")

    axs[-1].set_xlabel('Time [s]')
    fig.tight_layout()
    return fig, axs


def traceimage(data, dt=0.004, norm_indiv=False, figsize=[12, 6], cbar=True, climQ=90, cmap=default_cmap, ax=None):
    """
    Plots a seismic trace image of the given data.

    Parameters:
    -----------
    data : numpy.ndarray
        2D array containing seismic data with shape (n_traces, n_samples).
    dt : float, optional
        Time sampling interval for each trace, in seconds. Default is 0.004.
    norm_indiv : bool, optional
        If True, each trace will be individually normalized by its maximum absolute amplitude. Default is False.
    figsize : list, optional
        Figure size as [width, height]. Default is [12, 6].
    cbar : bool, optional
        If True, a colorbar will be displayed. Default is True.
    climQ : float, optional
        Percentile for clipping the color limits. Default is 90.
    cmap : str, optional
        Colormap to use for the plot. Default is 'seismic'.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot. If None, a new figure and axes are created. Default is None.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The Figure object containing the plot.
    ax : matplotlib.axes.Axes
        The Axes object containing the plot.

    Notes:
    ------
    The function plots the seismic data as a heatmap where the x-axis represents the receiver number and the y-axis 
    represents time. If `norm_indiv` is set to True, each trace is normalized by its own maximum absolute amplitude.
    The color limits of the plot are determined by the `climQ` percentile of the absolute values in the data.
    """

    # Sort extents
    t = np.arange(0,data.shape[1])*dt
    nr = data.shape[0]
   
    if norm_indiv:
        data = (data.T / np.max(abs(data), axis=1)).T

    clim = np.percentile(abs(data), climQ)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(data.T, 
                   extent=[0,nr,t[-1],t[0]],
                   interpolation=None,
                   cmap=cmap, vmin=-1 * clim, vmax=clim)
    if cbar: plt.colorbar(im, ax=ax, label='Seis. Amp.')
    ax.set_xlabel('Receiver #')
    ax.set_ylabel('Time [s]')
    ax.set_title('Seismic Recording')
    ax.axis('tight')
    if ax is None: fig.tight_layout()

    return fig, ax


def multiwiggleplot(datalist, 
                    dt=0.004, 
                    norm_indiv=True, 
                    datalabels=['vx', 'vy', 'vz'],
                    figsize=[12, 6], 
                    rec_label=True, 
                    xhline=True):
    """
    Plots multicoponent seismic traces as overlaid wiggles

    This function visualizes multiple seismic data components (e.g., vx, vy, vz) as a series of overlaid wiggle plots.
    Each plot represents the three components at a single receiver location, allowing for comparison 
    and visualization of the amplitude variations over time. It supports normalization and provides options for customizing
    labels, grid lines, and figure size.

    Parameters
    ----------
    datalist : list of numpy.ndarray
        A list of 2D numpy arrays, each representing a different seismic data component (e.g., vx, vy, vz). 
        Each array should have the same shape, where each row represents a seismic trace, and each column represents a time sample.
    dt : float, optional
        Time step between samples in seconds. Default is 0.004.
    norm_indiv : bool, optional
        If True, each trace is individually normalized to its maximum absolute amplitude for better visualization. 
        Default is True.
    datalabels : list of str, optional
        Labels for the different data components to be used in the legend. Default is ['vx', 'vy', 'vz'].
    figsize : list, optional
        List specifying the width and height of the figure in inches. Default is [12, 6].
    rec_label : bool, optional
        If True, labels indicating the receiver (trace) number are added to the y-axis of each trace plot. 
        Default is True.
    xhline : bool, optional
        If True, a horizontal line is drawn at zero amplitude for each trace. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure object containing the plotted wiggles.
    axs : numpy.ndarray of matplotlib.axes.Axes
        Array of Axes objects for the individual wiggle plots.

    Notes
    -----
    - If the number of traces (rows in `datalist[0]`) exceeds 15, a warning is issued as this may affect plot readability.
    - The function overlays the three components (`vx`, `vy`, `vz`) using different colors (`red`, `green`, `blue`).
    - The function adjusts the x and y axis limits based on the normalization option and data range.
    - A legend is added to the first subplot to differentiate the overlaid components.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> vx = np.random.randn(10, 1000)  # 10 traces, 1000 samples each for vx
    >>> vy = np.random.randn(10, 1000)  # 10 traces, 1000 samples each for vy
    >>> vz = np.random.randn(10, 1000)  # 10 traces, 1000 samples each for vz
    >>> fig, axs = multiwiggleplot([vx, vy, vz], dt=0.002)
    >>> plt.show()
    """

    t = np.arange(0,datalist[0].shape[1])*dt
    nr = datalist[0].shape[0]

    if nr > 15:
        warnings.warn("The number of traces (nr) is greater than 15. This may affect the readability of the plot.", 
                      UserWarning)
        
    fig, axs = plt.subplots(nr, 1, figsize=figsize, sharex=True)
    clist = ['r','g','b']

    for i, ax in enumerate(axs):
        if norm_indiv:
            for ci,data in enumerate(datalist):
                ax.plot(t, data[i] / np.max(abs(data[i])), clist[ci], label=datalabels[ci]);
        else:
            for ci,data in enumerate(datalist):
                ax.plot(t, data[i], clist[ci], label=datalabels[ci]);
        if rec_label: ax.set_ylabel(i+1)

    for ax in axs:
        ax.set_xlim([0, t[-1]]);
        if norm_indiv: ax.set_ylim([-1.1, 1.1]); 
        else:  ax.set_ylim([-1*np.max(abs(data)), np.max(abs(data))]); 
        if xhline: ax.hlines(0, 0, t[-1], color='gray', linestyles=':')
        ax.set_yticks([]);
        ax.spines['top'].set_visible(False);
        ax.spines['right'].set_visible(False);
        ax.spines['bottom'].set_visible(False);
        ax.axis('tight')
    hand, labl = axs[0].get_legend_handles_labels()
    axs[0].legend(np.unique(labl))

    axs[-1].set_xlabel('Time samples')
    fig.tight_layout()
    return fig, axs