import matplotlib.pyplot as plt
import numpy as np
import warnings


def wiggleplot(data, dt=0.004, norm_indiv=True, figsize=[12, 6], rec_label=True, xhline=True):

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


def traceimage(data, dt=0.004, norm_indiv=False, figsize=[12, 6], cbar=True, climQ=90, cmap='seismic', ax=None):
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


def multiwiggleplot(datalist, dt=0.004, norm_indiv=True, 
                    datalabels=['vx', 'vy', 'vz'],
                    figsize=[12, 6], rec_label=True, xhline=True):
    
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