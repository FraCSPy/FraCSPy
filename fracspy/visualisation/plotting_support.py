'''
Borrowed from Matteo
'''
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plotting_style():
    """Plotting syle
    Define plotting style for the entire project
    """
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 18

    plt.style.use('default')

    plt.rc('text', usetex=True)
    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('font', serif='Times New Roman')  # fontsize of the figure title


def plot_reconstruction_2d(data, datarec, Fop, x, t, dx, f, ks, vel):
    """2D reconstruction visualization
    Display original and reconstructed datasets and their error.
    Parameters
    ----------
    data : :obj:`np.ndarray`
        Full data of size :math:`n_x \times n_t`
    datarec : :obj:`np.ndarray`
        Reconstructed data of size :math:`n_x \times n_t`
    Fop : :obj:`pylops.LinearOperator`, optional
        2D Fourier operator
    x : :obj:`np.ndarray`
       Spatial axis
    t : :obj:`np.ndarray`
       Time axis
    dx : :obj:`float`
       Spatial sampling
    f : :obj:`np.ndarray`
       Frequency axis
    ks : :obj:`np.ndarray`
       Spatial wavenumber axis
    vel : :obj:`float`
       Velocity at receivers
    """
    D = Fop * data
    Drec = Fop * datarec
    nt = data.shape[1]

    fig, axs = plt.subplots(2, 3, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1]})
    axs[0, 0].imshow(data.T, cmap='gray', aspect='auto', vmin=-1, vmax=1,
                     extent=(x[0], x[-1], t[-1], t[0]))
    axs[0, 0].set_title('Original')
    axs[0, 0].set_xlabel('Offset (m)')
    axs[0, 0].set_ylabel('TWT (s)')
    axs[0, 1].imshow(datarec.T, cmap='gray', aspect='auto', vmin=-1, vmax=1,
                     extent=(x[0], x[-1], t[-1], t[0]))
    axs[0, 1].set_title('Reconstructed')
    axs[0, 1].set_xlabel('Offset (m)')
    axs[0, 2].imshow(data.T - datarec.T, cmap='gray', aspect='auto', vmin=-1, vmax=1,
                     extent=(x[0], x[-1], t[-1], t[0]))
    axs[0, 2].set_title('Error')
    axs[0, 2].set_xlabel('Offset (m)')

    axs[1, 0].imshow(np.fft.fftshift(np.abs(D).T)[nt // 2:], cmap='gist_ncar_r', aspect='auto', vmin=0, vmax=1e1,
                     extent=(np.fft.fftshift(ks)[0], np.fft.fftshift(ks)[-1], f[nt // 2 - 1], f[0]))
    axs[1, 0].plot(f / vel, f, 'w'), axs[1, 0].plot(f / vel, -f, 'w')
    axs[1, 0].set_xlim(-1 / (2 * dx), 1 / (2 * dx))
    axs[1, 0].set_ylim(50, 0)
    axs[1, 0].set_xlabel('Wavenumber (1/m)')
    axs[1, 0].set_ylabel('Frequency (Hz)')
    axs[1, 1].imshow(np.fft.fftshift(np.abs(Drec).T)[nt // 2:], cmap='gist_ncar_r', aspect='auto', vmin=0, vmax=1e1,
                     extent=(np.fft.fftshift(ks)[0], np.fft.fftshift(ks)[-1], f[nt // 2 - 1], f[0]))
    axs[1, 1].plot(f / vel, f, 'w'), axs[1, 1].plot(f / vel, -f, 'w')
    axs[1, 1].set_xlim(-1 / (2 * dx), 1 / (2 * dx))
    axs[1, 1].set_ylim(50, 0)
    axs[1, 1].set_xlabel('Wavenumber (1/m)')
    axs[1, 2].imshow(np.fft.fftshift(np.abs(D - Drec).T)[nt // 2:], cmap='gist_ncar_r', aspect='auto', vmin=0, vmax=1e1,
                     extent=(np.fft.fftshift(ks)[0], np.fft.fftshift(ks)[-1], f[nt // 2 - 1], f[0]))
    axs[1, 2].plot(f / vel, f, 'w'), axs[1, 2].plot(f / vel, -f, 'w')
    axs[1, 2].set_xlim(-1 / (2 * dx), 1 / (2 * dx))
    axs[1, 2].set_ylim(50, 0)
    axs[1, 2].set_xlabel('Wavenumber (1/m)')
    plt.tight_layout()


def clim(data:np.ndarray, ratio:float=95):
    """Clipping based on percentiles
    
    Define clipping values for plotting based on percentiles of input data
    
    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Full data
    ratio : :obj:`float`
        Clipping ratio

    Returns
    -------
    limits : :obj:`tuple`
        Tuple of (Minimum value, Maximum value)

    Notes
    -----
        If data has both negative and positive values, limits are (-c,c) where c is clipped maximum absolute value of data.
        If data has only non-negative values, limits are (0,c).
        If data has only non-positive values, limits are (-c,0).        
    """
    c = np.percentile(np.absolute(data), ratio)

    if np.all(data>=0):
        limits = (0,c)
    elif np.all(data<=0):
        limits = (-c,0)
    else:
        limits = (-c,c)

    return limits


def explode_volume(volume, t=None, x=None, y=None,
                   figsize=(8, 8), cmap='bone', clipval=None, p=98,
                   cbar=True, cbarlabel='',
                   tlim=None, xlim=None, ylim=None,
                   tcrop=None, xcrop=None, ycrop=None,
                   labels=('[s]', '[km]', '[km]'),
                   tlabel='t', xlabel='x', ylabel='y',
                   secondcrossloc=None, secondcrosslinespec=None,
                   crosslegend=None,
                   ratio=None, linespec=None, interp=None, title='',
                   filename=None, save_opts=None):
    """Display 3D volume
    Display 3D volume in exploding format (three slices)
    Credits : https://github.com/polimi-ispl/deep_prior_interpolation/blob/master/utils/plotting.py
    Parameters
    ----------
    volume : :obj:`numpy.ndarray`
        3D volume of size ``(nt, nx, ny)``
    t : :obj:`int`, optional
        Slicing index along time axis
    x : :obj:`int`, optional
        Slicing index along x axis
    y : :obj:`int`, optional
        Slicing index along y axis
    figsize : :obj:`bool`, optional
        Figure size
    cmap : :obj:`str`, optional
        Colormap
    clipval : :obj:`tuple`, optional
        Clipping min and max values
    p : :obj:`str`, optional
        Percentile of max value (to be used if ``clipval=None``)
    tlim : :obj:`tuple`, optional
        Limits of time axis in volume
    xlim : :obj:`tuple`, optional
        Limits of x axis in volume
    ylim : :obj:`tuple`, optional
        Limits of y axis in volume
    tlim : :obj:`tuple`, optional
        Limits of cropped time axis to be visualized
    xlim : :obj:`tuple`, optional
        Limits of cropped x axis to be visualized
    ylim : :obj:`tuple`, optional
        Limits of cropped y axis to be visualized
    labels : :obj:`bool`, optional
        Labels to add to axes as suffixes
    tlabels : :obj:`bool`, optional
        Label to use for time axis
    xlabels : :obj:`bool`, optional
        Label to use for x axis
    ylabels : :obj:`bool`, optional
        Label to use for y axis    
    secondcrossloc : :obj:`tuple`, optional, default is None
        Indices of second cross location [x,y,z]
    secondcrosslinespec : :obj:`dict`, optional
        Specifications for lines of second cross
    crosslegend : tuple of str, optional
        Legend labels for crosses, only used if secondcrossloc is not None, default is None
    ratio : :obj:`float`, optional
        Figure aspect ratio (if ``None``, inferred from the volume sizes directly)
    linespec : :obj:`dict`, optional
        Specifications for lines indicating the selected slices
    interp : :obj:`str`, optional
        Interpolation to apply to visualization
    title : :obj:`str`, optional
        Figure title
    filename : :obj:`str`, optional
        Figure full path (if provided the figure is saved at this path)
    save_opts : :obj:`dict`, optional
        Additional parameters to be provided to :func:`matplotlib.pyplot.savefig`
    Returns
    -------
    fig : :obj:`matplotlib.pyplot.Figure`
        Figure handle
    axs : :obj:`matplotlib.pyplot.Axis`
        Axes handles
    """
    secondcross = secondcrossloc is not None
    if linespec is None:
        linespec = dict(ls='-', lw=1.5, color='#0DF690')
    if secondcrosslinespec is None:
        secondcrosslinespec = dict(ls=':', lw=1.5, color='k')
    nt, nx, ny = volume.shape
    t_label, x_label, y_label = labels

    t = t if t is not None else nt // 2
    x = x if x is not None else nx // 2
    y = y if y is not None else ny // 2

    if tlim is None:
        t_label = "samples"
        tlim = (-0.5, nt - 0.5)
    if xlim is None:
        x_label = "samples"
        xlim = (-0.5, nx - 0.5)
    if ylim is None:
        y_label = "samples"
        ylim = (-0.5, ny - 0.5)

    # vertical lines for coordinates reference
    dt, dx, dy = (tlim[1] - tlim[0]) / nt, (xlim[1] - xlim[0]) / nx, (ylim[1] - ylim[0]) / ny
    tline = dt * t + tlim[0] + 0.5 * dt
    xline = dx * x + xlim[0] + 0.5 * dx
    yline = dy * y + ylim[0] + 0.5 * dy
    if secondcross:
        sc_tline = dt * secondcrossloc[2] + tlim[0] + 0.5 * dt
        sc_xline = dx * secondcrossloc[0] + xlim[0] + 0.5 * dx
        sc_yline = dy * secondcrossloc[1] + ylim[0] + 0.5 * dy

    # instantiate plots
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.94)
    if ratio is None:
        wr = (nx, ny)
        hr = (ny, nx)
    else:
        wr = ratio[0]
        hr = ratio[1]
    opts = dict(cmap=cmap, clim=clipval if clipval is not None else clim(volume, p), aspect='auto',
                interpolation=interp)
    gs = fig.add_gridspec(2, 2, width_ratios=wr, height_ratios=hr,
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    # central plot
    ax.imshow(volume[:, :, y], extent=[xlim[0], xlim[1], tlim[1], tlim[0]], **opts)
    ax.axvline(x=xline, **linespec)
    ax.axhline(y=tline, **linespec)
    if secondcross:
        ax.axvline(x=sc_xline, **secondcrosslinespec)
        ax.axhline(y=sc_tline, **secondcrosslinespec)
    if xcrop is not None:
        ax.set_xlim(xcrop)
    if tcrop is not None:
        ax.set_ylim(tcrop[1], tcrop[0])

    # top plot
    c = ax_top.imshow(volume[t].T, extent=[xlim[0], xlim[1], ylim[1], ylim[0]], **opts)
    if secondcross and crosslegend is not None:
        ax_top.axvline(x=xline, **linespec, label=crosslegend[0])
    else:
        ax_top.axvline(x=xline, **linespec)
    ax_top.axhline(y=yline, **linespec)
    if secondcross:
        if crosslegend is not None:
            ax_top.axvline(x=sc_xline, **secondcrosslinespec, label=crosslegend[1])
        else:
            ax_top.axvline(x=sc_xline, **secondcrosslinespec)
        ax_top.axhline(y=sc_yline, **secondcrosslinespec)
        if crosslegend is not None:
            ax_top.legend()
    ax_top.invert_yaxis()
    if xcrop is not None:
        ax_top.set_xlim(xcrop)
    if ycrop is not None:
        ax_top.set_ylim(ycrop[1], ycrop[0])
    if cbar:
        ax_topright = fig.add_subplot(gs[0, 1], sharex=ax)
        ax_topright.axis('off')
        cbaxes = inset_axes(ax_topright, width="3%", height="90%", loc=2)
        plt.colorbar(c, cax=cbaxes, orientation="vertical", shrink=0.75, label=cbarlabel)

    # right plot
    ax_right.imshow(volume[:, x], extent=[ylim[0], ylim[1], tlim[1], tlim[0]], **opts)
    ax_right.axvline(x=yline, **linespec)
    ax_right.axhline(y=tline, **linespec)
    if secondcross:
        ax_right.axvline(x=sc_yline, **secondcrosslinespec)
        ax_right.axhline(y=sc_tline, **secondcrosslinespec)
    if ycrop is not None:
        ax_right.set_xlim(ycrop)
    if tcrop is not None:
        ax_right.set_ylim(tcrop[1], tcrop[0])

    # labels
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_right.tick_params(axis="y", labelleft=False)
    ax.set_xlabel(xlabel + " " + x_label)
    ax.set_ylabel(tlabel + " " + t_label)
    ax_right.set_xlabel(ylabel + " " + y_label)
    ax_top.set_ylabel(ylabel + " " + y_label)

    if filename is not None:
        if save_opts is None:
            save_opts = {'format': 'png', 'dpi': 150, 'bbox_inches': 'tight'}
        plt.savefig(f"{filename}.{save_opts['format']}", **save_opts)

    return fig, (ax, ax_right, ax_top)