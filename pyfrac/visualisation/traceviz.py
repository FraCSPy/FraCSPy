import matplotlib.pyplot as plt
import numpy as np


def wiggleplot(data, norm_indiv=True, figsize=[12, 6], rec_label=True, xhline=True):
    ntr = np.shape(data)[0]  # num of traces
    fig, axs = plt.subplots(ntr, 1, figsize=figsize, sharex=True)
    for i, ax in enumerate(axs):
        if norm_indiv:
            ax.plot(data[i] / np.max(abs(data[i])), 'k');
        else:
            ax.plot(data[i], 'k');
        if rec_label: ax.set_ylabel(i)
    for ax in axs:
        ax.set_xlim([0, len(data[0])]);
        if norm_indiv: ax.set_ylim([-1, 1]);
        if xhline: ax.hlines(0, 0, len(data[0]), color='gray', linestyles=':')
        ax.set_xticks(np.arange(len(data[0])));
        ax.set_yticks([]);
        ax.spines['top'].set_visible(False);
        ax.spines['right'].set_visible(False);
        ax.spines['bottom'].set_visible(False);
        ax.axis('tight')
        ax.grid(axis="x")
    axs[-1].set_xlabel('Time samples')
    fig.tight_layout()
    return fig, axs


def traceimage(data, norm_indiv=False, figsize=[12, 6], cbar=True, climQ=90, cmap='seismic', ax=None):
    if norm_indiv:
        data = (data.T / np.max(abs(data), axis=1)).T

        

    clim = np.percentile(abs(data), climQ)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(data.T, aspect='auto', interpolation=None,
                   cmap=cmap, vmin=-1 * clim, vmax=clim)
    if cbar: plt.colorbar(im, ax=ax, label='Seis. Amp.')
    ax.set_xlabel('Receiver #')
    ax.set_ylabel('Time')
    ax.set_title('Seismic Recording')
    ax.axis('tight')
    if ax is None: fig.tight_layout()

    return ax

def multiwiggleplot(datalist, norm_indiv=True, figsize=[12, 6], rec_label=True, xhline=True):
    ntr = np.shape(datalist[0])[0]  # num of traces
    fig, axs = plt.subplots(ntr, 1, figsize=figsize, sharex=True)
    clist = ['k','r','b','g']
    for i, ax in enumerate(axs):
        if norm_indiv:
            for ci,data in enumerate(datalist):
                ax.plot(data[i] / np.max(abs(data[i])), clist[ci]);
        else:
            for ci,data in enumerate(datalist):
                ax.plot(data[i], clist[ci]);
        if rec_label: ax.set_ylabel(i)
    for ax in axs:
        ax.set_xlim([0, len(data[0])]);
        if norm_indiv: ax.set_ylim([-1, 1]);
        if xhline: ax.hlines(0, 0, len(data[0]), color='gray', linestyles=':')
        ax.set_xticks(np.arange(len(data[0])));
        ax.set_yticks([]);
        ax.spines['top'].set_visible(False);
        ax.spines['right'].set_visible(False);
        ax.spines['bottom'].set_visible(False);
        ax.axis('tight')
        ax.grid(axis="x")
    axs[-1].set_xlabel('Time samples')
    fig.tight_layout()
    return fig, axs