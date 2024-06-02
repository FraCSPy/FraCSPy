import matplotlib.pyplot as plt
from .plotting_support import explode_volume


def locimage3d(mig, x0, y0, z0, title='', p=99.9, clipval=None, cmap='bone'):
    fig, axs = explode_volume(mig.transpose(2, 0, 1),
                              p=p, clipval=clipval,
                              x=x0, y=y0, t=z0,
                              tlabel='z',
                              cmap=cmap)
    fig.suptitle(title, fontsize=18)
    fig.tight_layout()

    return fig,axs
