import matplotlib.pyplot as plt
from .plotting_support import explode_volume


def locimage3d(mig, x0, y0, pretitle=''):
    fig, axs = explode_volume(mig.transpose(2, 0, 1),
                              p=99.9,
                              x=x0, y=y0)
    fig.suptitle('%s x0=%i, y0=%i' %(pretitle,x0,y0), fontsize=18)
    fig.tight_layout()

    return fig,axs
