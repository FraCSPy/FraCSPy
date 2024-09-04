import matplotlib.pyplot as plt
from .plotting_support import explode_volume


def locimage3d(image, 
               x0, y0, z0, 
               intersect_linecol='k',
               title='', 
               p=99.9, 
               clipval=None, 
               cmap='hot_r'):
    linespec = dict(ls='-', lw=1.5, color=intersect_linecol)

    fig, axs = explode_volume(image.transpose(2, 0, 1),
                              p=p, clipval=clipval,
                              x=x0, y=y0, t=z0,
                              tlabel='z',
                              cmap=cmap,
                              linespec=linespec)
    fig.suptitle(title, fontsize=18)
    fig.tight_layout()

    return fig,axs
