import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from obspy.imaging.beachball import beach


def _mt4plt(mt):
    """
    Converts a 1x6 Moment Tensor array into a 3x3 matrix and a padded 5x5 matrix for visualization.

    Parameters
    ----------
    mt : numpy.ndarray [1x6]
        Moment Tensor array defined according to the format specified in `pyfrac.mtsolvers.mtutils.get_mt_computation_dict`.

    Returns
    -------
    mat_mt : numpy.ndarray [3x3]
        The Moment Tensor represented as a 3x3 matrix.
    mat_mt4plt : numpy.ndarray [5x5]
        The Moment Tensor represented as a padded 5x5 matrix for visualization purposes, with padding added for aesthetics.
    
    """
    # Define MT matrix
    mat_mt = np.empty([3 ,3])
    mat_mt[0 ,0] = mt[0]  # M_{x,x}
    mat_mt[1 ,1] = mt[1]  # M_{y,y}
    mat_mt[2 ,2] = mt[2]  # M_{z,z}
    mat_mt[0 ,1] = mt[3]  # M_{x,y}
    mat_mt[1 ,0] = mt[3]  # M_{y,x}
    mat_mt[0 ,2] = mt[4]  # M_{x,z}
    mat_mt[2 ,0] = mt[4]  # M_{z,x}
    mat_mt[1 ,2] = mt[5]  # M_{y,z}
    mat_mt[2 ,1] = mt[5]  # M_{z,y}

    # Make padded for plot
    mat_mt4plt = np.empty([5 ,5])
    mat_mt4plt[1:-1 ,1:-1] = mat_mt
    mat_mt4plt[0] = np.nan
    mat_mt4plt[-1] = np.nan
    mat_mt4plt[: ,0] = np.nan
    mat_mt4plt[: ,-1] = np.nan
    return mat_mt, mat_mt4plt


def MTMatrixplot(mt, ax, cmap=None, title=True):
    """
    Plots a Moment Tensor matrix as a heatmap on a given axis.

    Parameters
    ----------
    mt : numpy.ndarray [1x6]
        Moment Tensor array defined according to the format specified in `pyfrac.mtsolvers.mtutils.get_mt_computation_dict`.
    ax : matplotlib.axes.Axes
        The matplotlib axis on which to plot the Moment Tensor heatmap.
    cmap : matplotlib.colors.Colormap, optional
        Colormap used for displaying the Moment Tensor heatmap. If not provided, `matplotlib.cm.PiYG` is used as default.
    title : bool, optional
        If True, a title 'Source Moment Tensor' is displayed on the plot. Default is True.

    """

    if not cmap:
        cmap = matplotlib.cm.PiYG
    cmap.set_bad('w', 1.)

    mt_matrix, mat_mt4plt = _mt4plt(mt)
    ax.imshow(mat_mt4plt, cmap=cmap, vmin=-1, vmax=1)

    # Add lines around matrix
    linlocs = [0.5 ,1.5 ,2.5 ,3.5]
    for loc in linlocs:
        ax.hlines(y=loc ,xmin=0.5 ,xmax=3.5, color='k')
        ax.vlines(x=loc ,ymin=0.5 ,ymax=3.5, color='k')
    ax.set_xlim([0.25 ,3.75]); ax.set_ylim([3.75 ,0.25]);

    # Add text showing MT values
    ax.text(1 ,1 ,'%.1e ' %mt_matrix[0 ,0], va='center', ha='center', fontweight='bold', fontsize=12)
    ax.text(2 ,2 ,'%.1e ' %mt_matrix[1 ,1], va='center', ha='center', fontweight='bold', fontsize=12)
    ax.text(3 ,3 ,'%.1e ' %mt_matrix[2 ,2], va='center', ha='center', fontweight='bold', fontsize=12)
    ax.text(1 ,2 ,'%.1e ' %mt_matrix[0 ,1], va='center', ha='center', fontweight='bold', fontsize=12)
    ax.text(2 ,1 ,'%.1e ' %mt_matrix[0 ,1], va='center', ha='center', fontweight='bold', fontsize=12)
    ax.text(1 ,3 ,'%.1e ' %mt_matrix[0 ,2], va='center', ha='center', fontweight='bold', fontsize=12)
    ax.text(3 ,1 ,'%.1e ' %mt_matrix[0 ,2], va='center', ha='center', fontweight='bold', fontsize=12)
    ax.text(3 ,2 ,'%.1e ' %mt_matrix[1 ,2], va='center', ha='center', fontweight='bold', fontsize=12)
    ax.text(2 ,3 ,'%.1e ' %mt_matrix[1 ,2], va='center', ha='center', fontweight='bold', fontsize=12)

    # Title
    if title:
        ax.text(2 ,0.25 ,'Source Moment Tensor', va='center', ha='center', fontweight='bold', fontsize=14)
    # White background and removing axis
    ax.set_facecolor('w')
    ax.axis('off')


def MTBeachball(mt, ax):
    ''' Generating beachball plot of the 6 component moment tensor, heavily leveraging obspy!

    Parameters
    ----------
    mt : numpy array [1x6]
        Moment Tensor array following mt definition defined in  pyfrac.mtsolvers.mtutils.get_mt_computation_dict
    ax: pyplot axis
        Figure axis on which to plot, e.g., fig,ax = plt.subplots(1,1)

    Notes
    -----
    Must first convert the moment tensor into the obspy required format: NM x 6 (M11, M22, M33, M12, M13, M23 -
    the six independent components of the moment tensor, where the coordinate system is 1,2,3 = Up,South,East
    which equals r,theta,phi - Harvard/Global CMT convention). The relation to Aki and Richards x,y,z equals
    North,East,Down convention is as follows: Mrr=Mzz, Mtt=Mxx, Mpp=Myy, Mrt=Mxz, Mrp=-Myz, Mtp=-Mxy
    '''
    Mrr = mt[2]
    Mtt = mt[0]
    Mpp = mt[1]
    Mrt = mt[4]
    Mrp = -1 * mt[5]
    Mtp = -1 * mt[3]

    NMmatrix = [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp]

    # plot the collection
    collection = beach(fm=NMmatrix)
    ax.add_collection(collection)
    ax.autoscale_view(tight=False,)
    ax.axis('off')


def MTMatrix_comparisonplot(mt, mt_est):
    ''' 
    Creates a side-by-side matrix heatmap comparison between a known and an estimated Moment Tensor.

    Parameters
    ----------
    mt : numpy array [1x6]
        Known Moment Tensor array following mt definition defined in :func:`fracspy.mtinversion.utils.MT_comp_dict`
    mt_est : numpy array [1x6]
        Estimated Moment Tensor array following mt definition defined in :func:`fracspy.mtinversion.utils.MT_comp_dict`

    '''
    fig, axs = plt.subplots(1, 2, figsize=[10, 5])
    MTMatrixplot(mt, axs[0], title=False)
    MTMatrixplot(mt_est, axs[1], title=False)
    axs[0].text(2, 0.25, 'True MT', va='center', ha='center', fontweight='bold', fontsize=14)
    axs[1].text(2, 0.25, 'Estimated MT', va='center', ha='center', fontweight='bold', fontsize=14)


def MTBeachball_comparisonplot(mt, mt_est):
    '''
    Creates a side-by-side beachball plot comparison between a known and an estimated Moment Tensor.

    Parameters
    ----------
    mt : numpy array [1x6]
        Known Moment Tensor array following mt definition defined in  pyfrac.mtsolvers.mtutils.get_mt_computation_dict
    mt_est : numpy array [1x6]
        Estimated Moment Tensor array following mt definition defined in  pyfrac.mtsolvers.mtutils.get_mt_computation_dict

    '''
    fig, axs = plt.subplots(1, 2, figsize=[10, 5])
    MTBeachball(mt, axs[0])
    MTBeachball(mt_est, axs[1])
    axs[0].set_title('True Focal Mechanism', va='center', ha='center', fontweight='bold', fontsize=14)
    axs[1].set_title('Estimated Focal Mechanism', va='center', ha='center', fontweight='bold', fontsize=14)
    fig.tight_layout()
