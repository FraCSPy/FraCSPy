from pyfrac.locationsolvers.localisationutils import get_max_locs
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
import pylops


def XcorrObjFunc(x,y):
    """ Cross-correlation objective function
    Parameters
    ----------
    x : torch.tensor
        data 1
    y : torch.tensor
        data 2
    Returns
    -------
    type
        description
    """
    x = x/torch.linalg.norm(x)
    y = y/torch.linalg.norm(y)
    loss = - torch.sum(torch.mul(x,y))
    return loss


def xcorr_imaging(Op, data, n_xyz, niter=100, xceps=8e-1, lr=1e-5, nforhc=10, verbose=True):

    # Initialise with migrated image
    migrated = (Op.H @ data).reshape(n_xyz)
    dmigrated = Op @ migrated.ravel()
    scaling = data.max() / dmigrated.max()
    m = torch.from_numpy(migrated.copy().ravel() * scaling)
    m.requires_grad = True  # make sure we compute the gradient with respect to m

    # Initialise torch operator and torch tensors
    TOp = pylops.TorchOperator(Op)
    dobs = torch.from_numpy(data.copy().ravel())

    # Optimization
    optimizer = torch.optim.SGD([m], lr=lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    losshist = np.zeros(niter)
    for i in range(niter):
        optimizer.zero_grad()
        d = TOp(m)

        # data term
        lossd = XcorrObjFunc(d, dobs)
        # L1 reg
        reg = torch.sum(torch.abs(m))
        # total loss
        loss = lossd + (xceps * reg)

        loss.backward()
        optimizer.step()
        scheduler.step()
        losshist[i] = loss.item()
        if verbose and i % 10 == 0:
            print(f'Iteration {i}, Loss: {loss.item():.5f}')

    dls_torch = d.detach().cpu().numpy().reshape(data.shape)
    mls_torch = m.detach().cpu().numpy().reshape(n_xyz)
    hc, hcs = get_max_locs(mls_torch, n_max=nforhc, rem_edge=False)
    return mls_torch, hc, dls_torch, losshist
