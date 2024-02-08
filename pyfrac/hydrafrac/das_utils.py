import numpy as np
from pylops.basicoperators import FirstDerivative


def strain_from_velocity(perp_velocity):
    '''
    perp_velocity follow pytorch dims: 'instance'-by-x-by-t
    e.g., [cable_no, x, t]
    where x is axis over which to differentiate
    '''
    # handle if only 1 instance, i.e. single rec line
    if len(perp_velocity.shape) == 2:
        perp_velocity = np.expand_dims(perp_velocity, axis=0)

    # Make differential operator
    diffOp = FirstDerivative(dims=perp_velocity.shape,
                             axis=1,
                             order=5)

    # Take differentiate over velocity direction
    strain = diffOp @ perp_velocity

    return strain