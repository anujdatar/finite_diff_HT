""" Source matrix for unsteady problem

Author: anuj.datar@gmail.com
Created: 4/18/2017
"""

import numpy as np

def source_time(S_net, phi, rho, cp, dt):
    """ source matrix for unsteady problem using Implicit Euler
    Q_pt = Q_p + rho*Cp*T_p / dt

    Args:

    Returns:
    """

    factor = rho*cp/dt

    Source_time = np.add(S_net, np.multiply(factor, phi))

    return Source_time
