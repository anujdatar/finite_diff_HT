# -*- coding: utf-8 -*-
"""Heat out calculations
Created on Thu Apr 20 11:49:45 2017

@author: Anuj
"""

import numpy as np

# constants
STEF_BOL_C = 5.670367e-8

def heat_flux_out(T_inf, T_old, hc_air, emmi):
    """heat out calculating heat loss at surface air interface
    through surface convection and radiation

    Args:
        T_inf:
        T_old:
        hc_air:
        emmi:

    Returns:
        Q_out
    """

    #nz = T_old.shape[0]
    ny = T_old.shape[0]
    nx = T_old.shape[1]

    Q_out = np.zeros((ny, nx))
    h_eff = np.zeros((ny, nx))
    T_eff = np.zeros((ny, nx))
    for i in range(nx):
        for j in range(ny):
            T_eff[j, i] = ((T_old[j, i]**3) + (T_inf * T_old[j, i]**2)
                              + (T_old[j, i] * T_inf**2) + T_inf**3)

            h_eff[j, i] = hc_air + (emmi*STEF_BOL_C*T_eff[j, i])

            Q_out[j, i] = h_eff[j, i] * (T_old[j, i] - T_inf)

    return Q_out
