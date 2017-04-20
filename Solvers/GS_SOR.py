# -*- coding: utf-8 -*-
""" GS-SOR
Created on Thu Apr 20 13:34:00 2017

@author: Anuj
"""

import numpy as np

def perc_change(old, new):
    """ calculates percentage change """
    return abs(old - new)/old

def Gauss_Seidel(coeff, coeff_time, max_iter, update, phi,
                 source, source_time, omega, phi_inf, epsit):
    """ Gauss Seidel with Successive Over Relaxation

    Args:

    Returns:

    """
    #nz = phi.shape[0]
    ny = phi.shape[0]
    nx = phi.shape[1]

    A_et = coeff_time[0]
    A_wt = coeff_time[1]
    A_nt = coeff_time[2]
    A_st = coeff_time[3]
#    A_ut = coeff_time[4]
    A_dt = coeff_time[5]
    A_pt = coeff_time[6]

    A_e = coeff[0]
    A_w = coeff[1]
    A_n = coeff[2]
    A_s = coeff[3]
#    A_u = coeff[4]
    A_d = coeff[5]
    A_p = coeff[6]

    resid = np.zeros((ny, nx))

    resid_max = 0
    iter_count = 0
    conv_err = 10 * update

    while iter_count < max_iter and conv_err > update:
        iter_count += 1
        phi_max = 0
        dphi_max = 0
        errphi_max = 0

        # copy phi old for use later
        phi_old = np.copy(phi)

        for i in range(1, nx-1):
            for j in range(1, ny-1):
                phi_new = (source_time[j, i]
                           + A_et * phi[j, i+1] + A_wt * phi[j, i-1]
                           + A_nt * phi[j+1, i] + A_st * phi[j-1, i]
                           + A_dt * phi_inf) / A_pt
                phi[j, i] = omega * phi_new + (1-omega)*phi_old[j, i]


# enforce no more than 10% change in solution
                if perc_change(phi[j, i], phi_old[j, i]) > 0.1:
                    phi[j, i] = 1.1 * phi_old[j, i]

# enforce boundary condition for minimum allowed phi value
        phi[phi < phi_inf] = phi_inf

        phi_max = np.max(np.abs(phi))
        iter_update = np.abs(np.subtract(phi, phi_old))
        dphi_max = np.max(iter_update)
        errphi_max = np.max(np.divide(iter_update, phi_old))

        for i in range(1, nx-1):
            for j in range(1, ny-1):
                resid[j, i] = (source[j, i]
                               + A_e * phi[j, i+1] + A_w * phi[j, i-1]
                               + A_n * phi[j+1, i] + A_s * phi[j-1, i]
                               + A_d * phi_inf + A_p * phi[j, i])

        resid_max = np.max(np.abs(resid))

        conv_err = errphi_max
        outer_update = dphi_max
        outer_norm = epsit * phi_max

        phi_old = np.copy(phi)

    return phi, outer_update, outer_norm, resid_max
