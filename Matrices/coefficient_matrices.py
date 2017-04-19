""" setup coefficient matrices

Author: anuj.datar@gmail.com
Created: 4/18/2017
"""

import numpy as np

def coeff_matrices(k, dx, dy, dz):
    """ Coefficient matrices for the steady state problem

    Args:
        k:
        nx:
        ny:
        dx:
        dy:

    Returns:
        coeff mats for E, W, N, S, U, D
    """
    A_matrix = np.zeros(7)

    A_matrix[0] = k/(dx*dx) # East
    A_matrix[1] = k/(dx*dx) # West
    A_matrix[2] = k/(dy*dy) # North
    A_matrix[3] = k/(dy*dy) # South
    A_matrix[4] = k/(dz*dz) # Up
    A_matrix[5] = k/(dz*dz) # Dows
    A_matrix[6] = - (A_matrix[0] + A_matrix[1] + A_matrix[2]
                     + A_matrix[3] + A_matrix[4] + A_matrix[5]) # Pole

    return A_matrix

def coeff_matrix_time(A_matrix, rho, C_p, dt):
    """ Coefficient matrix for the unsteady problem
    using the Implicit Euler time advancement technique

    Args:

    Returns:
    """
    A_matrix_time = np.zeros(7)

    A_matrix_time[0] = - A_matrix[0] # East
    A_matrix_time[1] = - A_matrix[1] # West
    A_matrix_time[2] = - A_matrix[2] # North
    A_matrix_time[3] = - A_matrix[3] # South
    A_matrix_time[4] = - A_matrix[4] # Up
    A_matrix_time[5] = - A_matrix[5] # Down
    A_matrix_time[6] = (rho*C_p/dt) - A_matrix[6] # Pole

    return A_matrix_time
