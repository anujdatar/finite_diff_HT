"""finite difference tests in python

Author: anuj.datar@gmail
Created: 4/15/2017
"""
#pylint: disable=I0011,C0103,C0413,W0105,W0621,E0401,E1101
## I0011 - Ignores message that errors are being ignored
## C0103 - Ignores variable name issues
## C0413 - Ignores module import order messages
## W0105 - Ignores 'string statement has no effect' error'
## W0621 - Ignores variable scope messages, if you have same name inside and
### outside of a function within the same file
## E0401 - Ignores custom module import messages
## E1101 - Ignores message when pylint can't find method within module

# importing standard libraries
import time
import math
import numpy as np

import matplotlib
matplotlib.use('Agg') # enables backend/headless plotting
# Can use GTK or GTKAgg - but need Gtk package
import matplotlib.pyplot as plt

# import custom modules
# from Toolpaths import path_select
from Matrices import coeff_matrices
from Matrices import coeff_matrix_time
from Matrices import source_time
from Heat_Flux import heat_flux_out
from Solvers import Gauss_Seidel

PI = math.pi
# define material properties
C_P = 564  # heat capacity (J/kgK)
K = 6  # thermal conductivity (W/mK)
RHO = 4450  # material density (kg/m^3)
T_MELT = 1933  # melting point of material (K)
ABS = 0.2  # material absorptivity to laser radiation = 1-Reflectivity
alpha = K/(RHO*C_P)  # thermal diffusivity (m^2/s)

HC_AIR = 5
EMMI = 0.5

# define process properties
L_POW = 100  # laser power (W)
L_VEL = 5e-3  # laser scanning velocity (m/s)
L_SPOT_SIZE = 50e-5  # laser spot size -> diameter (m)
T_INF = 600  # ambient temperature of enclosure (K)
q_laser = ABS * L_POW / (L_SPOT_SIZE**2)

# grid definition
x = 20  # grid spaces in X
y = 20  # grid spaces in Y
z = 3  # grid spaces in Z
bdry_offset = 2  # extra nodes in X & Y to overcome boundary issues and discontinuity
lx = 5e-3  # length of slab in the X-direction
ly = 1e-3  # length of slab in the Y-direction
# lz = 2e-4 ## thickness of slab in the Z-direction

# grid definition
dx = lx/x  # grid spacing size in X
dy = ly/y  # grid spacing size in Y
dz = dx  # grid spacing size in Z
dt = dx/L_VEL  # time increment

nx = x + 1 + bdry_offset
ny = y + 1 + bdry_offset
nz = z

# setup coefficient matrices
A_matrix = coeff_matrices(K, dx, dy, dz)
A_matrix_time = coeff_matrix_time(A_matrix, RHO, C_P, dt)

# iterative algorithm parameters
MAX_IN_ITER = 10  # max number of inner iterations
MAX_OUT_ITER = 100  # max number of outer iterations
UPDATE_TARG = 0.01  # % update in solution, inner norm for GS-SOR
EPSIT = 1e-12  # very small number
RES_LARGE = 1e16  # for divergence check
OMEGA = 1.8  # relaxation parameter for GS-SOR

# defining empty matrices
Temp = np.zeros((ny, nx)) + 800  # initial temperature set to ambient
Temp_old = np.copy(Temp)  # copying temperature matrix for the time algorithm
Q_in = np.zeros((ny, nx))  # heat-flux-in matrix
Q_net = np.zeros((ny, nx))  # net heat-flux-in for the substrate


start_timer = time.clock()
curr_time = 0
while curr_time < dt:
    curr_time += dt

    outer_iter = 0
    outer_norm = EPSIT
    outer_update = 10 * outer_norm
    resid_max = 0

    while outer_iter < MAX_OUT_ITER and outer_update > outer_norm:
        outer_iter += 1

        Q_out = heat_flux_out(T_INF, Temp_old, HC_AIR, EMMI)
        Q_net = np.subtract(Q_in, Q_out)

        Q_time = source_time(Q_net, Temp, RHO, C_P, dt)

        # iterative solver
        Temp, outer_update, outer_norm, resid_max =\
        Gauss_Seidel(A_matrix, A_matrix_time, MAX_OUT_ITER, UPDATE_TARG,
                     Temp, Q_net, Q_time, OMEGA, T_INF, EPSIT)

        print('outer iter', outer_iter, 'update', outer_update, 'norm', outer_norm)
        print('resid max', resid_max)
        plt_name = './plt/t_plot_%d.png' % outer_iter
        plt.contourf(Temp)
        plt.jet()
        plt.colorbar()
        plt.savefig(plt_name)
        plt.close()

code_run_time = round(time.clock() - start_timer, 2)
print('runtime = ', code_run_time, 'seconds')



