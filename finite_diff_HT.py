"""finite difference tests in python

Author: anuj.datar@gmail
Created: 4/15/2017
"""

# importing standard libraries
import time
import math
import numpy as np

import matplotlib
matplotlib.use('Agg') # enables backend/headless plotting
## Can use GTK or GTKAgg - but need Gtk package
import matplotlib.pyplot as plt

# import custom modules
from Toolpaths import path_select
from Matrices import coeff_matrices
from Matrices import coeff_matrix_time

PI = math.pi
### define material properties
C_P = 564 # heat capacity (J/kgK)
K = 6 # thermal conductivity (W/mK)
RHO = 4450 # material density (kg/m^3)
T_MELT = 1933 # melting point of material (K)
ABS = 0.2 # material absorptivity to laser radiation = 1-Reflectivity
alpha = K/(RHO*C_P) # thermal diffusivity (m^2/s)

### define process properties
L_POW = 100 # laser power (W)
L_VEL = 5e-3 # laser scanning velocity (m/s)
L_SPOT_SIZE = 50e-5 # laser spot size -> diameter (m)
T_INF = 600 # ambient temperature of enclosure (K)
q_laser = ABS * L_POW / (L_SPOT_SIZE**2)

### grid definition
x = 50 # grid spaces in X
y = 10 # grid spaces in Y
z = 3 # grid spaces in Z
bdry_offset = 2 # extra nodes in X & Y to overcome boundary issues and discontinuity
lx = 5e-3 # length of slab in the X-direction
ly = 1e-3 # length of slab in the Y-direction
#lz = 2e-4 ## thickness of slab in the Z-direction

### grid definition
dx = lx/x # grid spacing size in X
dy = ly/y # grid spacing size in Y
dz = dx # grid spacing size in Z
dt = dx/L_VEL # time increment

nx = x + 1 + bdry_offset
ny = y + 1 + bdry_offset
nz = z

### setup coefficient matrices
A_matrix = coeff_matrices(K, dx, dy, dz)
A_matrix_time = coeff_matrix_time(A_matrix, RHO, C_P, dt)

### iterative algorithm parameters
MAX_IN_ITER = 10 # max number of inner iterations
MAX_OUT_ITER = 1000 # max number of outer iterations
UPDATE_TARG = 0.01 # % update in solution, inner norm for GS-SOR
EPSIT = 1e-12 # very small number
RES_LARGE = 1e16 # for divergence check
OMEGA = 1.8 # relaxation parameter for GS-SOR

### defining empty matrices
Temp = np.zeros((nz, ny, nx)) + T_INF # initial temperature set to ambient
Temp_old = np.copy(Temp) # copying temperature matrix for the time algorithm
Q_in = np.zeros((nz, ny, nx)) # heat-flux-in matrix
Q_net = np.zeros((nz, ny, nx)) # net heat-flux-in for the substrate


start_timer = time.clock()
curr_time = 0
while curr_time < dt:

    outer_iter = 0
    outer_norm = EPSIT
    outer_update = 100 * outer_norm
    resid_max = 0

    while outer_iter < MAX_OUT_ITER and outer_update > outer_norm:
        curr_time += dt/4



        Q_net = Q_in