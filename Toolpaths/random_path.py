# -*- coding: utf-8 -*-
"""
Generates a random set of coordinates for a given grid.

@author: anuj.datar@gmail.com
"""

from random import randint
import numpy as np

def random_path(x, y):
    """Generates a random set of coordinates for a given grid.

    Args:
        x: Number of grid points in the X-direction
        y: Number of grid points in the Y-direction

    Returns:
        tpath: A random tool-path, with a shader to plot in contourf
    """

    nm = x * y

# define initial empty matrices
    x_temporary = np.zeros(nm)
    y_temporary = np.zeros(nm)
    tpath = np.zeros((nm, 3))

# using this section to get rid of the zeros. because one of the indices
# is supposed to be 0. and the check loop gets stuck in an infinite loop.
    num_mat = np.zeros(nm)+1

    shader = 1  # for shading the path on a contourf plot

# generate a list of all possible grid points
    row_no = 0 # temporary variable for row number indexing
    for j in range(0, y):
        for i in range(0, x):
            y_temporary[row_no] = j
            x_temporary[row_no] = i
            row_no += 1

# generate a random list of rows to draw gridpoints from
    inc_var = 0
    while inc_var < nm:
        nu = randint(0, nm-1)
        chk = np.where(num_mat == nu)[0]
#        print(nu, chk)
        if len(chk) == 0:
            num_mat[inc_var] = nu
#            print(nu,chk)
            inc_var += 1

    for i in range(0, nm):
        tempo = num_mat[i]
        tpath[i, 0] = y_temporary[tempo]
        tpath[i, 1] = x_temporary[tempo]
        tpath[i, 2] = shader
        shader += 1

    return tpath
