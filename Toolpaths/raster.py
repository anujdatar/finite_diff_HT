"""Raster tool-path generation

Author: anuj.datar@gmail.com
Created: 4/15/2017
"""

# standard library imports
import numpy as np

def raster_path(m, n):
    """Defines a set of coordinates for a raster tool-path

    Args:
        m: Number of grid points in the Y-direction
        n: Number of grid points in the X-direction
    Returns:
        tpath: raster tool-path coordinates with a shader column for plotting
    """
    nm = n * m

    # define empty matrices
    #pathtemp = np.zeros((nm, 2))
    tpath = np.zeros((nm, 3))

    shader = 0

    ### set Y-coordinates
    # using the dirn variable to make sure toolpath rasters and is not unidirectional
    direction = 1
    for i in range(n):
        for j in range(1, m+1):
            y_index = i*m + j - 1
            if direction == 1:
                tpath[y_index, 0] = j - 1
                if j == m:
                    direction = -1
            else:
                tpath[y_index, 0] = m - j
                if j == m:
                    direction = 1

    ### set X-direction
    for i in range(n):
        for j in range(1, m+1):
            x_index = i*m + j - 1
            tpath[x_index, 1] = i

    for i in range(nm):
        shader += 1
        tpath[i, 2] = shader

    return tpath
