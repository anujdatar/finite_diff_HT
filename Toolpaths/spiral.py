# -*- coding: utf-8 -*-
"""Generates a spiral type tool-path for a given grid.

Starts in the center and spirals outwards.

@author: anuj.datar@gmail.com
"""

import math
import numpy as np

def spi_dirn(dire, nc):
    """Function to define direction in a spiral tool-path.

    Function to find new direction for laser scan, and the new dx & dy
    this function only used by the spiral_path() function.

    Args:
        dire: Previous direction of the tool-path
        nc: Loop checker or counter that makes sure the the program does not
            get stuck in an infinite loop.

    Returns:
        x: Increment in the X-direction for the next point in the path
        y: Increment in the Y-direction for the next point in the path
        dirn: New direction for the tool path, +/- x/y
        c: New value for the loop checker
    """

    if dire == 1:               # if dir north, check west
        y = 0
        x = -1
        dirn = 2
        c = nc + 1              # the loop checker
    elif dire == 2:             # if dir west, check south
        y = 1
        x = 0
        dirn = 3
        c = nc + 1
    elif dire == 3:             # if dir south, check east
        y = 0
        x = 1
        dirn = 4
        c = nc + 1
    else:                       # if dir east, check north
        y = -1
        x = 0
        dirn = 1
        c = nc + 1

    return x, y, dirn, c


##############################################################################


def spiral_path(m, n):
    """Function to define a raster tool-path.

    define a spiral path based on input grid size at present it only works
    for odd number of grid points, i.e. even number of grid spaces
    also, can only do square grids, x = y

    Args:
        m: Number of grid points in the Y-direction
        n: Number of gridpoints in the X-direction

    Returns:
        tpath: A random tool-path, with a shader to plot in contourf
    """
    nm = n*m
    nm1 = (n-1) * (m-1)

    # matrices for xand y coordinates of path
    px = np.zeros(nm)
    py = np.zeros(nm)
    sm = np.zeros(nm)
    # output path matrix
    tpath = np.zeros((nm, 3))

    # find center of grid for starting point of the spiral
    xmid = math.ceil(n/2)
    ymid = math.ceil(m/2)
    px[0] = xmid
    py[0] = ymid
    sm[0] = 1 # shader for coloring the graph

    # set initial direction and i = starting location in loop
    dirn = 1 # initial directions
    ndirn = dirn # new direction variable
    tc = 0 # counter in direction function to prevent infinite loops
    i = 1 # start at this element for the o/p matrix
    exists = 0 # checker for existing element combinations
    shader = 1 # initial value for shader

    while  (tc < 4) & (i < nm):
        # save old direction value and enter new direction
        olddirn = dirn
        dirn = ndirn

        # use the dirn function to find dx, dy, new direction and direction
        # tc is the loop checker
        dx, dy, ndirn, tc = spi_dirn(dirn, tc)
#        print(ndirn,tc)

        # find new x and y coordinates for spiral
        nx = px[i-1] + dx
        ny = py[i-1] + dy

        # use this in case odd number of grid spacings are entered
        ## this causes laser overlap on one segment, but the solution is still
        ### feasible (at least)
        if i == nm1-1:
            nx1 = nx
            ny1 = ny

        # check if coordinate already exists in the path matrix
        found_y = []             # to store row index of y positions found
        row_pos = 0             # temporary variable for storing row position

        for row in py:
            if row == ny:
                found_y.append(row_pos)
            row_pos += 1

        if len(found_y) > 0:
            for row in found_y:
                if px[row] == nx:
                    exists = 1

        if exists == 0:
            if (nx > 0) & (ny > 0):
                shader = shader + 1
                px[i] = nx
                py[i] = ny
                sm[i] = shader
                tc = 0
                i += 1
            else:
                px[i-1] = nx
                py[i-1] = ny
                tc = 0
        else:
            ndirn = olddirn
            exists = 0

    px[nm1-1] = nx1
    py[nm1-1] = ny1

    for k in range(0, nm):
        tpath[k, 0] = py[k] - 1
        tpath[k, 1] = px[k] - 1
        tpath[k, 2] = sm[k]

    return tpath
