# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:40:39 2017
@author: anuj.datar@gmail.com

Selects a toolpath based on the user settings in the main algorithm
"""

from random import randint
import numpy as np

from .raster import raster_path
from .spiral import spiral_path
from .random_path import random_path

def path_select(path_method, x, y, bdry_offset):
    """function to select the tool-path ##########################

    Function to select path based on the input and call the required path
    function to give desired path coordinates

    Args:
        path_method: Path type selection parameter set by main algorithm
        x: Number of nodes in the X-direction
        y: Number of nodes in the Y-direction
        bdry_offset: Boundary offset used in the algorithm to ensure
            feasibility, by adding extra nodes along the boundary

    Returns:
        A matrix containing the tool-path coordinates
    """
    nx = x - (2*bdry_offset)
    ny = y - (2*bdry_offset)

    if path_method == 1:
        path = raster_path(ny, nx)
        ptyp = 'raster'

    elif path_method == 2:
        path = spiral_path(ny, nx)
        ptyp = 'spiral-out'

    elif path_method == 3:
        path = np.zeros((nx*ny, 3))
        path_a = spiral_path(ny, nx)
        path_y = path_a[:, 0]
        path_y = path_y[::-1]
        path_x = path_a[:, 1]
        path_x = path_x[::-1]
        path[:, 0] = path_y
        path[:, 1] = path_x
        path[:, 2] = path_a[:, 2]
        ptyp = 'spiral-in'

    elif path_method == 4:
        path = random_path(ny, nx)
        ptyp = 'random'

    elif path_method == 5:
        ptyp = 'opt'
        path = np.zeros((nx*ny, 3))
        path[0, 0] = randint(0, ny)
        path[0, 1] = randint(0, nx)

    if path_method < 5:
        path[:, 0] = path[:, 0] + bdry_offset
        path[:, 1] = path[:, 1] + bdry_offset
    else:
        path[0, 0] = path[0, 0] + bdry_offset
        path[0, 1] = path[0, 1] + bdry_offset

    return path, ptyp
