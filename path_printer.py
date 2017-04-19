"""Toolpath selector

Author: anuj.dater@gmail.com
Created: 4/15/2017
"""

import numpy as np
import matplotlib.pyplot as plt

from Toolpaths import path_select

x = 5
y = 5

shader = np.zeros((y, x))

#path = raster_path(x,y)
path, ptyp = path_select(2, x, y, 0)
print(path)

for i in range(x*y):
    print(int(path[i, 0]), int(path[i, 1]))
    shader[int(path[i, 0]), int(path[i, 1])] = path[i, 2]

plt.contourf(shader)
plt.jet()
plt.savefig('img.jpg')
print(path)
