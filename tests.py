"""Just random tests for all the modules"""

import numpy as np
from Matrices import source_time

tht = np.ones(((2, 5, 5)))
Q_net = np.zeros(((2, 5, 5)))

rho = 4450
Cp = 564
dt = 0.05

print(rho*Cp/dt)

Q_tim = source_time(Q_net, tht, rho, Cp, dt)

print(Q_tim)
