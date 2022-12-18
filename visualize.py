#!/usr/bin/env python

from bodge import *

import seaborn as sns
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 1)
for ax in axs:
    ax.set_aspect('equal')
    ax.set_axis_off()

L_SC = 10
L_NM = 3
L_AM = 10

L_X = 2 * L_SC + 2*L_NM + L_AM
L_Y = 11
L_Z = 1

lattice = CubicLattice((L_X,L_Y,L_Z))

for i in lattice.sites():
    if i[0] + (i[1] - L_Y//2) < L_SC or i[0] + (i[1] - L_Y//2) >= L_X - L_SC:
        axs[0].scatter(x=i[0], y=i[1], color='#ff7f00')
    elif i[0] + (i[1] - L_Y//2) < (L_SC + L_NM) or i[0] + (i[1] - L_Y//2) >= L_X - (L_SC + L_NM):
        axs[0].scatter(x=i[0], y=i[1], color='k')
    else:
        axs[0].scatter(x=i[0], y=i[1], color='#984ea3')

for i in lattice.sites():
    if i[0] < L_SC or i[0] >= L_X - L_SC:
        axs[1].scatter(x=i[0], y=i[1], color='#ff7f00')
    elif i[0] < L_SC + L_NM or i[0] >= L_X - L_SC - L_NM:
        axs[1].scatter(x=i[0], y=i[1], color='k')
    else:
        axs[1].scatter(x=i[0], y=i[1], color='#984ea3')

plt.show()