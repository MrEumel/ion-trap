import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

starting_positions = np.loadtxt('starting_positions.txt')
np.seterr('raise')

"""Constants"""

r_0 = 0.003                          # [m]
r_laser = 0.0005                     # [m]
e = 1.602 * 10**(-19)                # [C]
amu = 550                            # [amu]
m = amu * 1.66054 * 10 ** (-27)      # [kg]
epsilon = 8.854187 * 10**(-12)       # [As/Vm]
beta = 1e-19 / m                     # Friction coefficient
U = 40
f = 5e5
OMEGA = f * 2 * np.pi

Pot_Const = 2 * (e / m) * (U / (r_0 ** 2))
Coul_Const = (e ** 2) / (4 * np.pi * epsilon * m)

q_factor = (4 * e * U) / (m * (r_0 ** 2) * (OMEGA ** 2))
V_eff = q_factor * U / 4
delta_E = 2 * U * ((r_laser/r_0)**2)



"""Integration parameters"""
spp = 49              # steps per period
T = 1 / f
per_period = T / spp

dt = per_period
steps = spp * 100     # spp * periods per integration

"""Set up ionsystem"""

number_of_ions = len(starting_positions)

t = np.zeros(steps+2)
x = np.zeros((number_of_ions, steps+2))
y = np.zeros((number_of_ions, steps+2))
vx = np.zeros((number_of_ions, steps+2))
vy = np.zeros((number_of_ions, steps+2))
ax = np.zeros((number_of_ions, steps+2))
ay = np.zeros((number_of_ions, steps+2))

E_kin = np.zeros((number_of_ions, steps+2))

"""Starting conditions"""

for i in range(len(starting_positions)):
    x[i, 0] = starting_positions[i, 0]
    y[i, 0] = starting_positions[i, 1]

"""Integration"""

for i in tqdm(range(0, steps+1)):
    t[i+1] = i * dt

    for k in range(0, number_of_ions):
        r_k = np.sqrt(x[k, i] ** 2 + y[k, i] ** 2)
        if r_k > 0.003:
            x[k, i + 1] = x[k, i]
            y[k, i + 1] = y[k, i]

        else:
            vx[k, i+1] = vx[k, i] + ax[k, i] * dt
            vy[k, i+1] = vy[k, i] + ay[k, i] * dt

            x[k, i+1] = x[k, i] + vx[k, i+1] * dt
            y[k, i+1] = y[k, i] + vy[k, i+1] * dt

            E_kin[k, i+1] = ((np.sqrt(vx[k, i+1] ** 2 + vy[k, i+1] ** 2)) ** 2) * m / 2

    for k in range(0, number_of_ions):
        r_k = np.sqrt(x[k, i] ** 2 + y[k, i] ** 2)
        if r_k > 0.003:
            continue

        else:
            ax[k, i+1] += - Pot_Const * np.cos(OMEGA * t[i+1]) * x[k, i+1]
            ay[k, i+1] += - Pot_Const * np.cos(OMEGA * t[i+1]) * y[k, i+1]
            ax[k, i+1] += - beta * vx[k, i+1]
            ay[k, i+1] += - beta * vy[k, i+1]

            for j in range(0, number_of_ions):
                r_j = np.sqrt(x[j, i] ** 2 + y[j, i] ** 2)
                if r_j > 0.003:
                    continue
                else:
                    if j == k:
                        continue
                    else:
                        dxy = np.sqrt((x[k, i+1] - x[j, i+1]) ** 2 + (y[k, i+1] - y[j, i+1]) ** 2) ** 3
                        ax[k, i+1] += (Coul_Const * (x[k, i+1] - x[j, i+1]) / dxy)
                        ay[k, i+1] += (Coul_Const * (y[k, i+1] - y[j, i+1]) / dxy)

"""Plot trajectories in x-y-plane"""

for k in range(0, number_of_ions):
    plt.plot(x[k, steps-1000:steps], y[k, steps-1000:steps], linewidth=1)
plt.grid()
plt.axis('scaled')
circle = plt.Circle((0, 0), 0.003, fill=False)
ax = plt.gca()
ax.set_xlim((-0.00004, 0.00004))
ax.set_ylim((-0.00004, 0.00004))
ax.add_artist(circle)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


