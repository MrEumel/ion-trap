import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""TO-DO:

- Create matrix of 2D arrays with amount of steps columns and number of ions rows
- Introduce Coulomb-Force and code iteration through all ion pairings

"""

"""Constants"""

U = 6                                # [V]
r_0 = 0.003                          # [m]
r_laser = 0.0005                     # [m]
f_0 = 2.5e6                          # [Hz]
e = 1.602 * 10**(-19)                # [C]
amu = 4                              # [amu]
m = amu * 1.66054 * 10 ** (-27)      # [kg]
epsilon = 8.854187 * 10**(-12)       # [As/Vm]
beta = 1e-20 / m                     # Friction coefficient
OMEGA = f_0 * 2 * np.pi              # [1/s]

Pot_Const = 2 * (e / m) * (U / (r_0 ** 2))

q_faktor = (4 * e * U) / ((m) * (r_0 ** 2) * (OMEGA ** 2))
V_eff = q_faktor * U / 4
delta_E = 2 * U * ((r_laser/r_0)**2)

"""Starting conditions"""

dt = 1e-8
steps = 10000
t = np.zeros(steps)

"""Ion 1"""

x = np.zeros((2, steps))
v = np.zeros((2, steps))
a = np.zeros((2, steps))

E_kin = np.zeros(steps)

x_start = np.array([0.0002, 0])
v_start = np.array([0, 200])

x[:, 0] = x_start
v[:, 0] = v_start

"""Integration"""

for i in tqdm(range(1, steps)):
    t[i] = i * dt

    a[0, i] = - Pot_Const * np.cos(OMEGA * t[i]) * x[0, i-1]
    a[1, i] = + Pot_Const * np.cos(OMEGA * t[i]) * x[1, i-1]
    #a[:, i] += - beta * v[:, i-1]      -> friction force

    v[:, i] = v[:, i-1] + a[:, i-1] * dt

    x[:, i] = x[:, i-1] + v[:, i] * dt

    E_kin[i] = ((np.fabs(np.linalg.norm(v[:, i]))) ** 2) * m / 2


"""Results & Plots"""

print('delta E: ' + str(delta_E) + ' [eV]')
print('q-Factor: ' + str(q_faktor))
print('effective Potential: ' + str(V_eff) + ' [eV]')

plt.plot(x[0], x[1])
plt.grid()
plt.axis('scaled')
plt.xlim(- 0.0003, 0.0003)
plt.ylim(- 0.0003, 0.0003)
plt.show()