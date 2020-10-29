from xmps.iTDVP import Trajectory
from xmps.Hamiltonians import Hamiltonian
from xmps.tdvp.tdvp_fast import MPO_TFI
from xmps.iMPS import iMPS
from xmps.spin import paulis
from xmps.iOptimize import find_ground_state
from scipy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
from exact_loschmidt import loschmidts
X, Y, Z = paulis(0.5)

D = 5

g0, g1 = 0.2, 1.5
h0 = Hamiltonian({'ZZ':-1, 'X':g0}).to_matrix()
h1 = Hamiltonian({'ZZ':-1, 'X':g1}).to_matrix()
A, es = find_ground_state(h0, D, tol=1e-3, noisy=True)

T = np.linspace(0, 2, 2000)
traj = Trajectory(mps_0=A, H=[h1])
traj.eulerint(T)
ls = traj.loschmidts()

plt.plot(T, ls)
plt.plot(T, loschmidts(T, g0, g1))
plt.savefig('loschmidts.pdf')
