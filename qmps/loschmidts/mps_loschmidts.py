from xmps.fTDVP import Trajectory
from xmps.Hamiltonians import Hamiltonian
from xmps.tdvp.tdvp_fast import MPO_TFI
from xmps.fMPS import fMPS
from xmps.iMPS import iMPS
from xmps.spin import paulis
from xmps.iOptimize import find_ground_state
from scipy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
X, Y, Z = paulis(0.5)

D = 2

g0, g1 = 0.5, 1.5
h0 = Hamiltonian({'ZZ':-1, 'X':g0}).to_matrix()
h1 = Hamiltonian({'ZZ':-1, 'X':g1}).to_matrix()
A, es = find_ground_state(h0, 2, tol=1e-2, noisy=True)

T = np.linspace(0, 6, 300)
traj = Trajectory(mps_0=A, H=h1)
traj.rk4int(T)
ls = traj.loschmidts()
plt.plot(ls)
plt.show()
