import numpy as np
from xmps.iMPS import iMPS
from xmps.spin import paulis, N_body_spins
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('pub_fast')
X, Y, Z = paulis(0.5)
XI, YI, ZI = N_body_spins(0.5, 1, 2)
IX, IY, IZ = N_body_spins(0.5, 2, 2)

T = np.linspace(0, 1, 100)
dt = T[1]-T[0]
D = 10
mps = iMPS().random(2, D).left_canonicalise()
evs = []

H = -(XI@IX+YI@IY+ZI@IZ)

for _ in T:
    k1 = mps.dA_dt([H])*dt
    k2 = (mps+k1/2).dA_dt([H])*dt
    k3 = (mps+k2/2).dA_dt([H])*dt
    k4 = (mps+k3).dA_dt([H])*dt
    mps = (mps+(k1+2*k2+2*k3+k4)/6).left_canonicalise()
    evs.append(mps.Es(paulis(0.5)))
plt.plot(evs)
#plt.scatter(T, evs, marker = 'x')
plt.show()


