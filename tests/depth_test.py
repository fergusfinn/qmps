from xmps.iMPS import iMPS
from qmps.tools import test_unitary, unitary_to_tensor, svals
import numpy as np
import cirq
import matplotlib.pyplot as plt
from tqdm import tqdm


N = 20
L = 8
p = 1
As = []
for i in range(N):
    U = test_unitary(L, 1)._unitary_()
    As.append(unitary_to_tensor(U))

Ds = []
for A in tqdm(As):
    _, _, C = iMPS([A]).mixed()
    plt.scatter(list(range(len(svals(C)))), svals(C), marker='x')
    Ds.append(len(svals(C)[np.logical_not(np.isclose(svals(C), 0))]))

plt.show()

plt.hist(Ds)
plt.show()


