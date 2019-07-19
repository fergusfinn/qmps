'''circuit_test.py: plots the bond dimension distribution of a random distribution over parametrised circuits'''
from xmps.iMPS import iMPS
from qmps.tools import random_circuit, random_qaoa_circuit, unitary_to_tensor, svals
import numpy as np
import cirq
from numpy import log2
import matplotlib.pyplot as plt
from tqdm import tqdm


N = 200
L = 6
p = 4
As = []
for i in range(N):
    U = random_circuit(L, p)._unitary_()
    As.append(unitary_to_tensor(U))

Ds = []
for A in tqdm(As):
    _, _, C = iMPS([A]).mixed()
    #plt.scatter(list(range(len(svals(C)))), svals(C), marker='x')
    Ds.append(len(svals(C)[np.logical_not(np.isclose(svals(C), 0))]))

plt.show()

plt.hist(log2(Ds).astype(int), bins=list(range(2, L+1)))
plt.xlabel('$\log_2(D)$')
plt.ylabel('frequency')
plt.title('bond dimension frequency for random p={} CNOT/Rx/Rz circuits'.format(p))
plt.xticks(list(range(1, L)))
plt.xlim([2, L])
plt.savefig('../images/p{}_random_circuit_bond_dimensions.pdf'.format(p))
plt.show()
