'''circuit_test.py: plots the bond dimension distribution of a random distribution over parametrised circuits'''
from xmps.iMPS import iMPS
from qmps.tools import random_circuit, random_qaoa_circuit, unitary_to_tensor, svals
from qmps.tools import random_full_rank_circuit
import numpy as np
import cirq
from numpy import log2
import matplotlib.pyplot as plt
from tqdm import tqdm


N = 100
L = 4
p = 1
As = []
print(random_full_rank_circuit(L, p).to_text_diagram(transpose=True))
for i in range(N):
    U = random_full_rank_circuit(L, p)._unitary_()
    As.append(unitary_to_tensor(U))

Ds = []
for A in tqdm(As):
    try:
        X = np.diag(iMPS([A]).left_canonicalise().L)
    except np.linalg.LinAlgError:
        raise Exception('Not good mps')
    plt.plot(list(range(len(X))), X, marker='x')
    plt.ylim([0, 1])
    Ds.append(len(X[np.logical_not(np.isclose(X, 0))]))

plt.show()

plt.hist(log2(Ds).astype(int), bins=list(range(2, L+1)))
plt.xlabel('$\log_2(D)$')
plt.ylabel('frequency')
plt.title('bond dimension frequency for random p={} CNOT/Rx/Rz circuits'.format(p))
plt.xticks(list(range(1, L)))
plt.xlim([2, L])
#plt.savefig('../images/p{}_random_circuit_bond_dimensions.pdf'.format(p))
plt.show()
