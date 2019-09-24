"""How does bond dimension affect optimization performance"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools

from qmps.ground_state import Hamiltonian, NonSparseFullEnergyOptimizer

# insu2N takes a vector in su(n) and returns the same vector in su(2n) 
# (i.e. if you trace out a middle qubit you get the same u)
from xmps.spin import insu2N, swap, extractv, SU
from numpy import kron
from functools import reduce

mpl.style.use('pub_fast')

XY = Hamiltonian({'XX': 1, 'YY': 1}).to_matrix()

es = []
Ds = (2**np.arange(1, 5)).astype(int)
initial_guess = None

def fixindices(v, ϵ = 1e-2):
    # don't feed the parameters straight into the 
    # lie algebra but add a swap and perturb slightly (ϵ)
    # necessary to embed D=2 result into D=4 &c. 
    # swap makes i, j have same tensor product structure
    # perturbation gets away from singular points
    N = int(np.sqrt(len(v)+1))
    U = SU(v+ϵ, N)
    n_qubits = int(np.log2(N))+1
    S12 = reduce(np.kron, [np.eye(2)]*(n_qubits-3)+[swap()])
    U = U@S12
    return extractv(U)

for i, D in enumerate(Ds):
    opt = NonSparseFullEnergyOptimizer(XY, D, initial_guess=initial_guess)
    opt.objective_function
    sets = opt.settings
    sets['store_values'] = True
    sets['method'] = 'Nelder-Mead'
    # 
    #sets['maxiter'] =  
    sets['tol'] = 1e-5
    opt.change_settings(sets)
    opt.optimize()

    es.append(opt.obj_fun_values[-1])
    initial_guess = fixindices(insu2N(opt.optimized_result.x))

marker = itertools.cycle(['x', '.', ',', '+', '*'])
plt.scatter(Ds, es, marker=next(marker), s=35)
plt.plot(Ds, es)
Ds_es = np.concatenate([np.expand_dims(np.array(Ds), 0), np.expand_dims(np.array(es), 0)], 0)
np.save('../images/figures/bond_dimension/data', Ds_es)
plt.xlabel('$E_0$')
plt.xlabel('$D$')
plt.savefig('../images/figures/bond_dimension/convergence_with_bond_dimension.pdf', bbox_inches='tight')
plt.show()
