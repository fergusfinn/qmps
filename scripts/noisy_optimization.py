"""How does sparsity + noise affect optimization performance"""
from qmps.ground_state import SparseFullEnergyOptimizer, NoisySparseFullEnergyOptimizer
from scipy import integrate
import numpy as np
import numpy.random as ra
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('pub_fast')

D = 2

gs = np.linspace(0.1, 2, 8)
exact_es = []
qmps_es = []

import itertools

marker = itertools.cycle(['x', '.', ',', '+', '*'])
ps = range(1, 6, 1)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
for i, p in enumerate(ps):
    if i==0:
        initial_guess = ra.randn(2*p)

    es = []
    es_ = []
    if initial_guess is not None and len(initial_guess)!=2*p:
        initial_guess = np.concatenate([initial_guess, np.zeros(2*p-len(initial_guess))])
    for j, g in tqdm(enumerate(gs)):
        J, g = -1, g
        f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
        E0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
        es_.append(E0_exact)
        H =  np.array([[J,g/2,g/2,0],
                       [g/2,-J,0,g/2],
                       [g/2,0,-J,g/2],
                       [0,g/2,g/2,J]] )


        opt = NoisySparseFullEnergyOptimizer(H, 1e-4,  D, p, initial_guess=initial_guess)
        sets = opt.settings
        sets['store_values'] = True
        sets['method'] = 'Nelder-Mead'
        sets['maxiter'] = 700
        sets['tol'] = 1e-6
        opt.change_settings(sets)
        opt.optimize()

        initial_guess = opt.optimized_result.x

        es.append(opt.obj_fun_values[-1])

    ax.scatter(gs, es, marker=next(marker), s=35)
    ax.plot(gs, es, label=str(p))

    qmps_es.append(es)
    exact_es.append(es_)
    np.save(f'{p}', es)

ax.set_xlabel('$\\lambda$')
ax.set_ylabel('$E_0$')
ax.plot(gs, np.array(exact_es)[-1], linestyle='--', label='analytical result')
ax.legend()
plt.savefig('error_vs_p.pdf', bbox_inches='tight')
#plt.title('D=2 Ising ground state energy', loc='right')


plt.show()

from scipy.linalg import norm
x = [norm(x) for x in np.array(exact_es)-np.array(qmps_es)]

fig, ax = plt.subplots(1, 1)
ax.scatter(ps, x, marker='x', s=45)
ax.plot(ps, x)
ax.set_xlabel('p')
ax.set_ylabel('$\epsilon$')

plt.savefig('total_error_vsp.pdf')
#plt.title('Deviation from exact $E_0$ curve vs. p', loc='right')
plt.show()
