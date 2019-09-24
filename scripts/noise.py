"""How does sparsity affect optimization performance"""
from qmps.ground_state import NoisySparseFullEnergyOptimizer
from scipy import integrate
import numpy as np
import numpy.random as ra
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('pub_fast')

D = 4

gs = np.linspace(0.1, 2, 10)
exact_es = []
qmps_es = []

import itertools

noise_strength = 5e-4
marker = itertools.cycle(['x', '.', ',', '+', '*'])
ps = range(2, 7)
gs = np.linspace(0.1, 2, 10)
for i, p in enumerate(ps):
    if i==0:
        first_initial_guess = ra.randn(2*p)
    initial_guess = first_initial_guess

    es = []
    es_ = []
    if initial_guess is not None and len(initial_guess)!=2*p:
        initial_guess = np.concatenate([np.zeros(2*p-len(initial_guess)), initial_guess])
    for j, g in tqdm(enumerate(gs)):
        J, g = -1, g
        f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
        E0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
        es_.append(E0_exact)
        H =  np.array([[J,g/2,g/2,0], 
                       [g/2,-J,0,g/2], 
                       [g/2,0,-J,g/2], 
                       [0,g/2,g/2,J]] )


        opt = NoisySparseFullEnergyOptimizer(H, noise_strength, D, p, initial_guess=initial_guess)
        sets = opt.settings
        sets['store_values'] = True
        sets['method'] = 'Powell'
        sets['maxiter'] = 20
        sets['tol'] = 1e-5
        sets['bayesian'] = True
        opt.change_settings(sets)
        opt.optimize()
        print(g)

        initial_guess = opt.optimized_result.x

        if j==0:
            first_initial_guess = initial_guess

        es.append(opt.optimized_result.fun)

    plt.scatter(gs, es, marker=next(marker), s=35)
    plt.plot(gs, es, label=str(p))
    plt.xlabel('$\\lambda$')
    plt.ylabel('$E_0$')
    qmps_es.append(es)
    exact_es.append(es_)

#plt.plot(np.array(qmps_es).T)
plt.plot(gs, np.array(exact_es)[-1], label='exact')
plt.legend()
#plt.savefig('/Users/fergusbarratt/Dropbox/PhD/google_quantum/qmps/images/sparsity/errorvsp.pdf', bbox_inches='tight')
plt.title('D=4 ground state energy vs. p', loc='right')
plt.show()

from scipy.linalg import norm
x = [norm(x) for x in np.array(exact_es)-np.array(qmps_es)]

plt.scatter(ps, x, marker='x', s=35)
plt.plot(ps, x)
plt.xlabel('p')
plt.ylabel('$\epsilon$')
#plt.savefig('/Users/fergusbarratt/Dropbox/PhD/google_quantum/qmps/images/sparsity/total_errorvsp.pdf')
plt.title('Deviation from exact $E_0$ curve vs. p', loc='right')
plt.show()
