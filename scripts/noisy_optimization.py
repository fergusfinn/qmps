
from xmps.iOptimize import find_ground_state
from qmps.ground_state import SparseFullEnergyOptimizer, NoisySparseFullEnergyOptimizer
from qmps.represent import ExactAfter4, ShallowCNOTStateTensor, ShallowCNOTStateTensor_nonuniform
from scipy import integrate
import numpy as np
import numpy.random as ra
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('pub_fast')

D = 2

gs = np.linspace(1, 2, 1)
gate = ShallowCNOTStateTensor_nonuniform
w = gate.params_per_iter(D)
print(w)

import itertools

marker = itertools.cycle(['*', 'x', '+', '.'])
ps = range(1, 9, 1)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
initial_guess = ra.randn(w*ps[0])

exact_gses = [0]
qmps_gses = [0]
max_tries = 5
for i, p in enumerate(ps):

    if initial_guess is not None:
        initial_guess = np.concatenate([initial_guess, np.random.randn(w*p-len(initial_guess))])

    print(f'p={p}')
    while True:
        tries = []
        J, g = -1, 1
        f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
        exact_gse = integrate.quad(f, 0, np.pi, args=(g,))[0]
        H =  np.array([[J,g/2,g/2,0],
                       [g/2,-J,0,g/2],
                       [g/2,0,-J,g/2],
                       [0,g/2,g/2,J]] )


        opt = SparseFullEnergyOptimizer(H, D, p, state_tensor=gate, initial_guess=initial_guess)
        sets = opt.settings
        sets['verbose'] = False
        opt.is_verbose = False
        sets['store_values'] = True
        #sets['method'] = 'Nelder-Mead'
        #sets['maxiter'] = 1500
        sets['tol'] = 1e-8
        opt.change_settings(sets)
        print('starting optimization: ... ')
        opt.optimize()
        print('optimization over')

        #initial_guess = opt.optimized_result.x
        if opt.optimized_result.fun <= qmps_gses[-1] and len(tries)<max_tries:
            print(f'{opt.optimized_result.fun}, {qmps_gses[-1]}: success')
            qmps_gse = max(tries) if len(tries)==max_tries else opt.optimized_result.fun
            qmps_gses.append(qmps_gses)
            print(qmps_gses)
            exact_gses.append(exact_gse)
            ax.scatter(gs, qmps_gses[-1]-exact_gses[-1], marker=next(marker), s=65, label=f'{p}')
            break
        else: 
            print(f'{opt.optimized_result.fun}, {qmps_gses[-1]}: trying again ...')
            tries.append(opt.optimized_result.fun)
            initial_guess = np.random.randn(len(initial_guess))

ax.set_xlabel('$\\lambda$')
ax.set_ylabel('$E_0$')
#ax.scatter(gs, es[-1]-es_, marker='+', label='D=2')
#ax.scatter(gs, np.array(exact_es)[-1], marker='o', label='analytical result')
ax.legend()
#plt.savefig('error_vs_p.pdf', bbox_inches='tight')
#plt.title('D=2 Ising ground state energy', loc='right')


plt.show()

from scipy.linalg import norm
x = [norm(x) for x in np.array(exact_gses)-np.array(qmps_gses)]

fig, ax = plt.subplots(1, 1)
ax.scatter(ps, x[1:], marker='x', s=45, c='C1')
ax.plot(ps, x[1:], c='C1')
ax.set_xlabel('p')
ax.set_ylabel('$\epsilon$')

D2_gse = -1.269909412573
ax.axhline(D2_gse-exact_gses[-1], linestyle='--', label='D=2', color='gray')

ax.set_yscale('log')
#plt.savefig('total_error_vsp.pdf')
#plt.title('Deviation from exact $E_0$ curve vs. p', loc='right')
plt.show()
