import numpy as np
from numpy.random import randn
from xmps.iMPS import iMPS, Map
from xmps.spin import U4
from qmps.ground_state import Hamiltonian
from qmps.tools import from_real_vector
from itertools import permutations
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from tqdm import tqdm 
plt.style.use('pub_fast')

def Nsphere(v):
    # Spherical coordinates for the (len(v)+1)-sphere
    def sts(v):
        # [a, b, c..] -> [[a], [a, b], [a, b, c], ..]
        return [np.array(v[:b]) for b in range(1, len(v)+1)]
    def cs(v):
        # [[a], [a, b], [a, b, c], ..] -> [prod([cos(a)]), prod([sin(a), cos(b)]), ...]
        return np.prod(np.array([*np.sin(v[:-1]), np.cos(v[-1])]))
    def ss(v):
        # same as cs but with just sines
        return np.prod(np.sin(v))
    return np.array([cs(v) for v in sts(v)]+[ss(v)])

def vec(p):
    """vec

    :param p:
    """
    return from_real_vector(Nsphere(p))
    #return U4(p)[0]

def random_test(N=100, dt=1e-2, max_tries=20, η=0., noisy=False, k=1, tol=1e-10):
    evals = []
    tqer = tqdm(range(N)) if not noisy else range(N)
    for _ in tqer:
        H = Hamiltonian({'ZZ': 1, 'X':1}).to_matrix()

        A = iMPS().random(2, 2).left_canonicalise()
        B = (A + dt*A.dA_dt([H])).left_canonicalise()
        EAB = Map(A[0], B[0])

        def objective(p, EAB=EAB, η=η):
            EAB = EAB.asmatrix()
            er, eim, p1, p2 = p[0], p[1], p[2:17], p[17:]
            e = er+1j*eim
            #v = EAB@vec(p1)-e*vec(p1)
            v = vec(p1)
            return np.real(v.conj().T@EAB.conj().T@EAB@v) + np.abs(e)**2 - 2*np.real(e*v.conj().T@EAB@v)+ η*np.abs((0.99-e))**2
            #return np.real(v.conj().T@v) + η*np.abs((1-e))**2

        def abs_objective(p, EAB=EAB, η=η):
            EAB = EAB.asmatrix()
            er, eim, p1, p2 = p[0], p[1], p[2:17], p[17:]
            e = er+1j*eim
            #v = EAB@vec(p1)-e*vec(p1)
            v = vec(p1)
            return np.real(v.conj().T@EAB.conj().T@EAB@v) + np.abs(e)**2 - 2*np.abs(e)*np.abs(v.conj().T@EAB@v)+ η*np.abs((0.99-e))**2
            #return np.real(v.conj().T@v) + η*np.abs((1-e))**2

        e, r = EAB.right_fixed_point()

        x = randn(9)
        x[0] = 1
        x[1] = 0

        abs_λ = 0
        tries = 0
        eigval_results = []
        eigvec_results = []
        while tries<max_tries and np.abs(abs_λ)<1-k*dt:
            abs_res = minimize(abs_objective, randn(9), method='BFGS', options={'disp': noisy}, tol=tol)
            abs_λ, abs_result_vector = abs_res.x[0]+1j*abs_res.x[1], vec(abs_res.x[2:9])
            tries+=1
            eigval_results.append(np.abs(abs_λ))
            eigvec_results.append(abs_result_vector)
            #print(np.abs(e), np.abs(abs_λ), 1-10*dt**2)
            if not noisy:
                tqer.set_description(f'{tries} repeats, {np.round(np.abs(abs_λ), 3)}')
                if tries==max_tries-1:
                    tqer.set_description('max_iters reached')
                    abs_λ = np.max(eigval_results)
                    abs_result_vector = eigvec_results[np.argmax(eigval_results)]

        def dephase(v):
            return v/np.exp(1j*np.angle(v[0]))

        if noisy:
            print('eigenvectors')
            print('actual: ', r.reshape(-1).real)
            print('variational: ', dephase(result_vector).real)
            print('abs variational: ', dephase(abs_result_vector).real)
            print('\n')
            print('eigenvalues')
            print('actual: ', np.abs(e))
            print('variational: ', np.abs(λ))
            print('abs variational: ', np.abs(abs_λ))
        evals.append(np.abs(e)+1j*np.abs(abs_λ))

    return np.array(evals)

def sequential_test(N=100, dt=1e-2, max_tries=20, η=0., noisy=False, k=1, tol=1e-10):
    evals = []
    tqer = tqdm(range(N)) if not noisy else range(N)
    x = randn(9)
    A = iMPS().random(2, 2).left_canonicalise()
    try_results = []

    for _ in tqer:
        H = Hamiltonian({'ZZ': 1, 'X':1}).to_matrix()

        B = (A + dt*A.dA_dt([H])).left_canonicalise()
        EAB = Map(A[0], B[0])
        e, r = EAB.right_fixed_point()
        A = B

        def objective(p, EAB=EAB, η=η):
            EAB = EAB.asmatrix()
            er, eim, p1, p2 = p[0], p[1], p[2:17], p[17:]
            e = er+1j*eim
            #v = EAB@vec(p1)-e*vec(p1)
            v = vec(p1)
            return np.real(v.conj().T@EAB.conj().T@EAB@v) + np.abs(e)**2 - 2*np.real(e*v.conj().T@EAB@v)+ η*np.abs((0.99-e))**2
            #return np.real(v.conj().T@v) + η*np.abs((1-e))**2

        def abs_objective(p, EAB=EAB, η=η):
            EAB = EAB.asmatrix()
            er, eim, p1, p2 = p[0], p[1], p[2:17], p[17:]
            e = er+1j*eim
            #v = EAB@vec(p1)-e*vec(p1)
            v = vec(p1)
            return np.real(v.conj().T@EAB.conj().T@EAB@v) + np.abs(e)**2 - 2*np.abs(e)*np.abs(v.conj().T@EAB@v)+ η*np.abs((0.99-e))**2
            #return np.real(v.conj().T@v) + η*np.abs((1-e))**2

        abs_λ = 0
        tries = 0
        eigval_results = []
        eigvec_results = []
        while tries<max_tries and np.abs(abs_λ)<1-k*dt:
            abs_res = minimize(abs_objective, x, method='BFGS', options={'disp': noisy}, tol=tol)
            abs_λ, abs_result_vector = abs_res.x[0]+1j*abs_res.x[1], vec(abs_res.x[2:9])
            tries+=1
            eigval_results.append(np.abs(abs_λ))
            eigvec_results.append(abs_result_vector)
            x = randn(9)
            #print(np.abs(e), np.abs(abs_λ), 1-10*dt**2)
            if not noisy:
                tqer.set_description(f'{tries} repeats, {np.round(np.abs(abs_λ), 3)}')
                if tries==max_tries-1:
                    tqer.set_description('max_iters reached')
                    abs_λ = np.max(eigval_results)
                    abs_result_vector = eigvec_results[np.argmax(eigval_results)]
        x = abs_res.x
        try_results.append(tries)

        def dephase(v):
            return v/np.exp(1j*np.angle(v[0]))

        if noisy:
            print('eigenvectors')
            print('actual: ', r.reshape(-1).real)
            print('variational: ', dephase(result_vector).real)
            print('abs variational: ', dephase(abs_result_vector).real)
            print('\n')
            print('eigenvalues')
            print('actual: ', np.abs(e))
            print('variational: ', np.abs(λ))
            print('abs variational: ', np.abs(abs_λ))
        evals.append(np.abs(e)+1j*np.abs(abs_λ))

    return np.array(evals), np.array(try_results)

if __name__=='__main__':
    N, dt, max_tries, k, tol = 2000, 1e-2, 50, 1, 1e-10
    #evals = random_test(N, dt, max_tries=max_tries, k=k, tol=tol)
    evals, tries = sequential_test(N, dt, max_tries=max_tries, k=k, tol=tol)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.scatter(evals.real, evals.imag, marker='+')
    ax.set_xlabel('true eigenvalue')
    ax.set_ylabel('variational eigenvalue')
    ax.set_xlim([1-0.1*k*dt, 1])
    ax.set_ylim([1-0.1*k*dt, 1])
    ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), linestyle='--', c='black')
    plt.savefig('scatter.pdf', bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.scatter(list(range(len(tries))), tries, marker='+')
    ax.set_ylabel('number of repetitions')
    ax.set_xlabel('timestep')
    plt.savefig('repetitions.pdf')
    plt.show()

    import seaborn as sns
    sns.set()
    sns.set_style('whitegrid')
    res = evals.imag-evals.real

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    sns.distplot(res, kde=False, ax=ax)
    ax.set_xlim([-max(np.abs(res)), max(np.abs(res))])
    ax.set_xlabel('$|\\eta_{\mathrm{exact}}|-|\\eta_{\mathrm{variational}}|$')
    ax.set_ylabel('frequency')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.savefig('error_hist.pdf')
    
    plt.show()
