import numpy as np
from xmps.spin import paulis, CNOT, swap, H, CZ, CRy
from scipy.linalg import expm
from qmps.tools import split_ns, unitary_to_tensor
from qmps.tools import get_env_exact
from functools import reduce
from scipy import integrate
import matplotlib.pyplot as plt

from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.models.spins import SpinModel
from tenpy.algorithms import dmrg

plt.style.use('pub_slow')
X, Y, Z = paulis(0.5)
I = np.eye(2)

def D2_gse(g):
    def example_DMRG_heisenberg_xxz_infinite(Jx=1, Jy=1, Jz=1, hx=0, hy=0, hz=0, conserve='best', verbose=False, chi_max=100, S=0.5):
        if verbose:
            print("infinite DMRG, Heisenberg XXZ chain")
            print("Jz={Jz:.2f}, conserve={conserve!r}".format(Jz=Jz, conserve=conserve))
        model_params = dict(
            L=2,
            S=S,  # spin 1/2
            Jx=Jx,
            Jy=Jy,
            Jz=Jz,  # couplings
            hx=hx,
            hy=hy,
            hz=hz,
            bc_MPS='infinite',
            conserve=conserve,
            verbose=verbose)
        M = SpinModel(model_params)
        product_state = ["up", "up"]  # initial Neel state
        psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
        dmrg_params = {
            'mixer': True,  # setting this to True helps to escape local minima
            'trunc_params': {
                'chi_max': chi_max,
                'svd_min': 1.e-10,
            },
            'max_E_err': 1.e-10,
            'verbose': verbose,
        }
        info = dmrg.run(psi, M, dmrg_params)
        E = info['E']
        if verbose:
            print("E = {E:.13f}".format(E=E))
            print("final bond dimensions: ", psi.chi)
        Sz = psi.expectation_value("Sz")  # Sz instead of Sigma z: spin-1/2 operators!
        mag_z = np.mean(Sz)
        if verbose:
            print("<S_z> = [{Sz0:.5f}, {Sz1:.5f}]; mean ={mag_z:.5f}".format(Sz0=Sz[0],
                                                                         Sz1=Sz[1],
                                                                         mag_z=mag_z))
        # note: it's clear that mean(<Sz>) is 0: the model has Sz conservation!
        if verbose:
            print("correlation length:", psi.correlation_length())
        corrs = psi.correlation_function("Sz", "Sz", sites1=range(10))
        if verbose:
            print("correlations <Sz_i Sz_j> =")
            print(corrs)
        return E, psi, M
    e2, _, _ = example_DMRG_heisenberg_xxz_infinite(Jx=0, Jy=0, Jz=-4., hx=-2*g, hy=0, hz=0, chi_max = 2, S=0.5)
    return e2

def e(g):
    f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
    return integrate.quad(f, 0, np.pi, args=(g,))[0]

def Rx(θ):
    return expm(-1j*θ*X/2)

def Ry(θ):
    return expm(-1j*θ*Y/2)

def Rz(θ):
    return expm(-1j*θ*Z/2)

def ansatz(p, n=2):
    U = np.eye(4)
    if len(p) % 4 != 0:
        p = np.concatenate([p, np.zeros(4-len(p)%4)])
    for w, x, u, v in split_ns(p, 4):
        local1 = np.kron(Rx(w), Rx(x))
        local2 = np.kron(Rz(u), Rz(v)) 
        entangling = CNOT()
        U = entangling@local2@local1@U
    return U

def real_ansatz(p, n=2):
    U = np.eye(4)
    if len(p) % 2 != 0:
        p = np.concatenate([p, np.zeros(2-len(p)%2)])
    for w, x in split_ns(p, 2):
        local1 = np.kron(Ry(w), Ry(x))
        entangling = CZ()
        U = entangling@local1@U
    return U

def real_hermitian_ansatz(p, n=2):
    U = np.eye(4)
    for w in p:
        local1 = np.kron(Ry(w), I)
        entangling = swap()@CRy(np.pi-w)@swap()@CRy(-w)
        U = entangling@local1@U
    return U

#p = np.array([1])
#A = np.real(real_hermitian_ansatz(p))
#print(A, unitary_to_tensor(A)[1], sep='\n')
#raise Exception
def mb(ops):
    return reduce(np.kron, ops)

def state(p):
    U = ansatz(p)
    V = get_env_exact(U)
    return mb([U, I, I])@mb([I, U, I])@mb([I, I, V])@mb([np.array([1, 0])]*4)

def Ha(λ):
    return -np.kron(Z, Z)+(λ/2)*(np.kron(I, X)+np.kron(X, I))

def ϵ(p, λ=1):
    return np.real(state(p).conj().T@mb([I, Ha(λ), I])@state(p)) 

def plot_convergence(λ):
    energies  = []
    ps = [4, 8, 12, 16]
    last_energy = 0
    tries = 0
    max_tries = 3
    eps = 1e-4
    for n in ps:
        stop = False
        while not stop:
            try:
                p0 = np.random.randn(n)
                res = minimize(lambda p: ϵ(p, λ), p0, method='BFGS', tol=1e-10)
                print(n, res.fun, D2_gse(λ), e(λ))
                if res.fun < last_energy+eps or tries > max_tries:
                    last_energy = res.fun
                    energies.append(res.fun)
                    stop = True
                    tries = 0
                else:
                    print('trying again')
                    tries = tries+1
                    stop = False
            except np.linalg.LinAlgError:
                print('fail')

    plt.scatter(ps, np.array(energies)-e(λ), marker='+', label='quantum variational')
    plt.axhline(D2_gse(λ)-e(λ), linestyle='--', c='black', label='D=2 classical')
    plt.xticks(ps)
    plt.ylabel('$\epsilon-\epsilon_{\mathrm{exact}}$')
    plt.xlabel('number of ansatz parameters')
    plt.legend()
    plt.yscale('log')
    plt.show()

def plot_phase_diagram(n_λs=21):
    λs = np.linspace(0, 2, n_λs)
    ps = [4, 8, 12]
    energiess  = []
    last_energy = 0
    tries = 0
    max_tries = 3
    eps = 1e-3
    for λ in λs:
        energies = []
        for n in ps:
            stop = False
            while not stop:
                attempts = []
                try:
                    p0 = np.random.randn(n)
                    res = minimize(lambda p: ϵ(p, λ), p0, method='BFGS', tol=1e-10)
                    print(n, res.fun, D2_gse(λ), e(λ))
                    attempts.append(res.fun)
                    if res.fun < last_energy+eps or tries > max_tries:
                        last_energy = min(attempts)
                        energies.append(last_energy)
                        stop = True
                        tries = 0
                    else:
                        print('trying again')
                        tries = tries+1
                        stop = False
                except np.linalg.LinAlgError:
                    print('fail')
        energiess.append(energies)
    energiess = np.array(energiess)
    print(energiess.shape)

    plt.plot(λs, energiess[:, 0]-np.array([e(λ) for λ in λs]), marker='+', label='p=1, 4 parameters')
    plt.plot(λs, energiess[:, 1]-np.array([e(λ) for λ in λs]), marker='1', label='p=2, 8 parameters')
    plt.plot(λs, energiess[:, 2]-np.array([e(λ) for λ in λs]), marker='x', label='p=3, 12 parameters')

    plt.plot(λs, [D2_gse(λ)-e(λ) for λ in λs], linestyle='--', c='black',  label='D=2 classical')
    #plt.plot(λs, [e(λ) for λ in λs], linestyle='--', c='black', label='analytical')
    plt.ylabel('$\epsilon-\epsilon_{\mathrm{exact}}$')
    plt.xlabel('$\lambda$')
    plt.legend()
    plt.yscale('log')
    fig = plt.gcf()
    fig.set_size_inches(4, 4.5)
    plt.savefig('/Users/fergusbarratt/Desktop/ConvergenceFullCNOTIII.pdf', bbox_inches='tight')
    plt.show()

from scipy.optimize import minimize

if __name__=='__main__':
    #plot_convergence(1)
    plot_phase_diagram()
