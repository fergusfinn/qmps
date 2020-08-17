import numpy as np
from xmps.spin import paulis, CNOT, swap, H, CZ, CRy
from scipy.linalg import expm
from functools import reduce
from scipy import integrate
from scipy.linalg import norm
import matplotlib.pyplot as plt
from xmps.iMPS import Map, iMPS

def split_ns(x, n):
    """split_ns: take a list: [β, γ, β, γ, ...], return [[β, γ, β], [γ, β, γ], ...]
    """
    return [x[i:i+n] for i in range(len(x)) if not i%n]

from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.models.spins import SpinModel
from tenpy.algorithms import dmrg

from scipy.optimize import minimize

plt.style.use('pub_fast')
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
        entangling = swap()@CNOT()
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

def mb(ops):
    return reduce(np.kron, ops)

#def brick_wall_state(p, depth=2, support=2, n=2):
#    U = real_ansatz(p)
#    n_qubits = int((depth-1)*n+np.ceil(support/n)*n)
#    ψ = mb([np.array([1, 0])]*(n_qubits))
#    structure = '0'*n_qubits
#    for width in reversed(range(1, depth+1)):
#        string = 'I'*(depth-width)*int(n/2)+ 'U'*n*int(n_qubits/n-(depth-width))+ 'I'*(depth-width)*int(n/2)
#        structure+='\n'+string
#
#        ψ = ψ@mb([I]*(depth-width) +[U]*int(n_qubits/n-(depth-width)) + [I]*(depth-width))
#    return ψ, structure

class brick_wall_state(object):
    def __init__(self, p, depth=2, support=2, n=2):
        self.p = p
        self.depth = depth
        self.support = support
        self.n = n
        U = real_ansatz(p)
        self.n_qubits = n_qubits = int((depth-1)*n+np.ceil(support/n)*n)
        ψ = mb([np.array([1, 0])]*(n_qubits))
        structure = '0'*n_qubits
        for width in reversed(range(1, depth+1)):
            string = 'I'*(depth-width)*int(n/2)+ 'U'*n*int(n_qubits/n-(depth-width))+ 'I'*(depth-width)*int(n/2)
            structure+='\n'+string

            ψ = ψ@mb([I]*(depth-width) +[U]*int(n_qubits/n-(depth-width)) + [I]*(depth-width))

        self.ψ = ψ
        self.structure = structure
        self.ψ_struc = (ψ, structure)
    
    def grow(self, W):
        p, depth, support, n = self.p, self.depth+2, self.support, self.n
        U = real_ansatz(p)
        n_qubits = int((depth-1)*n+np.ceil(support/n)*n)
        ψ = mb([np.array([1, 0])]*(n_qubits))
        structure = '0'*n_qubits
        for width in reversed(range(1, depth+1)):
            if width > 2:
                string = 'I'*(depth-width)*int(n/2)+ 'U'*n*int(n_qubits/n-(depth-width))+ 'I'*(depth-width)*int(n/2)
                structure+='\n'+string

                ψ = ψ@mb([I]*(depth-width) +[U]*int(n_qubits/n-(depth-width)) + [I]*(depth-width))
            elif width <= 2:
                string = 'I'*(depth-width)*int(n/2)+ 'W'*n*int(n_qubits/n-(depth-width))+ 'I'*(depth-width)*int(n/2)
                structure+='\n'+string

                ψ = ψ@mb([I]*(depth-width) +[W]*int(n_qubits/n-(depth-width)) + [I]*(depth-width))

        self.ψ = ψ
        self.structure = structure
        self.ψ_struc = (ψ, structure)
        return self

    def ev(self, H):
        if H.shape[0] == 2:
            H = np.kron(H, I)
        H = mb([I]*int((self.n_qubits-2)/2) + [H] + [I]*int((self.n_qubits-2)/2))
        return np.real(self.ψ.conj().T@H@self.ψ)

    def evs(self, Hs):
        evs = []
        for h in Hs:
            evs.append(self.ev(h))
        return evs


def brick_wall_unitary(p, depth=2):
    U = real_ansatz(p)
    E = np.eye(2**(depth+1))
    structure = '0'*(depth+1)
    for i in range(depth):
        E = mb([I]*(depth-1-i)+[U]+[I]*i)@E
        structure+='\n'+'I'*(depth-1-i)+'UU'+'I'*i
    return E

def evolved_brick_wall_unitary(p, W, depth=2, sym = 'WW'):
    U = real_ansatz(p)
    E = np.eye(2**(depth+3))
    structure = '0'*(depth+3)
    for i in range(depth+2):
        if i < depth:
            E = mb([I]*(depth+1-i)+[U]+[I]*i)@E
            structure+='\n'+'I'*(depth+1-i)+'UU'+'I'*i
        elif i >= depth:
            E = mb([I]*(depth+1-i)+[W]+[I]*i)@E
            structure+='\n'+'I'*(depth+1-i)+sym+'I'*i
    return E

def brick_wall_unitary_to_tensor(U):
    n_qubits = int(np.log2(U.shape[0]))
    U = np.tensordot(U.reshape(*[2]*int(2*n_qubits-2), 4), np.array([1, 0, 0, 0]), [-1, 0])
    #print([n_qubits-2, n_qubits-1]+list(range(n_qubits-2))+list(range(n_qubits, 2*n_qubits-2)))
    U = U.transpose([n_qubits-2, n_qubits-1]+list(range(n_qubits-2))+list(range(n_qubits, 2*n_qubits-2)))
    return U.reshape(4, 2**(n_qubits-2), 2**(n_qubits-2))

def converge_finite_size_overlap(p1, p2, depth):
    max_width=11
    U = real_ansatz(p1)
    V = real_ansatz(p2)
    overlaps = []
    for width in range(4, max_width, 1):
        ψ = mb([np.array([1, 0])]*(width))
        φ = mb([np.array([1, 0])]*(width))
        for k in range(int(depth/2)):
            #print(k, width)
            if width%2 !=0:
                ψ = mb([U]*int(width/2)+[I])@ψ
                ψ = mb([I]+[U]*(int(width/2)))@ψ

                φ = mb([V]*int(width/2)+[I])@φ
                φ = mb([I]+[V]*(int(width/2)))@φ
            else:
                ψ = mb([U]*int(width/2))@ψ
                ψ = mb([I]+[U]*(int(width/2)-1)+[I])@ψ

                φ = mb([V]*int(width/2))@φ
                φ = mb([I]+[V]*(int(width/2)-1)+[I])@φ
        overlaps.append(np.abs(ψ.conj().T@φ))
    return overlaps

def Ha(λ):
    return -np.kron(Z, Z)+(λ/2)*(np.kron(I, X)+np.kron(X, I))

def ϵ(p, depth=2, λ=1, support=2, n=2):
    state1, structure1 = brick_wall_state(p, depth, support, n).ψ_struc
    structure1 += '\n'+ 'I'*(depth-1)+'HH'+'I'*(depth-1)
    ϵ_1 = np.real(state1.conj().T@mb([I]*(depth-1)+ [Ha(λ)]+ [I]*(depth-1))@state1) 

    state2, structure2 = brick_wall_state(p, depth, support+int(n/2), n).ψ_struc
    structure2 += '\n'+ 'I'*(depth)+'HH'+'I'*(depth)
    ϵ_2 = np.real(state2.conj().T@mb([I]*(depth)+ [Ha(λ)]+ [I]*(depth))@state2) 
    return (ϵ_1+ϵ_2)/2

def uniform_mps_overlap(p1, p2, depth):
    AB = brick_wall_unitary_to_tensor(brick_wall_unitary(p1, depth))
    AB_ = brick_wall_unitary_to_tensor(brick_wall_unitary(p2, depth))
    E = Map(AB, AB_)
    η, _ = E.left_fixed_point()
    return np.abs(η)

def uniform_local_overlap(p1, p2, depth):
    ψ = brick_wall_state(p1, depth, support=2).ψ
    ϕ = brick_wall_state(p2, depth, support=2).ψ
    return np.abs(ψ.conj().T@ϕ)

def evolved_mps_overlap(p1, p2, W, depth):
    U = evolved_brick_wall_unitary(p1, W, depth)
    V = evolved_brick_wall_unitary(p2, np.eye(4), depth, 'II')
    AB = brick_wall_unitary_to_tensor(U)
    AB_ = brick_wall_unitary_to_tensor(V)
    E = Map(AB, AB_)
    η, _ = E.left_fixed_point()
    return np.abs(η)

def evolved_local_overlap(p1, p2, W, depth):
    ψ = brick_wall_state(p1, depth, support=2).grow(W)
    ϕ = brick_wall_state(p2, depth, support=5)
    ϕ = ϕ.ψ
    ψ = ψ.ψ
    return np.abs(ψ.conj().T@ϕ)

def optimize_evolved_local_overlap(p, W, depth):
    print('optimizing ... ')
    res = minimize(lambda x: -evolved_local_overlap(p, x, W, depth), p+0.1*np.random.randn(2))
    print('optimized', '\n')
    return res.x, res.fun

def optimize_evolved_mps_overlap(p, W, depth):
    print('optimizing ... ')
    res = minimize(lambda x: -evolved_mps_overlap(p, x, W, depth), p+0.1*np.random.randn(2))
    print('optimized', '\n')
    return res.x, res.fun

##############################################################################
#                                  TESTS                                     #
##############################################################################

def test_mps():
    p = np.array([1, 1])
    AB = brick_wall_unitary_to_tensor(brick_wall_unitary(p))
    from xmps.iMPS import Map, iMPS
    E = Map(AB, AB)
    _, l = E.left_fixed_point()
    _, r = E.left_fixed_point()
    print(l, '\n', r)
    AB = iMPS([AB])
    BA = iMPS([BA])
    ϵ_mps = (AB.E(Ha(1))+BA.E(Ha(1)))/2
    print(ϵ_mps, ϵ(p))

def plot_ground_state_energies():
    np.set_printoptions(precision=3, suppress=True)
    λ = 1
    energies  = []
    p = 8 
    p0 = np.random.randn(p)
    res = minimize(lambda p: ϵ(p, 2), p0, method='BFGS')
    print(res.fun)
    energies.append(res.fun)

    print(energies)
    #plt.plot(ps, np.array(energies)-e(λ), marker='+', label='tree depth {}'.format(depth))
    #plt.axhline(D2_gse(λ)-e(λ), linestyle='--', c='black', label='D=2 classical')
    #plt.xticks(ps)
    plt.ylabel('$\epsilon-\epsilon_{\mathrm{exact}}$')
    plt.xlabel('number of ansatz parameters')
    plt.title('depth 2')
    plt.legend()
    #plt.yscale('log')

def scatter_mps_uniform_local_overlaps():
    depth, dt, N = 2, 0.1, 1000
    overlaps = []
    for _ in range(N):
        p1, p2 = np.array([1, 2]), np.array([1, 2])
        overlaps.append(uniform_local_overlap(p1, p2, depth)+1j*uniform_mps_overlap(p1, p2, depth))

    overlaps = np.array(overlaps)
    m, c = np.polyfit(np.squeeze(overlaps.real), np.squeeze(overlaps.imag), deg=1)
    plt.style.use('pub_slow')
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([1-1.2*dt, 1])
    ax.set_ylim([1-1.2*dt, 1])
    #ax.set_adjustable('datalim')
    #ax.set_aspect('equal')
    ax.scatter(overlaps.real, overlaps.imag, marker='.')
    #ax.plot(overlaps.real, m*overlaps.real+c, linestyle='--', c='black')
    ax.set_ylabel('true overlap')
    ax.set_xlabel('local overlap')
    plt.show()
    # plt.savefig('figures/local_vs_global_overlap.pdf', bbox_inches='tight')

def scatter_mps_evolved_local_overlaps():
    depth, dt, N = 2, 1e-4, 300
    overlaps = []
    for _ in range(N):
        #p1, p2 = np.array([1, 2]), np.array([1, 2])
        p1 = np.random.randn(2)
        p2 = p1#np.random.randn(2)
        W = expm(-1j*dt*Ha(1))
        try:
            overlap = evolved_local_overlap(p1, p2, W, depth)+1j*evolved_mps_overlap(p1, p2, W, depth)
        except:
            print('fail')
        print(overlap)
        overlaps.append(overlap)

    overlaps = np.array(overlaps)
    #m, c = np.polyfit(np.squeeze(overlaps.real), np.squeeze(overlaps.imag), deg=1)
    plt.style.use('pub_slow')
    fig, ax = plt.subplots(1, 1)
    #ax.set_adjustable('datalim')
    #ax.set_aspect('equal')
    ax.scatter(overlaps.real, overlaps.imag, marker='.')
    #ax.plot(overlaps.real, m*overlaps.real+c, linestyle='--', c='black')
    ax.set_ylabel('true overlap')
    ax.set_xlabel('local overlap')
    plt.show()
    # plt.savefig('figures/local_vs_global_overlap.pdf', bbox_inches='tight')

def optimized_mps_local_overlaps():
    depth, dt, N = 2, 5e-2, 100 
    overlaps = []
    for _ in range(N):
        p = np.random.randn(2)
        W = expm(-1j*dt*Ha(1))
        try:
            x, local_overlap = optimize_evolved_local_overlap(p, W, depth)
            mps_overlap = evolved_mps_overlap(p, x, W, depth)
            overlap = -local_overlap+1j*mps_overlap
        except:
            print('fail')
        print(overlap)
        overlaps.append(overlap)

    overlaps = np.array(overlaps)
    #m, c = np.polyfit(np.squeeze(overlaps.real), np.squeeze(overlaps.imag), deg=1)
    plt.style.use('pub_fast')
    fig, ax = plt.subplots(1, 1)
    plt.xlim([0.9, 1])
    plt.ylim([0.9, 1])
    #ax.set_adjustable('datalim')
    #ax.set_aspect('equal')
    ax.scatter(overlaps.real, overlaps.imag, marker='.')
    #ax.plot(overlaps.real, m*overlaps.real+c, linestyle='--', c='black')
    ax.set_ylabel('true overlap')
    ax.set_xlabel('local overlap')
    plt.show()

def trajectory_local_overlaps():
    np.set_printoptions(precision=3, suppress=True)
    depth, dt, N = 3, 1e-2, 5000 
    overlaps = []
    evs = []
    x1 = x2 = np.random.randn(2)
    for _ in range(N):
        W = expm(-1j*dt*Ha(1))
        try:
            x1, local_overlap = optimize_evolved_local_overlap(x1, W, depth)
            x2, mps_overlap = optimize_evolved_mps_overlap(x2, W, depth)
        except:
            print('fail')
        psi1 = brick_wall_state(x1)
        psi2 = brick_wall_state(x2)
        evs.append([psi1.ev(X), psi2.ev(X)])
        overlap = -local_overlap+1j*mps_overlap
        print(overlap)
        overlaps.append(overlap)

    evs = np.array(evs)
    plt.plot(evs[:, 0], label='local')
    plt.plot(evs[:, 1], label='mps')
    plt.legend()
    plt.show()

if __name__=='__main__':
    np.set_printoptions(precision=3, suppress=True)
    plot_ground_state_energies()
    #plot_ground_state_energies()

