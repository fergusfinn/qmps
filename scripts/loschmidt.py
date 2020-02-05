import numpy as np
import cirq
from tqdm import tqdm

from scipy.linalg import expm, null_space, norm
from scipy.optimize import minimize
from scipy.integrate import quad

from xmps.iMPS import iMPS, Map
from xmps.spin import paulis
from xmps.tensor import rotate_to_hermitian
from xmps.iOptimize import find_ground_state

from qmps.ground_state import Hamiltonian
from qmps.represent import ShallowFullStateTensor, FullStateTensor, Environment
from qmps.represent import ShallowCNOTStateTensor, ShallowQAOAStateTensor
from qmps.tools import get_env_exact, unitary_to_tensor, environment_from_unitary
from qmps.tools import tensor_to_unitary, environment_to_unitary

import matplotlib.pyplot as plt
import matplotlib as mpl

import cycler
mpl.style.use('pub_fast')

I, X, Y, Z = np.eye(2), *paulis(0.5)

def merge(A, B):
        # -A- -B-  ->  -A-B-
    #  |   |        ||
    return np.tensordot(A, B, [2, 1]).transpose([0, 2, 1, 3]).reshape(2*A.shape[0], 2, 2)

def put_env_on_left_site(q, ret_n=False):
    # Take a matrix q (2x2) and create U such that
    # (right 0-|---|--0
    #          | u |        =  q_{ij}
    # (left) i-|---|--j
    q = q.T
    a, b, c, d = q.reshape(-1)
    n = np.sqrt(np.abs(a)**2+ np.abs(c)**2+ np.abs(b)**2+ np.abs(d)**2)
    guess = np.array([[a, c.conj(), b, d.conj()], [c, -a.conj(), d, -b.conj()]])/n
    orth = null_space(guess).conj().T
    A = np.concatenate([guess, orth], axis=0)
    A = cirq.unitary(cirq.SWAP)@A
    if ret_n:
        return A, n
    else:
        return A

def get_env_off_left_site(A):
    z = np.array([1, 0])
    return np.tensordot(np.tensordot(A.reshape(2, 2, 2, 2), z, [3, 0]), z, [1, 0]).T

def put_env_on_right_site(q, ret_n=False):
    q = q
    a, b, c, d = q.reshape(-1)
    n = np.sqrt(np.abs(a)**2+ np.abs(c)**2+ np.abs(b)**2+ np.abs(d)**2)
    guess = np.array([[a, b, d.conj(), -c.conj()], [c, d, -b.conj(), a.conj()]])/n
    orth = null_space(guess).conj().T
    A = np.concatenate([guess, orth], axis=0)
    #A = cirq.unitary(cirq.SWAP)@A
    if ret_n:
        return A, n
    else:
        return A

def get_env_off_right_site(A):
    z = np.array([1, 0])
    return np.tensordot(np.tensordot(A.reshape(2, 2, 2, 2), z, [2, 0]), z, [0, 0])

def test():
    N = 50

    for _ in range(50):
        q = np.random.randn(2, 2)+1j*np.random.randn(2, 2)
        assert np.allclose(get_env_off_left_site(np.prod(put_env_on_left_site(q, ret_n=True))), q)
        assert np.allclose(get_env_off_right_site(np.prod(put_env_on_right_site(q, ret_n=True))), q)
        U = put_env_on_left_site(q)
        V = put_env_on_right_site(q)
        assert np.allclose(V.conj().T@V, np.eye(U.shape[0]))
        assert np.allclose(U.conj().T@U, np.eye(U.shape[0]))

    for _ in range(N):
        A = iMPS().random(2, 2).left_canonicalise()[0]
        B = iMPS().random(2, 2).left_canonicalise()[0]#np.tensordot(expm(-1j*Z*dt), A, [1, 0])

        U = Environment(tensor_to_unitary(A), 'U')
        U_ = Environment(tensor_to_unitary(B), 'U\'')

        x, r = Map(merge(A, A), merge(B, B)).right_fixed_point()
        x_, l = Map(merge(A, A), merge(B, B)).left_fixed_point()
        L = put_env_on_right_site(l)
        R = put_env_on_left_site(r)
        assert np.allclose(get_env_off_left_site(put_env_on_left_site(r)), r)
        assert np.allclose(get_env_off_right_site(put_env_on_right_site(l)), l)
        U = put_env_on_left_site(r)
        V = put_env_on_right_site(l)
        assert np.allclose(V.conj().T@V, np.eye(U.shape[0]))
        assert np.allclose(U.conj().T@U, np.eye(U.shape[0]))
    N = 10
    for _ in range(N):
        A = iMPS().random(2, 2).left_canonicalise()[0]
        B = iMPS().random(2, 2).left_canonicalise()[0]#np.tensordot(expm(-1j*Z*dt), A, [1, 0])

        E = Map(A, B)

        x, r = E.right_fixed_point()
        x_, l = E.left_fixed_point()


        U = Environment(tensor_to_unitary(A), 'U')
        U_ = Environment(tensor_to_unitary(B), 'U\'')

        R = Environment(put_env_on_left_site(r), 'R')
        L = Environment(put_env_on_right_site(l.conj().T), 'L')


        qbs = cirq.LineQubit.range(4)
        for g in zip([cirq.I, cirq.X, cirq.Y, cirq.Z], [I, X, Y, Z]):
            C = cirq.Circuit.from_ops([cirq.H(qbs[1]), cirq.CNOT(*qbs[1:3]),
                                       R(*qbs[2:]),
                                       g[0](qbs[1]),
                                       cirq.CNOT(*qbs[1:3]), cirq.H(qbs[1])])
            s = cirq.Simulator()
            assert np.allclose(2*s.simulate(C).final_state[0]-np.trace(g[1]@r), 0, 1e-6, 1e-6)
        # r is the matrix on the 1st qubit

        qbs = cirq.LineQubit.range(4)
        for g in zip([cirq.I, cirq.X, cirq.Y, cirq.Z], [I, X, Y, Z]):
            C = cirq.Circuit.from_ops([cirq.H(qbs[1]), cirq.CNOT(*qbs[1:3]),
                                       U(*qbs[0:2]),
                                       R(*qbs[2:]),
                                       g[0](qbs[0]),
                                       cirq.inverse(U_)(*qbs[0:2]),
                                       cirq.CNOT(*qbs[1:3]), cirq.H(qbs[1])])
            s = cirq.Simulator()
            assert np.allclose(2*s.simulate(C).final_state[0]-x*np.trace(g[1]@r), 0, 1e-6, 1e-6)

        qbs = cirq.LineQubit.range(5)
        for g in zip([cirq.I, cirq.X, cirq.Y, cirq.Z], [I, X, Y, Z]):
            C = cirq.Circuit.from_ops([cirq.H(qbs[2]), cirq.CNOT(*qbs[2:4]),
                                       U(*qbs[1:3]),
                                       U(*qbs[0:2]),
                                       R(*qbs[3:]),
                                       g[0](qbs[0]),
                                       cirq.inverse(U_)(*qbs[0:2]),
                                       cirq.inverse(U_)(*qbs[1:3]),
                                       cirq.CNOT(*qbs[2:4]), cirq.H(qbs[2])])
            s = cirq.Simulator()
            #print(C.to_text_diagram(transpose=True))
            #raise Exception
            assert np.allclose(2*s.simulate(C).final_state[0]-x**2*np.trace(g[1]@r), 0, 1e-6, 1e-6)


        qbs = cirq.LineQubit.range(3)
        for g in zip([cirq.I, cirq.X, cirq.Y, cirq.Z], [I, X, Y, Z]):
            C = cirq.Circuit.from_ops([cirq.H(qbs[1]), cirq.CNOT(*qbs[1:3]),
                                       L(*qbs[:2]),
                                       g[0](qbs[2]),
                                       cirq.CNOT(*qbs[1:3]), cirq.H(qbs[1])])
            s = cirq.Simulator()

            assert np.allclose(2*s.simulate(C).final_state[0]-np.trace(g[1]@l.conj()), 0, 1e-6, 1e-6)
        # r is the matrix on the 1st qubit

        qbs = cirq.LineQubit.range(4)
        for g in zip([cirq.I, cirq.X, cirq.Y, cirq.Z], [I, X, Y, Z]):
            C = cirq.Circuit.from_ops([cirq.H(qbs[2]), cirq.CNOT(*qbs[2:4]),
                                       U(*qbs[1:3]),
                                       L(*qbs[:2]),
                                       g[0](qbs[3]),
                                       cirq.inverse(U_)(*qbs[1:3]),
                                       cirq.CNOT(*qbs[2:4]), cirq.H(qbs[2])])
            s = cirq.Simulator()
            #print(C.to_text_diagram(transpose=True))
            #raise Exception
            assert np.allclose(2*s.simulate(C).final_state[0]-x*np.trace(g[1]@l.conj()), 0, 1e-6, 1e-6)

        qbs = cirq.LineQubit.range(5)
        for g in zip([cirq.I, cirq.X, cirq.Y, cirq.Z], [I, X, Y, Z]):
            C = cirq.Circuit.from_ops([cirq.H(qbs[3]), cirq.CNOT(*qbs[3:5]),
                                       U(*qbs[2:4]),
                                       U(*qbs[1:3]),
                                       L(*qbs[0:2]),
                                       g[0](qbs[4]),
                                       cirq.inverse(U_)(*qbs[1:3]),
                                       cirq.inverse(U_)(*qbs[2:4]),
                                       cirq.CNOT(*qbs[3:5]), cirq.H(qbs[3])])
            s = cirq.Simulator()
            assert np.allclose(2*s.simulate(C).final_state[0]-x**2*np.trace(g[1]@l.conj()), 0, 1e-6, 1e-6)

        qbs = cirq.LineQubit.range(6)
        C = cirq.Circuit.from_ops([cirq.H(qbs[3]), cirq.CNOT(*qbs[3:5]),
                                   U(*qbs[2:4]),
                                   U(*qbs[1:3]),
                                   L(*qbs[0:2]),
                                   R(*qbs[4:]),
                                   cirq.inverse(U_)(*qbs[1:3]),
                                   cirq.inverse(U_)(*qbs[2:4]),
                                   cirq.CNOT(*qbs[3:5]), cirq.H(qbs[3])])
        s = cirq.Simulator()
        assert np.allclose(2*s.simulate(C).final_state[0], x**2*np.trace(l.conj().T@r))

def gate(v, symbol='U'):
    """gate to use for U
    """
    return ShallowCNOTStateTensor(2, v)

def obj(p, A, WW):
    B = iMPS([unitary_to_tensor(cirq.unitary(gate(p)))]).left_canonicalise()[0]

    E = Map(np.tensordot(WW, merge(A, A), [1, 0]), merge(B, B))

    x, r = E.right_fixed_point()
    x_, l = E.left_fixed_point()
    l = r

    U = Environment(tensor_to_unitary(A), 'U')
    U_ = Environment(tensor_to_unitary(B), 'U\'')

    R = Environment(put_env_on_left_site(r), 'θR')
    left = put_env_on_right_site(l.conj().T)
    L = Environment(left, 'θL')

    W = Environment(WW, 'W')

    qbs = cirq.LineQubit.range(6)
    C = cirq.Circuit.from_ops([cirq.H(qbs[3]), cirq.CNOT(*qbs[3:5]),
                               U(*qbs[2:4]),
                               U(*qbs[1:3]),
                               W(*qbs[2:4]),
                               L(*qbs[0:2]),
                               R(*qbs[4:]),
                               cirq.inverse(U_)(*qbs[1:3]),
                               cirq.inverse(U_)(*qbs[2:4]),
                               cirq.CNOT(*qbs[3:5]), cirq.H(qbs[3])])
    s = cirq.Simulator(dtype=np.complex128)
    ff = np.sqrt(2*np.abs(s.simulate(C).final_state[0]))
    return -ff

def noisy_obj(p, A, WW, prob=0):
    B = iMPS([unitary_to_tensor(cirq.unitary(gate(p)))]).left_canonicalise()[0]

    E = Map(np.tensordot(WW, merge(A, A), [1, 0]), merge(B, B))

    x, r = E.right_fixed_point()
    x_, l = E.left_fixed_point()
    l = r

    U = Environment(tensor_to_unitary(A), 'U')
    U_ = Environment(tensor_to_unitary(B), 'U\'')

    R = Environment(put_env_on_left_site(r), 'θR')
    left = put_env_on_right_site(l.conj().T)
    L = Environment(left, 'θL')

    W = Environment(WW, 'W')

    qbs = cirq.LineQubit.range(6)
    C = cirq.Circuit.from_ops([cirq.H(qbs[3]), cirq.CNOT(*qbs[3:5]),
                               U(*qbs[2:4]),
                               U(*qbs[1:3]),
                               W(*qbs[2:4]),
                               L(*qbs[0:2]),
                               R(*qbs[4:]),
                               cirq.inverse(U_)(*qbs[1:3]),
                               cirq.inverse(U_)(*qbs[2:4]),
                               cirq.CNOT(*qbs[3:5]), cirq.H(qbs[3])])

    noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(prob))
    system_qubits = sorted(C.all_qubits())
    noisy_circuit = cirq.Circuit()
    for moment in C:
        noisy_circuit.append(noise.noisy_moment(moment, system_qubits))
    s = cirq.Simulator(dtype=np.complex128)
    ff = np.sqrt(2*np.abs(s.simulate(noisy_circuit).final_state[0]))
    return -ff

def noisy_sampled_obj(p, A, WW, prob = 1e-4, repetitions=5000):
    B = iMPS([unitary_to_tensor(cirq.unitary(gate(p)))]).left_canonicalise()[0]

    E = Map(np.tensordot(WW, merge(A, A), [1, 0]), merge(B, B))

    x, r = E.right_fixed_point()
    x_, l = E.left_fixed_point()
    l = r

    U = Environment(tensor_to_unitary(A), 'U')
    U_ = Environment(tensor_to_unitary(B), 'U\'')

    R = Environment(put_env_on_left_site(r), 'θR')
    left = put_env_on_right_site(l.conj().T)
    L = Environment(left, 'θL')

    W = Environment(WW, 'W')

    qbs = cirq.LineQubit.range(6)
    C = cirq.Circuit.from_ops([cirq.H(qbs[3]), cirq.CNOT(*qbs[3:5]),
                               U(*qbs[2:4]),
                               U(*qbs[1:3]),
                               W(*qbs[2:4]),
                               L(*qbs[0:2]),
                               R(*qbs[4:]),
                               cirq.inverse(U_)(*qbs[1:3]),
                               cirq.inverse(U_)(*qbs[2:4]),
                               cirq.CNOT(*qbs[3:5]), cirq.H(qbs[3]),
                               cirq.measure(*qbs, key='result')])

    noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(prob))
    noisy_circuit = cirq.Circuit()
    for moment in C:
        noisy_circuit.append(noise.noisy_moment(moment, qbs))

    s = cirq.Simulator(dtype=np.complex128)
    output = s.run(noisy_circuit, repetitions=repetitions).measurements['result']
    p0 = sum([int(not any(x)) for x in output])/len(output)
    ff = np.sqrt(2*np.sqrt(p0))
    return -ff

def f(z, g0, g1):
    def theta(k, g):
        return np.arctan2(np.sin(k), g-np.cos(k))/2
    def phi(k, g0, g1):
        return theta(k, g0)-theta(k, g1)
    def epsilon(k, g1):
        return -2*np.sqrt((g1-np.cos(k))**2+np.sin(k)**2)
    def integrand(k):
        return -1/(2*np.pi)*np.log(np.cos(phi(k, g0, g1))**2 + np.sin(phi(k, g0, g1))**2 * np.exp(-2*z*epsilon(k, g1)))

    return quad(integrand, 0, np.pi)[0]

def loschmidt(t, g0, g1):
    return (f(t*1j, g0, g1)+f(-1j*t, g0, g1))

if __name__=='__main__':
    g0, g1 = 1.5, 0.2
    noises = [0, 1e-4, 1e-3, 1e-2]
    ps = [8]
    T = np.linspace(0, 3, 150)
    dt = T[1]-T[0]
    WW = expm(-1j*Hamiltonian({'ZZ':-1, 'X':g1}).to_matrix()*2*dt)
    ops = paulis(0.5)

    Q = [loschmidt(t, g0, g1) for t in T]

    A, es = find_ground_state(Hamiltonian({'ZZ':-1, 'X':g0}).to_matrix(), 2, tol=1e-2, noisy=False)

    lesss = [] # loschmidt echoes, all noise, all ps
    evsss = [] # expectation values, all noise, all ps
    errsss = [] # errors, all noise all ps
    for noise in tqdm(noises):
        less = [] # loschmidt echoes, fixed noise, all ps
        evss = [] # expectation values, fixed noise, all ps
        errss = [] # errors, fixed noise all ps
        for N in tqdm(ps):
            res = minimize(obj, np.random.randn(N),
                           (A[0], np.eye(4)),
                           method='Nelder-Mead',
                           options={'disp':False}) # get the initial state
            params = res.x

            paramss = [params]
            evs = []
            les = []
            errs = [res.fun]

            for _ in tqdm(T):
                A_ = iMPS([unitary_to_tensor(cirq.unitary(gate(params)))]).left_canonicalise()
                evs.append(A_.Es(ops))
                les.append(A_.overlap(A))
                res = minimize(noisy_obj, params, (A_[0], WW, noise), options={'disp':False})

                params = res.x
                errs.append(res.fun)
                paramss.append(params)

            less.append(les)
            evss.append(evs)
            errss.append(errs)
        lesss.append(less)
        evsss.append(evss)
        errsss.append(errss)
    print(np.array(lesss).shape)
    n = 4
    color = (plt.cm.viridis(np.linspace(0, 1, n)))
    mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    #mpl.style.use("pub_fast")
    #markers = [':', '--', '-.', ':']*2
    markers = ['-']*8
    fig, ax = plt.subplots(1, 1, sharex=True)
    #ax[0].plot(T, evss)

    for j, noise in enumerate(noises):
        for i, p in enumerate(ps):
            ax.plot(T, -np.log(np.array(lesss))[j, i, :, 0], label='depth = {}, $\eta={}$'.format(p, noise), linestyle = markers[i])
            ax.set_ylabel('Loschmidt Echo')

    #ax.plot(T, -np.log(np.array(less)).T[0][:, j], label='depth = {}'.format(i), linestyle = markers[j])
    ax.plot(T, Q, c='black', label='exact', linestyle='--')

    ax.legend(loc=1)

    plt.ylim([0, 1])
    #ax[0].set_ylabel('Expectation Values')
    ax.set_xlabel('time (t/J)')
    plt.savefig('loschmidts.pdf', bbox_inches='tight')
    plt.show()
