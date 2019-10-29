from rotosolve import double_rotosolve
import numpy as np
import cirq
from xmps.iMPS import iMPS, Map
from qmps.represent import get_env_exact, FullStateTensor, Environment
from qmps.represent import StateGate
from qmps.tools import unitary_to_tensor, environment_from_unitary
from scipy.linalg import expm
from qmps.ground_state import Hamiltonian
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('pub_fast')

from qmps.represent import ShallowFullStateTensor
from qmps.tools import tensor_to_unitary
from qmps.rotosolve import gate as gate_
from scipy.linalg import null_space, norm
from tqdm import tqdm
from xmps.spin import paulis
from qmps.represent import ShallowFullStateTensor
from qmps.tools import tensor_to_unitary
from qmps.rotosolve import gate
from scipy.linalg import null_space
from scipy.optimize import minimize
I, X, Y, Z = np.eye(2), *paulis(0.5)
def merge(A, B):
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

def run_tests(N):
    """run_tests: just a whole bunch of tests.

    :param N: number of iterations to run
    """
    for _ in range(N):
        q = np.random.randn(2, 2)+1j*np.random.randn(2, 2)
        assert np.allclose(get_env_off_left_site(np.prod(put_env_on_left_site(q, ret_n=True))), q)
        assert np.allclose(get_env_off_right_site(np.prod(put_env_on_right_site(q, ret_n=True))), q)
        U = put_env_on_left_site(q)
        V = put_env_on_right_site(q)
        assert np.allclose(V.conj().T@V, np.eye(U.shape[0]))
        assert np.allclose(U.conj().T@U, np.eye(U.shape[0]))
        
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
    return ShallowFullStateTensor(2, v, symbol)

def state_gate(v, symbol='R'):
    return StateGate(v, symbol)

def obj(p_, A, WW):
    p, rs = p_[:15], p_[15:]
    
    B = iMPS([unitary_to_tensor(cirq.unitary(gate(p)))]).left_canonicalise()[0]
    U = Environment(tensor_to_unitary(A), 'U')
    U_ = Environment(tensor_to_unitary(B), 'U\'')
    
    R = state_gate(rs)
    L = Environment(put_env_on_right_site(environment_from_unitary(cirq.unitary(R)).conj().T), 'L')
    
    W = Environment(WW, 'W')
    
    qbs = cirq.LineQubit.range(5)
    C = cirq.Circuit.from_ops([R(*qbs[3:5]),
                               U(*qbs[2:4]),
                               U(*qbs[1:3]),
                               W(*qbs[2:4]),
                               L(*qbs[0:2]),
                               cirq.inverse(U_)(*qbs[1:3]),
                               cirq.inverse(U_)(*qbs[2:4]),
                               cirq.CNOT(*qbs[3:5]), cirq.H(qbs[3])])
    
    s = cirq.Simulator(dtype=np.complex128)
    return -np.sqrt(np.abs(np.sqrt(2)*s.simulate(C).final_state[0]))

def obj_state(p_, A, WW):
    p, rs = p_[:15], p_[15:]
    
    B = iMPS([unitary_to_tensor(cirq.unitary(gate(p)))]).left_canonicalise()[0]
    U = Environment(tensor_to_unitary(A), 'U')
    U_ = Environment(tensor_to_unitary(B), 'U\'')
    
    R = state_gate(rs)
    #R = Environment(environment_to_unitary(r), 'R')
    L = Environment(put_env_on_right_site(environment_from_unitary(cirq.unitary(R)).conj().T), 'L')
    
    W = Environment(WW, 'W')
    
    qbs = cirq.LineQubit.range(5)
    C = cirq.Circuit.from_ops([R(*qbs[3:5]),
                               U(*qbs[2:4]),
                               U(*qbs[1:3]),
                               W(*qbs[2:4]),
                               L(*qbs[0:2]),
                               cirq.inverse(U_)(*qbs[1:3]),
                               cirq.inverse(U_)(*qbs[2:4]),
                               cirq.CNOT(*qbs[3:5]), cirq.H(qbs[3])])
    
    s = cirq.Simulator(dtype=np.complex128)
    return s.simulate(C).final_state

def obj_H():
    return -np.diag(np.eye(2**5)[0])

if __name__=='__main__':
    A = iMPS().random(2, 2).left_canonicalise() # initial state should be an iMPS object

    T = np.linspace(0, 1, 10) # timesteps
    dt = T[1]-T[0]
    # parametrise the initial state by minimizing the overlap with parametrised unitary. 
    # This is janky and would be way better the other way around. 
    # unless res.fun == -1, the initial state of the time evo is not what you want it to be. 
    #res = minimize(obj, np.random.randn(15), 
    #               (A[0], np.eye(4)), 
    #               method='Nelder-Mead',
    #               options={'disp':True})

    #params = res.x
    params = np.random.randn(21)

    # Define the time evolution operator
    WW = expm(-1j*Hamiltonian({'ZZ':-1, 'X':1}).to_matrix()*dt)


    # What to get expectation values of
    ops = paulis(0.5)

    ps = [params]
    evs = []
    les = []
    for _ in tqdm(T[1:]):
        # current mps tensor
        A_ = iMPS([unitary_to_tensor(cirq.unitary(gate(params)))]).left_canonicalise()

        #es, params = minimize(obj, params, (A_[0], WW), options={'disp':True})
        es, params = double_rotosolve(obj_H(), obj_state, params, (A_[0], WW))
        plt.plot(es)
        plt.show()
        raise Exception

        # store params, expectation values and loschmidt echo
        evs.append(A_.Es(ops))
        les.append(A_.overlap(A))
        ps.append(params)

    np.save('params', np.array(ps))

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(T[1:], evs)
    ax[1].plot(T[1:], -np.log(np.array(les)))
    ax[0].set_title('Expectation values', loc='right')
    ax[1].set_title('Loschmidt Echo', loc='right')
    ax[1].set_xlabel('t')
    plt.show()
    
