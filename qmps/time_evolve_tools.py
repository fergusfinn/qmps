import numpy as np
import cirq

from xmps.iMPS import iMPS, Map
from xmps.spin import paulis
from xmps.tensor import rotate_to_hermitian

from qmps.represent import StateGate, FullStateTensor, Environment, ShallowFullStateTensor
from qmps.tools import tensor_to_unitary, environment_to_unitary, unitary_to_tensor, environment_from_unitary, get_env_exact
from qmps.rotosolve import gate

from scipy.linalg import null_space, norm,expm

import matplotlib.pyplot as plt
import matplotlib as mpl

from tqdm import tqdm_notebook

I, X, Y, Z = np.eye(2), *paulis(0.5)
def merge(A, B):
    # -A- -B-  ->  -A-B-
    #  |   |        ||
    return np.tensordot(A, B, [2, 1]).transpose([0, 2, 1, 3]).reshape(2*A.shape[0], 2, 2)

def Nsphere(v):
    # Spherical coordinates for the (len(v)-1)-sphere
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

def gate(v, symbol='U'):
    #return ShallowCNOTStateTensor(2, v[:-1])
    return ShallowFullStateTensor(2, v, symbol)
    #return FullStateTensor(U4(v))
    
def egate(v, symbol='R'):
    return StateGate(v, symbol)

def get_overlap_exact(p1, p2, gate=gate, testing=True):
    A  = iMPS([unitary_to_tensor(cirq.unitary(gate(p1)))]).left_canonicalise()[0]
    B  = iMPS([unitary_to_tensor(cirq.unitary(gate(p2)))]).left_canonicalise()[0]
    x, r = Map(A, B).right_fixed_point()
    if testing:
        return np.abs(x)**2, r
    else: 
        return np.abs(x)**2

from scipy.optimize import minimize

def get_overlap(p1, p2, gate=gate, egate=egate, initial=None):
    initial = np.random.randn(8) if initial is None else initial

    def obj(rs, initial_params = initial, WW=None):
        WW = np.eye(4) if WW is None else WW
        
        A = iMPS([unitary_to_tensor(cirq.unitary(gate(p1)))]).left_canonicalise()[0]
        B = iMPS([unitary_to_tensor(cirq.unitary(gate(p2)))]).left_canonicalise()[0]

        U = Environment(tensor_to_unitary(A), 'U')
        U_ = Environment(tensor_to_unitary(B), 'U\'')

        r = rotate_to_hermitian((rs[:4]+1j*rs[4:]).reshape(2, 2))
        r /= np.sqrt(np.trace(r.conj().T@r))
        l = r
        
        R = Environment(put_env_on_left_site(r), 'R')
        L = Environment(put_env_on_right_site(l.conj().T), 'L')
        
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
        return -np.abs(s.simulate(C).final_state[0])*2
    
    res = minimize(obj, initial, (None,), method='Nelder-Mead', options={'disp':True})
    return res.fun

if __name__=='__main__':
    p1, p2 = np.random.randn(15), np.random.randn(15)
    x, r = get_overlap_exact(p1, p2)
    print(x, get_overlap(p1, p2, initial=np.concatenate([r.reshape(-1).real, r.reshape(-1).imag])))
    #print(get_overlap_exact(p1, p2), get_overlap(p1, p2))
    raise Exception

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
