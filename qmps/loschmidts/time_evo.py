import numpy as np
import cycler
from xmps.iOptimize import find_ground_state
from scipy.integrate import quad
import cirq
from xmps.iMPS import iMPS, Map
from qmps.represent import FullStateTensor, Environment
from qmps.tools import get_env_exact, unitary_to_tensor, environment_from_unitary
from scipy.linalg import expm
from qmps.ground_state import Hamiltonian
import matplotlib.pyplot as plt
import matplotlib as mpl
from qmps.represent import ShallowFullStateTensor
from qmps.tools import tensor_to_unitary, environment_to_unitary
from qmps.rotosolve import gate as gate_
from scipy.linalg import null_space, norm
from tqdm import tqdm
from xmps.spin import paulis
from qmps.represent import ShallowFullStateTensor
from qmps.tools import tensor_to_unitary
from qmps.rotosolve import gate
from scipy.linalg import null_space
from scipy.optimize import minimize
from xmps.tensor import rotate_to_hermitian
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

def gate(v, symbol='U'):
    #return ShallowCNOTStateTensor(2, v)
    #return ShallowQAOAStateTensor(2, v)
    return ShallowFullStateTensor(2, v, symbol)
    #return FullStateTensor(U4(v))

def obj(p, A, WW):
    B = iMPS([unitary_to_tensor(cirq.unitary(gate(p)))]).left_canonicalise()[0]
    WW_ = np.eye(WW.shape[0])

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
    C = cirq.Circuit([cirq.H(qbs[3]), cirq.CNOT(*qbs[3:5]),
                               U(*qbs[2:4]),
                               U(*qbs[1:3]),
                               W(*qbs[2:4]),
                               L(*qbs[0:2]),
                               R(*qbs[4:]),
                               cirq.inverse(U_)(*qbs[1:3]),
                               cirq.inverse(U_)(*qbs[2:4]),
                               cirq.CNOT(*qbs[3:5]), cirq.H(qbs[3])])
    #qbs = cirq.LineQubit.range(4)
    #normC = cirq.Circuit.from_ops([cirq.H(qbs[1]),
    #                               cirq.CNOT(*qbs[1:3]),
    #                               L(*qbs[:2]), 
    #                               R(*qbs[-2:]),
    #                               cirq.CNOT(*qbs[1:3]),
    #                               cirq.H(qbs[1])
    #                               ])
    s = cirq.Simulator(dtype=np.complex128)
    ff = np.sqrt(2*np.abs(s.simulate(C).final_state[0]))#/np.abs(s.simulate(normC).final_state[0])), np.sqrt(np.abs(x[0]))
    #print(ff[0]-ff[1])
    #print(ff[0], ff[1])
    return -ff

g0, g1 = 1.5, 0.2
A, es = find_ground_state(Hamiltonian({'ZZ':-1, 'X':g0}).to_matrix(), 2, tol=1e-2, noisy=True)
print(es[-1])
lles = []
eevs = []
eers = []
ps = [15]
for N in tqdm(ps):

    T = np.linspace(0, 6, 300)
    dt = T[1]-T[0]
    res = minimize(obj, np.random.randn(N), 
                   (A[0], np.eye(4)), 
                   method='Nelder-Mead',
                   options={'disp':True})
    params = res.x
    
    WW = expm(-1j*Hamiltonian({'ZZ':-1, 'X':g1}).to_matrix()*2*dt)
    ps = [params]
    ops = paulis(0.5)
    evs = []
    les = []
    errs = [res.fun]

    for _ in tqdm(T):
        A_ = iMPS([unitary_to_tensor(cirq.unitary(gate(params)))]).left_canonicalise()
        evs.append(A_.Es(ops))
        les.append(A_.overlap(A))
        res = minimize(obj, params, (A_[0], WW), options={'disp':True})

        params = res.x
        errs.append(res.fun)
        ps.append(params)
    lles.append(les)
    eevs.append(evs)
    eers.append(errs)

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


ps = [15]
for q, i in enumerate(ps):
    j = int((np.max(list(ps))-i)/2)
    np.save('lles', -np.log(np.array(lles)).T[0][:, j])
