from xmps.iMPS import iMPS, Map
import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import eig, expm, null_space
from scipy.optimize import minimize
from functools import reduce 
from xmps.spin import U4
from xmps.tensor import rotate_to_hermitian
from qmps.tools import environment_to_unitary
from qmps.ground_state import Hamiltonian
from tqdm import tqdm 

import pickle
import matplotlib.pyplot as plt

# tensor:
"""
                   0      0      j
                   | -U2- |      |
A[i, σ1,σ2, j] =   |      | -U1- |
                   |      |      |
                   i      σ1      σ2
                   
"""

# alternate_tensor:
"""
                   i      0      0
                   |      | -U2- |
A[i, σ1,σ2, j] =   | -U1- |      |
                   |      |      |
                   σ1     σ2     j
"""

Z = np.array([
    [1,0],
    [0,-1]
])

X = np.array([
    [0,1],
    [1,0]
])

I = np.eye(2)

def tensor_prod(tensors):
    return reduce(lambda t1,t2: np.kron(t1,t2), tensors)

# Find the exact right environment:
def merge(A, B):
    # -A- -B-  ->  -A-B-
    #  |   |        ||
    return np.tensordot(A, B, [2, 1]).transpose([0, 2, 1, 3]).reshape(A.shape[0]**2, 2, 2)


def tensor(U2, U1):
    """
    Produce a tensor A[(σ1σ2),i,j]:
    
    A[σ1σ2,i,j] = Σα U1[σ1,σ2,α,j]U2[i,α,0,0]
    """
    
    return np.transpose(np.tensordot(
        U1.reshape(2,2,2,2),
        U2[:,0].reshape(2,2),
        (2,1)
    ),(0,1,3,2)).reshape(4,2,2)


def exact_right_env(U2,U1,Ū2,Ū1, Ut):
    A1 = tensor(U2,U1)
    A2 = tensor(Ū2,Ū1)
    
    tm = Map(np.tensordot(Ut, merge(A1, A1), [1,0]), merge(A2,A2))
    ηr, r = tm.right_fixed_point()
    return ηr, r

# Compare to the right environment that you get from trivial with adjustment:

def alternate_tensor(U2, U1):
    """
    Produce a tensor A[(σ1σ2),i,j]:
    
    A[(σ1σ2),i,j] = Σα U1[σ1,σ2,i,α]U2[α,j,0,0]
    """
    
    return np.tensordot(
        U1.reshape(2,2,2,2),
        U2[:,0].reshape(2,2),
        (3,0)).reshape(4,2,2)

def adjust_trivial_environment(U2,Ū2,mr):
    """
    Adjust the trivial environment with a matrix, mr:
    
    0      0
    | -U2- |
    |(3)   |
    i      |(2)
           mr
    j(0)   |(1)
    |      |
    | -Ū2- |
    0      0
    
    It is not clear why we have to transpose the mr matrix.
    """
    
    Rᵢⱼ = np.einsum(
        Ū2.conj().T[0,:].reshape(2,2), [0,1],  
        mr.T,                          [1,2],
        U2[:,0].reshape(2,2),          [3,2],
        [3,0])
    
    
    return rotate_to_hermitian(Rᵢⱼ) /(np.sign(Rᵢⱼ[0,0]) * np.linalg.norm(Rᵢⱼ))

def right_env(U2,U1,Ū2,Ū1):
    A1 = alternate_tensor(U2,U1)
    A2 = alternate_tensor(Ū2,Ū1)
    
    alt_tm = Map(A1,A2)
    
    η, mr = alt_tm.right_fixed_point()
    
    R = adjust_trivial_environment(U2,Ū2,mr)
    return R

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
    
    if ret_n:
        return A, n
    else:
        return A

    # Cirq Code:

import cirq 
from qmps.represent import ShallowFullStateTensor, Environment, FullStateTensor

class ShallowFullStateTensor(cirq.Gate):
    def __init__(self, bond_dim, βγs, symbol='U'):
        self.βγs = βγs
        self.p = len(βγs)
        self.n_qubits = int(np.log2(bond_dim)) + 1
        self.symbol = symbol

    def num_qubits(self):
        return self.n_qubits

    def _decompose_(self, qubits):
        return [cirq.rz(self.βγs[0])(qubits[0]), cirq.rx(self.βγs[1])(qubits[0]), cirq.rz(self.βγs[2])(qubits[0]),
                cirq.rz(self.βγs[3])(qubits[1]), cirq.rx(self.βγs[4])(qubits[1]), cirq.rz(self.βγs[5])(qubits[1]),
                cirq.CNOT(qubits[0], qubits[1]),
                cirq.ry(self.βγs[6])(qubits[0]),
                cirq.CNOT(qubits[1], qubits[0]),
                cirq.ry(self.βγs[7])(qubits[0]), cirq.rz(self.βγs[8])(qubits[1]),
                cirq.CNOT(qubits[0], qubits[1]),
                cirq.rz(self.βγs[9])(qubits[0]), cirq.rx(self.βγs[10])(qubits[0]), cirq.rz(self.βγs[11])(qubits[0]),
                cirq.rz(self.βγs[12])(qubits[1]), cirq.rx(self.βγs[13])(qubits[1]), cirq.rz(self.βγs[14])(qubits[1])]

    def _circuit_diagram_info_(self, args):
        return [self.symbol] * self.n_qubits


class Depth2State(cirq.Gate):
    def __init__(self, Us, n = 1):
        assert len(Us) == 2
        self.Us = Us
        self.n_phys_qubits = 2*n
        
    def _decompose_(self, qubits):
        return [self.Us[0](*qubits[n:n+2]) for n in range(0,self.n_phys_qubits,2)] + [self.Us[1](*qubits[n:n+2]) for n in range(1,self.n_phys_qubits+1, 2)]
    
    def num_qubits(self):
        return self.n_phys_qubits + 1
    
    def _circuit_diagram_info_(self, args):
        return ["U1,2"] * self.num_qubits()
    
def build_ciruit(U1, U2, Ū1, Ū2, Ut):
    _,r = exact_right_env(*map(cirq.unitary, [U2,U1,Ū2,Ū1,Ut]))
    R = Environment(put_env_on_left_site(r),'R')
    L = Environment(put_env_on_right_site(r.conj().T),'L')

    State = Depth2State([U2,U1], n = 2)
    S̄ = Depth2State([Ū2,Ū1], n = 2)

    qubits = cirq.LineQubit.range(8)
    circuit = cirq.Circuit()
    circuit.append([
        cirq.H.on(qubits[5]),
        cirq.CNOT.on(qubits[5],qubits[6]),
        State.on(*qubits[1:6]),
        L.on(*qubits[0:2]),
        Ut.on(*qubits[2:6]),
        R.on(*qubits[6:8]),

        cirq.inverse(S̄.on(*qubits[1:6])),
        cirq.CNOT.on(qubits[5],qubits[6]),
        cirq.H.on(qubits[5])
    ])
    
    return circuit

def overlap(U1, U2, Ū1, Ū2, Ut):
    circuit = build_ciruit(U1, U2,Ū1, Ū2, Ut)
    sim = cirq.Simulator()
    return 2*np.abs(sim.simulate(circuit).final_state_vector[0])

def circuit_state(U1, U2, Ū1, Ū2, Ut):
    circuit = build_ciruit(U1, U2, Ū1, Ū2, Ut)
    sim = cirq.Simulator()
    return sim.simulate(circuit).final_state_vector

def State3Vector(U1, U2):
    S = Depth2State([U2,U1], n=2)
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(6)
    circuit.append([
        U2.on(*qubits[4:6]),
        cirq.decompose_once(S.on(*qubits[0:5]))
    ])
    sim = cirq.Simulator()
    return sim.simulate(circuit).final_state_vector

def State2Vector(U1, U2):
    S = Depth2State([U2,U1], n=1)
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(4)
    circuit.append([
        U2.on(*qubits[2:4]),
        cirq.decompose_once(S.on(*qubits[0:3]))
    ])
    sim = cirq.Simulator()
    return sim.simulate(circuit).final_state_vector

def TimeEvoObj(p, U1, U2, Ut):
    Ū1 = ShallowFullStateTensor(2, p[:15], 'U1')
    Ū2 = ShallowFullStateTensor(2, p[15:], 'U2')
    return -overlap(U1, U2, Ū1, Ū2, Ut)

def EnergyObj(p, H):
    U1 = ShallowFullStateTensor(2, p[:15], 'U1')
    U2 = ShallowFullStateTensor(2, p[15:], 'U2')
    ψ = State3Vector(U1, U2)
    
    E = np.real(ψ.conj().T @ H @ ψ)
    return E

def Param2State(p):
    U1 = ShallowFullStateTensor(2, p[:15], 'U1')
    U2 = ShallowFullStateTensor(2, p[15:], 'U2')
    return State2Vector(U1, U2)

def Ham(J,g):
    return sum([
                (J/3)*sum([
                    tensor_prod([Z,Z,I,I]),
                    tensor_prod([I,Z,Z,I]),
                    tensor_prod([I,I,Z,Z])]),
                (g/4)*sum([
                    tensor_prod([X,I,I,I]),
                    tensor_prod([I,X,I,I]),
                    tensor_prod([I,I,X,I]),
                    tensor_prod([I,I,I,X])
                ])])


if __name__ == "__main__":
    # Loschimdt Echos:

    initial_state = np.random.rand(30)
    g0, g1 = 0,2
    STEPS = 500
    DT = 0.02


    Ut = FullStateTensor(expm(-1j * Ham(-1, g1) * DT))

    params = []
    init_param = [0]*30
    for _ in tqdm(range(STEPS)):
        params.append(init_param)
        U1 = ShallowFullStateTensor(2, init_param[:15], 'U1')
        U2 = ShallowFullStateTensor(2, init_param[15:], 'U2')

        next_step = minimize(
            TimeEvoObj,
            x0 = init_param,
            method = "Nelder-Mead",
            args = (U1, U2, Ut),
            options = {"disp":False}
        )
        
        init_param = next_step.x

    with open("bwLoschmidtEchosd2g00.pkl","wb") as f:
        pickle.dump(params, f)