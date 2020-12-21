from tqdm import tqdm
import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import eig, expm, null_space
from scipy.optimize import minimize
from functools import reduce 
from xmps.spin import U4
from xmps.tensor import rotate_to_hermitian
from qmps.tools import environment_to_unitary
from qmps.ground_state import Hamiltonian
from itertools import chain
import pickle
import cirq 
from qmps.represent import Environment, FullStateTensor, Tensor
from xmps.iMPS import iMPS, Map


I = np.eye(2)
Z = np.array([
    [1,0],
    [0,-1]
])
X = np.array([
    [0,1],
    [1,0]
])


def merge(A, B):
    # -A- -B-  ->  -A-B-
    #  |   |        ||
    return np.tensordot(A, B, [2, 1]).transpose([0, 2, 1, 3]).reshape(A.shape[0]**2, 2, 2)


def tensor_prod(tensors):
    return reduce(lambda t1,t2: np.kron(t1,t2), tensors)


def unitaries_to_tensor(U2, U1):
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
    A1 = unitaries_to_tensor(U2,U1)
    A2 = unitaries_to_tensor(Ū2,Ū1)
    
    tm = Map(np.tensordot(Ut, merge(A1, A1), [1,0]), merge(A2,A2))
    ηr, r = tm.right_fixed_point()
    return ηr, r

def exact_left_env(U2,U1,Ū2,Ū1,Ut):
    A1 = unitaries_to_tensor(U2,U1)
    A2 = unitaries_to_tensor(Ū2,Ū1)
    tm = Map(np.tensordot(Ut, merge(A1, A1), [1,0]), merge(A2,A2))
    ηl, l = tm.left_fixed_point()
    return ηl, l


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

class KakFullStateTensor(cirq.Gate):
    def __init__(self, params, symbol = "U"):
        self.p = params
        self.symbol = symbol

    def num_qubits(self):
        return 2

    def _decompose_(self, qubits):
        p = self.p

        return[
            # general single qubit unitary on both input legs
            cirq.rz(p[0])(qubits[0]), cirq.ry(p[1])(qubits[0]), cirq.rz(p[2])(qubits[0]),
            cirq.rz(p[3])(qubits[1]), cirq.ry(p[4])(qubits[1]), cirq.rz(p[5])(qubits[1]),
            
            # interaction qubit, exp(i * xXX + yYY +zZZ)
            cirq.XXPowGate(exponent = p[6])(*qubits),
            cirq.YYPowGate(exponent = p[7])(*qubits),
            cirq.ZZPowGate(exponent = p[8])(*qubits),

            # general single qubit unitary on both output legs
            cirq.rz(p[9])(qubits[0]), cirq.ry(p[10])(qubits[0]), cirq.rz(p[11])(qubits[0]),
            cirq.rz(p[12])(qubits[1]), cirq.ry(p[13])(qubits[1]), cirq.rz(p[14])(qubits[1]),
        ]

    def _circuit_diagram_info_(self, args):
        return [self.symbol] * 2



def build_circuit_wrong_order(U1, U2, Ū1, Ū2, Ut):
    """
    build he quantum circuits who's expectation values defines the TDVP update
    cost function.

    Us come in as the uninverted cirq.Gate objects.
    """
    q = cirq.LineQubit.range(8)
    c = cirq.Circuit()

    # turn cirq gates into unitaries with cirq.unitary
    _, r = exact_right_env(*map(cirq.unitary, [U1, U2, Ū1, Ū2, Ut]))

    R = Environment(put_env_on_left_site(r), "R")
    L = Environment(put_env_on_right_site(r.conj().T), "L")

    c.append([
        cirq.H(q[2]), cirq.CNOT(q[2], q[1]),
        U2(*[q[5],q[6]]), U2(*[q[3],q[4]]),
        U1(*[q[4],q[5]]), U1(*[q[2],q[3]]),
        Ut(*[q[2],q[3],q[4],q[5]]),
        R(*[q[0],q[1]]),
        L(*[q[6],q[7]]),
        cirq.inverse(Ū1(*[q[4],q[5]])), cirq.inverse(Ū1(*[q[2],q[3]])), 
        cirq.inverse(Ū2(*[q[3],q[4]])), cirq.inverse(Ū2(*[q[5],q[6]])),
        cirq.CNOT(q[2],q[1]), 
        cirq.H(q[2])
    ])

    return c

def build_circuit(U1, U2, Ū1, Ū2, Ut):
    q = cirq.LineQubit.range(8)
    c = cirq.Circuit()

    # turn cirq gates into unitaries with cirq.unitary
    eta_r, r = exact_right_env(*map(cirq.unitary, [U1, U2, Ū1, Ū2, Ut]))
    #eta_l, l = exact_left_env( *map(cirq.unitary, [U1, U2, Ū1, Ū2, Ut]))

    R = Environment(put_env_on_left_site(r), "R")
    L = Environment(put_env_on_right_site(r.conj().T), "L")

    c.append([
        cirq.H(q[5]), cirq.CNOT(q[5], q[6]),
        U2(*q[1:3]), U2(*q[3:5]),
        U1(*q[2:4]), U1(*q[4:6]),
        Ut(*q[2:6]),
        R(*q[6:]),
        L(*q[0:2]),
        cirq.inverse(Ū1(*q[2:4])), cirq.inverse(Ū1(*q[4:6])), 
        cirq.inverse(Ū2(*q[1:3])), cirq.inverse(Ū2(*q[3:5])),
        cirq.CNOT(q[5],q[6]), 
        cirq.H(q[5])
    ])
    
    norm = np.trace(r.conj().T @ r)
    return c , norm


def time_evo_cost_func(p, U1, U2, Ut):
    Ū1 = ShallowFullStateTensor(2, p[:15])
    Ū2 = ShallowFullStateTensor(2, p[15:])
    
    circuit, norm = build_circuit(U1, U2, Ū1, Ū2, Ut)
    sim = cirq.Simulator()
    return -2*np.abs(sim.simulate(circuit).final_state_vector[0] / norm)

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
    DEPTH = 2
    g0, g1 = 0,2
    STEPS = 100
    DT = 0.1
    FILENAME = f"d{DEPTH}ReOrderedEnvSimResults.pkl"

    Ut = FullStateTensor(expm(-1j * Ham(-1,g1) * DT))

    params = []
    init_param = [0] * (15 * DEPTH) # ground_state
    for _ in tqdm(range(STEPS)):
        params.append(init_param)
        U1, U2 = [ShallowFullStateTensor(2, init_param[d*15:(d+1)*15]) for d in range(DEPTH)]

        next_step = minimize(
            time_evo_cost_func,
            x0 = init_param,
            method = "Nelder-Mead",
            args = (U1, U2, Ut),
            options = {"disp":True, "maxiter":10000},
            tol = 1e-3
        )

        init_param = next_step.x
        
    with open(FILENAME, "wb") as f:
        pickle.dump(params, f)
    


