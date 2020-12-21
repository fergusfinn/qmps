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

I = np.eye(2)
Z = np.array([
    [1,0],
    [0,-1]
])

X = np.array([
    [0,1],
    [1,0]
])

def tensor_prod(tensors):
    return reduce(lambda t1,t2: np.kron(t1,t2), tensors)

class ShallowFullStateTensor(cirq.Gate):
    def __init__(self, βγs, symbol='U'):
        self.βγs = βγs
        self.p = len(βγs)
        self.symbol = symbol

    def num_qubits(self):
        return 2

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
        return [self.symbol] * self.num_qubits()


class ShallowParam(cirq.Gate):
    def __init__(self, params):
        self.p = params
        self.d = len(self.p)/2
        assert (len(self.p) % 2) == 0

    def num_qubits(self):
        return 2

    def _decompose_(self, qubits):
        circuit = []
        for i in range(int(len(self.p)/2)):
            circuit.append(cirq.ry(self.p[2*i])(qubits[0]))
            circuit.append(cirq.ry(self.p[(2*i)+1])(qubits[1]))
            circuit.append(cirq.CNOT(qubits[0], qubits[1]))

        return circuit

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


def build_noenv_NCircuits(Us, Ūs, Ut, depth):
    # Us = [U1, U2, U3, ...]
    assert depth == len(Us)
    
    ŪsInv = list(map(cirq.inverse, Ūs))
    
    qubits = (depth + 1) * 2
    
    q = cirq.LineQubit.range(qubits)
    circuit = cirq.Circuit()
    
    # this is the number of unitary tensors that is required in the top layer.
    # This decreases by 1 in each iteration
    num_tensors = depth + 1
    
    # create a list that says which tensor to use and which qubtis to use in each iteration.
    # [UNumber, qubit1, qubit2]
    # [e.g. [0, 0,1], [0,2,3], ... ]
    order = []
    for t in range(depth):
        for qs in range(t, qubits - t,2):
            order.append([t, qs,qs+2])
    
    time_evo_qubits = list(range((depth-1),(depth+4)))
    
    # Append the Us, 
    for qs in order:
        circuit.append(list(reversed(Us))[qs[0]].on(*q[qs[1]:qs[2]]))
        
    circuit.append(Ut.on(*q[time_evo_qubits[0]:time_evo_qubits[-1]]))
    
    for qs in reversed(order):
        circuit.append(list(reversed(ŪsInv))[qs[0]].on(*q[qs[1]:qs[2]]))
        
    return circuit


def DepthNState(Us, depth = 2):
    # Us come in as [U1, U2, U3, ...]
    assert len(Us) == depth
    nq = (depth + 1) * 2 # number of qubits needed based on the depth
    
    q = cirq.LineQubit.range(nq)
    circuit = cirq.Circuit()
    
    # this is the number of unitary tensors that is required in the top layer.
    # This decreases by 1 in each iteration
    num_tensors = depth + 1
    
    # create a list that says which tensor to use and which qubtis to use in each iteration.
    # [UNumber, qubit1, qubit2]
    # [e.g. [0, 0,1], [0,2,3], ... ]
    order = []
    for t in range(depth):
        for qs in range(t, nq - t,2):
            order.append([t, qs,qs+2])
            
    for qs in order:
        circuit.append(list(reversed(Us))[qs[0]].on(*q[qs[1]:qs[2]]))
    
    sim = cirq.Simulator()
    return sim.simulate(circuit).final_state_vector

def overlap(Us, Ūs, Ut, env = True, time_evo_in = False, depth = 2):
    circuit = build_noenv_NCircuits(Us, Ūs, Ut, depth = depth)
    sim = cirq.Simulator()
    return 2*np.abs(sim.simulate(circuit).final_state_vector[0])


def TimeEvoObj(p, Us, Ut, env, time_evo_in, PARAM, depth = 2):
    ppg = PARAM["ppg"]
    Ūs = [PARAM["gate"](p[d*ppg:(d+1)*ppg]) for d in range(depth)]
    
    return -overlap(Us, Ūs, Ut, env, time_evo_in, depth)

def EnergyObj(p, H, depth):
    
    Us = [ShallowFullStateTensor(p[d*15:(d+1)*15]) for d in range(depth)]
    
    ψ = DepthNState(Us, depth)
    
    return np.real(ψ.conj().T @ H @ ψ)

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

def simulateNoEnv(depth, g0, g1, dt, steps, parametrisation, paramdepth = 1):
    DEPTH = depth
    STEPS = steps
    DT = dt
    PARAM = parametrisation
    FILENAME = f"d{DEPTH}NoEnvSimResults{PARAM['name']}.pkl"

    # PARAM is a dictionary with three keys:
    # PARAM = {"gate":Gate_Class,
    #          "ppg": Params_Per_Gate,
    #          "name": Gate_Name}
    
    Ut = FullStateTensor(expm(-1j * Ham(-1,g1) * DT))

    params = []
    
    # params per gate
    ppg = PARAM["ppg"]

    init_param = [0] * (ppg * DEPTH) # ground_state
    for _ in tqdm(range(STEPS)):
        params.append(init_param)
        Us = [PARAM["gate"](init_param[d*ppg:(d+1)*ppg]) for d in range(DEPTH)]

        next_step = minimize(
            TimeEvoObj,
            x0 = init_param,
            method = "Nelder-Mead",
            args = (Us, Ut, False, False, PARAM, DEPTH),
            tol = 1e-5
        )

        init_param = next_step.x
        
    with open(FILENAME, "wb") as f:
        pickle.dump(params, f)


if __name__ == "__main__":
    # DEPTH = 2
    # g0, g1 = 0,2
    # STEPS = 100
    # DT = 0.1
    # PARAM = {
    #     "gate":ShallowFullStateTensor,
    #     "ppg": 15,
    #     "name":"ShallowFullNoEnv"
    # }

    # simulateNoEnv(
    #     DEPTH,
    #     g0,g1,
    #     DT,
    #     STEPS,
    #     PARAM
    #     )
    from scipy.stats import unitary_group
    Us = [KakFullStateTensor(np.random.rand(15))] * 3    
    Ut = FullStateTensor(unitary_group.rvs(16))

    circuit = build_noenv_NCircuits(Us, Us, Ut,3)
    print(circuit)