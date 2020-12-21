
from xmps.iMPS import iMPS
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
from itertools import chain
import pickle
import matplotlib.pyplot as plt
import sys
import cirq 
from qmps.represent import Environment, FullStateTensor, Tensor



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
    return np.tensordot(A, B, [2, 1]).transpose([0, 2, 1, 3]).reshape(A.shape[0]**2, A.shape[1], A.shape[2])

 
def depthNtensor(N_Us):
    # depth N
    N = len(N_Us)
    N_Us = map(lambda a: a.reshape(2,2,2,2), N_Us)
    # number indices after contraction
    ni = 2+2*N
    
    # bond dimension
    d = 2**(N - 1)
    
    # corresponds to the iₐ indices - reversed list of odd indices between 3 and ni-3 inclusive 
    i_indices = [i for i in reversed(range(3,ni-2,2))]
    
    # corresponds to the jₐ indices - reversed list of even indices between 2 and ni-4 inclusive
    j_indices = [j for j in reversed(range(2,ni-3,2))]
    
    # add the physical indices, [0,1], to the auxillary indices, and then the indices set to 0s
    indices = [0,1] + i_indices + j_indices + [ni-2, ni-1]
    
    # Tensordot all the tensors together in the order:
    #    1: UN-1, UN = A1
    #    2: UN-2, A1 = A2
    #    3  UN-3, A2 = A3, ...
    # Then reshape into the index order specified by indices
    A = reduce(lambda a,b: np.tensordot(b, a, [2,1]), N_Us).transpose(*indices).reshape(4,d,d,4)[...,0]
    
    return A

# Compare to the right environment that you get from trivial with adjustment:

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
        
    
def LoschmidtOverlapDepthN(p0, params, DEPTH, PARAM):
    
    ppg = PARAM["ppg"]
    Us = [cirq.unitary(PARAM["gate"](p0[d*ppg:(d+1)*ppg])) for d in range(DEPTH)]
    Ūs = [cirq.unitary(PARAM["gate"](params[d*ppg:(d+1)*ppg])) for d in range(DEPTH)]
    
    try:
        
        A = (iMPS([depthNtensor(Us)]) + iMPS([0.0001 * np.random.rand(4,bd,bd)])).left_canonicalise()
        Ā = iMPS([depthNtensor(Ūs)]).left_canonicalise()
    
    except:
        bd = int(2**(DEPTH-1))
        A = (iMPS([depthNtensor(Us)]) + iMPS([1e-5 * np.random.rand(4,bd,bd)])).left_canonicalise()
        Ā = (iMPS([depthNtensor(Ūs)]) + iMPS([1e-5 * np.random.rand(4,bd,bd)])).left_canonicalise()
    
    return -np.log10(np.abs(A.overlap(Ā)))
    
    
def PlotLoschimdt(param_results, DEPTH, PARAM):
    
    p0 = param_results[0]
    
    results = []
    for p in param_results[0:]:
        results.append(LoschmidtOverlapDepthN(p0,p,DEPTH, PARAM))
        
    return results

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


if __name__ == "__main__":
    d2file = "./d2NoEnvSimResults.pkl"
    d2envfile = "./d2ReOrderedEnvSimResults.pkl"
    d2kakfile = "./d2NoEnvSimResultsKakNoEnv.pkl"

    d3file = "./d3NoEnvSimResults.pkl"
    d3kakfile = "./d3NoEnvSimResultsKakNoEnv.pkl"

    d4file = "./d4KakNoEnvSimResults.pkl"
    d4kakfile = "./d4NoEnvSimResultsKakNoEnv"

    shallowd2 = "./d2NoEnvSimResultsShallowParamDepth2.pkl"
    shallowd4 = "./d2NoEnvSimResultsShallowParamDepth4.pkl"
    filename = d2file #sys.argv[1]
    # outputname = sys.argv[2]
    depth = 2

    with open(filename, "rb") as f:
        ps = pickle.load(f)    

    print(len(ps[0]))
    PARAM = {
        "gate":ShallowFullStateTensor,
        "ppg":15
    }

    r = PlotLoschimdt(ps, depth, PARAM)
    
    plt.figure()
    plt.plot(r)
    plt.show()
    #plt.savefig(outputname)
