from .tools import cT, direct_sum, unitary_extension

from numpy import concatenate, allclose, tensordot, swapaxes
from scipy.linalg import null_space, norm
import cirq

def sampled_tomography_env_objective_function(U, V, reps=10000):
    """sampled_environment_objective_function: return norm of difference of (sampled) bloch vectors
       of qubit 0 in 

        | | |   | | | 
        | ---   | | |       
        |  v    | | |  
        | ---   | | |  
        | | |   | | |           (2)
        --- |   --- |  
         u  |    v  |  
        --- |   --- |  
        | | | = | | |             
        ρ | |   σ | |  

    """
    U, V = U._unitary_(), V._unitary_()
    qbs = cirq.LineQubit.range(3)
    r = 0

    LHS, RHS = Circuit(), Circuit()
    LHS.append([State2(U, V)(*qbs)])
    RHS.append([Environment2(V)(*qbs[:2])])

    LHS = sampled_bloch_vector_of(qbs[0], LHS, reps)
    RHS = sampled_bloch_vector_of(qbs[0], RHS, reps)
    return norm(LHS-RHS)

def full_tomography_env_objective_function(U, V):
    """full_environment_objective_function: return norm of difference of bloch vectors
       of qubit 0 in 

        | | |   | | | 
        | ---   | | |       
        |  v    | | |  
        | ---   | | |  
        | | |   | | |           (2)
        --- |   --- |  
         u  |    v  |  
        --- |   --- |  
        | | | = | | |             
        j | |   j | |  

    """
    U, V = U._unitary_(), V._unitary_()
    qbs = cirq.LineQubit.range(3)
    r = 0

    LHS, RHS = Circuit(), Circuit()
    LHS.append([State2(U, V)(*qbs)])
    RHS.append([Environment2(V)(*qbs[:2])])

    sim = Simulator()
    LHS = sim.simulate(LHS).bloch_vector_of(qbs[0])
    RHS = sim.simulate(RHS).bloch_vector_of(qbs[0])
    return norm(LHS-RHS)

class State(cirq.Gate):
    def __init__(self, u: ShallowStateTensor, v: ShallowEnvironment, n: int):
        self.u = u
        self.v = v
        self.n_phys_qubits = n
        self.bond_dim = int(2 ** (u.num_qubits() - 1))

    def _decompose_(self, qubits):
        v_qbs = self.v.num_qubits()
        u_qbs = self.u.num_qubits()
        n = self.n_phys_qubits
        return [self.v(*qubits[n:n+v_qbs])] + [self.u(*qubits[i:i+u_qbs]) for i in range(n)]

    def num_qubits(self):
        return self.n_phys_qubits + self.v.num_qubits()
