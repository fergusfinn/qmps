import cirq

from xmps.iMPS import iMPS, TransferMatrix
from xmps.spin import U4

from .tools import cT, direct_sum, unitary_extension, sampled_bloch_vector_of, Optimizer, cirq_qubits, log2, split_ns, split_2s, split_3s, from_real_vector, to_real_vector, environment_to_unitary, unitary_to_tensor
from typing import List, Callable, Dict

from numpy import concatenate, allclose, tensordot, swapaxes, log2, diag
from numpy.linalg import eig
from numpy.random import randn
import numpy as np

from scipy.linalg import null_space, norm, cholesky, polar
from scipy.optimize import minimize


def get_env(U, C0=randn(2, 2)+1j*randn(2, 2), sample=False, reps=100000):
    '''NOTE: just here till we can refactor optimize.py
       return v satisfying

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
        '''

    def f_obj(v, U=U):
        """f_obj: take an 8d real vector, use mat to turn it into a 4d complex vector, environment_to_unitary to 
           turn it into a unitary, then calculate the objective function.
        """
        r = full_tomography_env_objective_function(FullStateTensor(U), 
                FullEnvironment(environment_to_unitary(from_real_vector(v))))
        return r

    def s_obj(v, U=U):
        """s_obj: take an 8d real vector, use mat to turn it into a 4d complex vector, environment_to_unitary to 
           turn it into a unitary, then calculate the (sampled) objective function.
        """
        r = sampled_env_objective_function(FullStateTensor(U), 
                FullEnvironment(environment_to_unitary(from_real_vector(v))))
        return r

    obj = s_obj if sample else f_obj

    res = minimize(obj, to_real_vector(C0.reshape(-1)), method='Nelder-Mead')
    return environment_to_unitary(from_real_vector(res.x))


#######################
# Objective Functions #
#######################

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
    qbs = cirq.LineQubit.range(3)
    r = 0

    LHS, RHS = cirq.Circuit(), cirq.Circuit()
    LHS.append([State(U, V)(*qbs)])
    RHS.append([V(*qbs[:2])])

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
    qbs = cirq.LineQubit.range(3)
    r = 0

    LHS, RHS = cirq.Circuit(), cirq.Circuit()
    LHS.append([State(U, V)(*qbs)])
    RHS.append([V(*qbs[:2])])

    sim = cirq.Simulator()
    LHS = sim.simulate(LHS).bloch_vector_of(qbs[0])
    RHS = sim.simulate(RHS).bloch_vector_of(qbs[0])
    return norm(LHS-RHS)

def trace_distance_cost_function(params, U):
    '''
        Circuit 1:              Circuit 2:              Circuit 3:
        Trace distance objective function:
        |   |   |   |   |   |`  |   |   |   |   |   |   |   |   |   |   |
        |   |-V-|   |   |   |`  |   |-V-|   |   |-V-|   |   |   |   |   |
        |-U-|   |   |-V-|   |`  |-U-|   |   |-U-|   |   |   |-V-|   |-V-|
        @-----------X       |`  @-----------X           |   @-------X
        H                   |`  H                       |   H
                         break1 |                    break2 |
                               rho                         sigma
    '''

    environment = FullStateTensor(U4(params))
    state = State(U, environment, 1)
    
    state_qubits = state.num_qubits()
    env_qubits = environment.num_qubits()

    aux_qubits = int(env_qubits/2)
    control_qubits = list(range(aux_qubits))

    target_qubits1 = list(range(state_qubits, state_qubits+aux_qubits))
    target_qubits2 = list(range(state_qubits, state_qubits+aux_qubits))
    target_qubits3 = list(range(env_qubits, env_qubits+aux_qubits))
    
    total_qubits = (2 * state_qubits)
    qubits = cirq.LineQubit.range(total_qubits)
    
    cnots1 = [cirq.CNOT(qubits[i], qubits[j]) for i, j in zip(control_qubits, target_qubits1)]
    cnots2 = [cirq.CNOT(qubits[i], qubits[j]) for i, j in zip(control_qubits, target_qubits2)]
    cnots3= [cirq.CNOT(qubits[i], qubits[j]) for i, j in zip(control_qubits, target_qubits3)]

    hadamards = [cirq.H(qubits[i]) for i in control_qubits]

    circuit1 = cirq.Circuit.from_ops([state(*qubits[:state_qubits]),
                                      environment(*qubits[state_qubits: state_qubits+env_qubits])] +
                                      cnots1 + hadamards)

    circuit2 = cirq.Circuit.from_ops([state(*qubits[:state_qubits]),
                                      state(*qubits[state_qubits:total_qubits])] + cnots2 + hadamards)

    circuit3 = cirq.Circuit.from_ops([environment(*qubits[:env_qubits]),
                                      environment(*qubits[env_qubits: 2*env_qubits])] + cnots3 +
                                      hadamards)

    simulator = cirq.Simulator() 
    results1 = simulator.simulate(circuit1)
    results2 = simulator.simulate(circuit2)
    results3 = simulator.simulate(circuit3)

    circuit1qubits = [*qubits[:aux_qubits]] + [*qubits[state_qubits: state_qubits + aux_qubits]]
    circuit3qubits = [*qubits[:aux_qubits]] + [*qubits[env_qubits: env_qubits + aux_qubits]]

    r_s = 1 - 2*results1.density_matrix_of(circuit1qubits)[-1, -1]
    r_squared = 1 - 2*results2.density_matrix_of(circuit1qubits)[-1, -1]
    s_squared = 1 - 2*results3.density_matrix_of(circuit3qubits)[-1, -1]

    score = (r_squared + s_squared - 2 * r_s).real
    return np.abs(score)

class TraceDistanceOptimizer(Optimizer):
    
    def objective_function(self, params):
        return trace_distance_cost_function(params, self.u)

    
    
############################################
# Cirq Gate Classes  #
############################################ 

class Tensor(cirq.Gate):
    def __init__(self, unitary, symbol):
        self.U = unitary
        self.n_qubits = int(log2(unitary.shape[0]))
        self.symbol = symbol

    def _unitary_(self):
        return self.U

    def num_qubits(self):
        return self.n_qubits

    def _circuit_diagram_info_(self, args):
        return [self.symbol] * self.n_qubits

    def __pow__(self, power, modulo=None):
        if power == -1:
            return self.__class__(self.U.conj().T, symbol=self.symbol + '†')
        else:
            return self.__class__(np.linalg.multi_dot([self.U] * power))


class StateTensor(Tensor):
    pass


class Environment(Tensor):
    pass


class FullStateTensor(StateTensor):
    """StateTensor: represent state tensor as a unitary"""

    def __init__(self, unitary, symbol='U'):
        super().__init__(unitary, symbol)

    def raise_power(self, power):
        return PowerCircuit(state=self, power=power)


class FullEnvironment(Environment):
    """Environment: represents the environment tensor as a unitary"""

    def __init__(self, unitary, symbol='V'):
        super().__init__(unitary, symbol)


class PowerCircuit(cirq.Gate):
    def __init__(self, state:FullStateTensor, power):
        self.power = power
        self.state = state

    def _decompose_(self, qubits):
        n_u_qubits = self.state.num_qubits()
        return (FullStateTensor(self.state.U)(*qubits[i:n_u_qubits + i]) for i in reversed(range(self.power)))

    def num_qubits(self):
        return self.state.num_qubits() + (self.power - 1)

    def _set_power(self, power):
        self.power = power


class State(cirq.Gate):
    def __init__(self, u: cirq.Gate, v: cirq.Gate, n=1):
        self.u = u
        self.v = v
        self.n_phys_qubits = n
        self.bond_dim = int(2 ** (u.num_qubits() - 1))

    def _decompose_(self, qubits):
        v_qbs = self.v.num_qubits()
        u_qbs = self.u.num_qubits()
        n = self.n_phys_qubits
        return [self.v(*qubits[n:n+v_qbs])] + [self.u(*qubits[i:i+u_qbs]) for i in list(range(n))[::-1]]

    def num_qubits(self):
        return self.n_phys_qubits + self.v.num_qubits()


class ShallowQAOAStateTensor(cirq.Gate):
    """ShallowStateTensor: shallow state tensor based on the QAOA circuit"""

    def __init__(self, bond_dim, βγs):
        self.βγs = βγs
        self.p = len(βγs)
        self.n_qubits = int(log2(bond_dim)) + 1

    def num_qubits(self):
        return self.n_qubits

    def _decompose_(self, qubits):
        return [[cirq.X(qubit) ** β for qubit in qubits] + \
                [cirq.ZZ(qubits[i], qubits[i + 1]) ** γ for i in range(self.n_qubits - 1)]
                for β, γ in split_2s(self.βγs)]

    def _circuit_diagram_info_(self, args):
        return ['U'] * self.n_qubits


class ShallowCNOTStateTensor(cirq.Gate):
    def __init__(self, bond_dim, βγs):
        self.βγs = βγs
        self.p = len(βγs)
        self.n_qubits = int(log2(bond_dim)) + 1

    def num_qubits(self):
        return self.n_qubits

    def params_per_iter():
        return 2

    def _decompose_(self, qubits):
        return [[cirq.rz(β)(qubit) for qubit in qubits] + \
                [cirq.rx(γ)(qubit) for qubit in qubits] + \
                [cirq.H(qubits[0])]+\
             list(reversed([cirq.CNOT(qubits[i], qubits[i + 1]) for i in range(self.n_qubits - 1)])) 
                #+\
#                 [cirq.SWAP(qubits[i], qubits[i+1 if i!= self.n_qubits-1 else 0]) for i in list(range(self.n_qubits))]
                for β, γ in split_2s(self.βγs)]

    def _circuit_diagram_info_(self, args):
        return ['U'] * self.n_qubits

class ShallowCNOTStateTensor_nonuniform(cirq.Gate):
    def __init__(self, bond_dim, βγs):
        self.βγs = βγs
        self.p = len(βγs)
        self.n_qubits = int(log2(bond_dim)) + 1
        self.D = bond_dim

    def num_qubits(self):
        return self.n_qubits

    def params_per_iter(D):
        return int((log2(D)+1)*2)

    def _decompose_(self, qubits):
        return [[cirq.rz(params[i])(qubit) for i, qubit in enumerate(qubits)]+
                [cirq.rx(params[i+self.n_qubits])(qubit) for i, qubit in enumerate(qubits)]+
                list(reversed([cirq.CNOT(qubits[i], qubits[i + 1]) for i in range(self.n_qubits - 1)]))
                for params in split_ns(self.βγs, self.n_qubits*2)]

    def _circuit_diagram_info_(self, args):
        return ['U'] * self.n_qubits

class ShallowCNOTStateTensor3(cirq.Gate):
    def __init__(self, bond_dim, βγs):
        self.βγs = βγs
        self.p = len(βγs)
        self.n_qubits = int(log2(bond_dim)) + 1

    def num_qubits(self):
        return self.n_qubits

    def _decompose_(self, qubits):
        return [[cirq.rz(β)(qubit) for qubit in qubits] + \
                [cirq.rx(γ)(qubit) for qubit in qubits] + \
                [cirq.rz(ω)(qubit) for qubit in qubits] + \
                [cirq.H(qubits[0])]+\
             list(reversed([cirq.CNOT(qubits[i], qubits[i + 1]) for i in range(self.n_qubits - 1)])) 
                #+\
#                 [cirq.SWAP(qubits[i], qubits[i+1 if i!= self.n_qubits-1 else 0]) for i in list(range(self.n_qubits))]
                for β, γ, ω in split_3s(self.βγs)]

    def _circuit_diagram_info_(self, args):
        return ['U'] * self.n_qubits

class ExactAfter4(cirq.Gate):
    """ExactAfter4"""
    def __init__(self, bond_dim, βγs):
        self.βγs = βγs
        self.p = len(βγs)
        self.n_qubits = int(log2(bond_dim)) + 1
        self.params_per_iter = 6

    def params_per_iter():
        return 6

    def num_qubits(self):
        return self.n_qubits

    def _decompose_(self, qubits):
        return [[cirq.rz(a)(qubits[0]), cirq.rz(d)(qubits[1])] + \
                [cirq.rx(b)(qubits[0]), cirq.rx(e)(qubits[1])] + \
                [cirq.rz(c)(qubits[0]), cirq.rz(f)(qubits[1])] + \
             list(reversed([cirq.CNOT(qubits[i], qubits[i + 1]) for i in range(self.n_qubits - 1)])) 
                +\
                 [cirq.SWAP(qubits[i], qubits[i+1 if i!= self.n_qubits-1 else 0]) for i in list(range(self.n_qubits))]
                for a, b, c, d, e, f in split_ns(self.βγs, 6)]

    def _circuit_diagram_info_(self, args):
        return ['U'] * self.n_qubits

class ShallowFullStateTensor(cirq.Gate):
    def __init__(self, bond_dim, βγs, symbol='U'):
        self.βγs = βγs
        self.p = len(βγs)
        self.n_qubits = int(log2(bond_dim)) + 1
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

class StateGate(cirq.Gate):
    def __init__(self, βγs, symbol='U'):
        self.βγs = βγs
        self.p = len(βγs)
        self.n_qubits = 2
        self.symbol = symbol

    def num_qubits(self):
        return self.n_qubits

    def _decompose_(self, qubits):
        a, b, c, d, e, f = self.βγs[:6]
        return [cirq.rx(a)(qubits[0]), cirq.rx(b)(qubits[1]),
                cirq.rz(c)(qubits[0]), cirq.rz(d)(qubits[1]),
                (cirq.XX**e)(*qubits), (cirq.YY**f)(*qubits)]

    def _circuit_diagram_info_(self, args):
        return [self.symbol] * self.n_qubits

class ShallowEnvironment(cirq.Gate):
    """ShallowEnvironmentTensor: shallow environment tensor based on the QAOA circuit"""

    def __init__(self, bond_dim, βγs):
        self.βγs = βγs
        self.p = len(βγs)
        self.n_qubits = 2 * int(log2(bond_dim))

    def num_qubits(self):
        return self.n_qubits

    def _decompose_(self, qubits):
        return [[cirq.X(qubit) ** β for qubit in qubits] +
                [cirq.ZZ(qubits[i], qubits[i + 1]) ** γ for i in range(self.n_qubits - 1)]
                for β, γ in split_2s(self.βγs)]

    def _circuit_diagram_info_(self, args):
        return ['V'] * self.n_qubits

