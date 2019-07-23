import cirq

from xmps.iMPS import iMPS, TransferMatrix

from .tools import cT, direct_sum, unitary_extension,sampled_bloch_vector_of, Optimizer, cirq_qubits, log2, split_2s 
from .tools import from_real_vector, to_real_vector, environment_to_unitary
from .tools import unitary_to_tensor

from typing import List, Callable, Dict

from numpy import concatenate, allclose, tensordot, swapaxes, log2
from numpy.linalg import eig
from numpy import diag
from numpy.random import randn

from scipy.linalg import null_space, norm, cholesky
from scipy.optimize import minimize
from scipy.linalg import polar

import numpy as np

def sqrtm(X):
    Λ, V = eig(X)
    return V@csqrt(diag(X))

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

def get_env_exact(U):
    η, l, r = TransferMatrix(unitary_to_tensor(U)).eigs()
    return environment_to_unitary(cholesky(r))


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

############################################
# Tensor, StateTensor, Environment, State  #
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
        return [self.v(*qubits[n:n+v_qbs])] + [self.u(*qubits[i:i+u_qbs]) for i in range(n)]

    def num_qubits(self):
        return self.n_phys_qubits + self.v.num_qubits()


class ShallowStateTensor(cirq.Gate):
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


class HorizontalSwapTest(cirq.Gate):
    def __init__(self, u: cirq.Gate, v: cirq.Gate, bond_dim: int):
        self.U = u
        self.V = v
        self.state = State(self.U, self.V, 1)
        self.bond_dim = bond_dim

    def num_qubits(self):
        return self.state.num_qubits() + self.V.num_qubits()

    def _decompose_(self, qubits):
        state_qbs = self.state.num_qubits()
        env_qbs = self.V.num_qubits()
        target_qbs = range(int(log2(self.bond_dim)))

        cnots = [cirq.CNOT(qubits[i], qubits[i+state_qbs]) for i in target_qbs]
        hadamards = [cirq.H(qubits[i]) for i in target_qbs]
        return [self.state._decompose_(qubits[:state_qbs]), self.V(*qubits[state_qbs:state_qbs+env_qbs])] +\
            cnots + hadamards


class VerticalSwapOptimizer(Optimizer):
    def objective_function(self, v_params):
        trial_environment = ShallowEnvironment(self.bond_dim, v_params)
        state = State(self.u, trial_environment, 1)

        qubits = cirq_qubits(state.num_qubits())
        aux_qubits = int(self.v.num_qubits() / 2)
        circuit = cirq.Circuit.from_ops([state.on(*qubits),
                                         cirq.inverse(trial_environment).on(*(qubits[:aux_qubits] +
                                                                              qubits[-aux_qubits:]))])
        self.circuit = circuit

        simulator = cirq.Simulator()
        results = simulator.simulate(circuit)

        target_qubits = self.get_target_qubits(aux_qubits, state.num_qubits())
        prob_zeros = sum(np.absolute(results.final_simulator_state.state_vector[:int(target_qubits)])**2)
        return 1-prob_zeros

    @staticmethod
    def get_target_qubits(aux_qubits, num_qubits):
        other_qbs = num_qubits - aux_qubits
        return 2**other_qbs - 1

    def update_final_circuits(self):
        v_params = self.optimized_result.x
        self.v = ShallowEnvironment(self.bond_dim, v_params)


class VerticalSampleSwapOptimizer(Optimizer):
    def __init__(self, u_original: cirq.Gate, v_original: cirq.Gate, reps: int, **kwargs):
        super().__init__(u_original, v_original, **kwargs)
        self.reps = reps

    def objective_function(self, params):
        trial_environment = ShallowEnvironment(self.bond_dim, params)
        state = State(self.u, trial_environment, 1)

        qubits = cirq_qubits(state.num_qubits())
        aux_qubs = int(self.v.num_qubits() / 2)
        circuit = cirq.Circuit.from_ops([state.on(*qubits),
                                         cirq.inverse(trial_environment).on(*(qubits[:aux_qubs] + qubits[-aux_qubs:])),
                                         cirq.measure(*qubits[:aux_qubs])])

        self.circuit = circuit
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=self.reps)

        key = list(result.measurements.keys())[0]
        counter = result.histogram(key=key, fold_func=lambda e: sum(e))  # sum up the number of 1s in the measurements
        mean = sum(counter.elements()) / self.reps
        return mean

    def update_final_circuits(self):
        v_params = self.optimized_result.x
        self.v = ShallowEnvironment(self.bond_dim, v_params)


class HorizontalSampleSwapOptimizer(Optimizer):
    def __init__(self, u_original: cirq.Gate, reps: int, v_original: cirq.Gate = None, **kwargs):
        super().__init__(u_original, v_original, **kwargs)
        self.reps = reps

    def objective_function(self, params):
        trial_environment = ShallowEnvironment(self.bond_dim, params)
        state = State(self.u, trial_environment, 1)
        horizontal_swap_gates = HorizontalSwapTest(self.u, trial_environment, self.bond_dim)

        state_qubits = state.num_qubits()
        env_qubits = trial_environment.num_qubits()
        total_qubits = state_qubits+env_qubits
        aux_qubs = int(env_qubits/2)
        qubits = cirq_qubits(total_qubits)

        circuit = cirq.Circuit.from_ops([horizontal_swap_gates(*qubits),
                                         cirq.measure(qubits[:aux_qubs] + qubits[state_qubits:state_qubits+aux_qubs])])
        self.circuit = circuit

        simulator = cirq.Simulator()
        results = simulator.run(circuit, repetitions=self.reps)
        key = list(results.measurements.keys())[0]
        counter = results.histogram(key=key)

        measure_qubits = 2*aux_qubs
        all_ones = int((2**measure_qubits)-1)
        prob_all_ones = counter[all_ones]/self.reps
        return prob_all_ones

    def update_final_circuits(self):
        v_params = self.optimized_result.x
        self.v = ShallowEnvironment(self.bond_dim, v_params)


class HorizontalSwapOptimizer(Optimizer):
    def __init__(self, u_original: cirq.Gate, qaoa_depth: int, v_original: cirq.Gate = None, **kwargs):
        super().__init__(u_original, v_original, qaoa_depth=qaoa_depth, **kwargs)

    def objective_function(self, params):
        trial_environment = ShallowEnvironment(self.bond_dim, params)
        state = State(self.u, trial_environment, 1)
        horizontal_swap_gates = HorizontalSwapTest(self.u, trial_environment, self.bond_dim)

        # useful qubit numbers to build the circuit
        state_qubits = state.num_qubits()
        env_qubits = trial_environment.num_qubits()
        total_qubits = state_qubits+env_qubits
        aux_qubs = int(env_qubits/2)

        qubits = cirq_qubits(total_qubits)
        circuit = cirq.Circuit.from_ops([horizontal_swap_gates(*qubits)])
        self.circuit = circuit

        simulator = cirq.Simulator()
        results = simulator.simulate(circuit)

        # identify the reduced density matrix of the measured qubits and minimise the probability of
        # measuring all |11...>
        density_matrix_qubits = list(qubits[:aux_qubs]) + list(qubits[state_qubits:state_qubits+aux_qubs])
        density_matrix = results.density_matrix_of(density_matrix_qubits)
        prob_all_ones = density_matrix[-1, -1]
        return np.abs(prob_all_ones)

    def update_final_circuits(self):
        v_params = self.optimized_result.x
        self.v = ShallowEnvironment(self.bond_dim, v_params)
