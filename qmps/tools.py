from numpy import eye, concatenate, allclose, swapaxes, tensordot
from numpy import array, pi as π, arcsin, sqrt, real, imag, split
from numpy import zeros, block, diag, log2
from numpy.random import rand, randint, randn
from numpy.linalg import svd, qr
import numpy as np
from xmps.spin import U4
from scipy.linalg import null_space, norm, svd
from scipy.optimize import minimize
from qmps.States import FullStateTensor, ShallowStateTensor, State
import matplotlib.pyplot as plt

import os
from typing import Callable, List, Dict
import cirq


def get_circuit(state, decomp=None):
    if decomp == 'Full':
        return cirq.Circuit.from_ops(cirq.decompose(state(*cirq_qubits(state.num_qubits()))))
    elif decomp == 'Once':
        return cirq.Circuit.from_ops(cirq.decompose_once(state(*cirq_qubits(state.num_qubits()))))
    else:
        return cirq.Circuit.from_ops(state(*cirq_qubits(state.num_qubits())))

def random_unitary(*args):
    return qr(randn(*args))[0]

def svals(A):
    return svd(A)[1]


def from_real_vector(v):
    '''helper function - put list of elements (real, imaginary) into a complex vector'''
    re, im = split(v, 2)
    return (re+im*1j)


def to_real_vector(A):
    '''takes a matrix, breaks it down into a real vector'''
    re, im = real(A).reshape(-1), imag(A).reshape(-1)  
    return concatenate([re, im], axis=0)


def eye_like(A):
    """eye_like: identity same shape as A
    """
    return eye(A.shape[0])


def cT(tensor):
    """H: Hermitian conjugate of last two indices of a tensor

    :param tensor: tensor to conjugate
    """
    return swapaxes(tensor.conj(), -1, -2)


def direct_sum(A, B):
    '''direct sum of two matrices'''
    (a1, a2), (b1, b2) = A.shape, B.shape
    O = zeros((a2, b1))
    return block([[A, O], [O.T, B]])


def unitary_extension(Q, D=None):
    '''extend an isometry to a unitary (doesn't check its an isometry)'''
    s = Q.shape
    flipped=False
    N1 = null_space(Q)
    N2 = null_space(Q.conj().T)
    
    if s[0]>s[1]:
        Q_ = concatenate([Q, N2], 1)
    elif s[0]<s[1]:
        Q_ = concatenate([Q.conj().T, N1], 1).conj().T
    else:
        Q_ = Q

    if D is not None:
        if D > Q_.shape[0]:
            Q_ = direct_sum(Q_, eye(D-Q_.shape[0]))

    return Q_


def environment_to_unitary(v):
    '''put matrix in form
              ↑ ↑
              | |
              ___
               v
              ___
              | |
      '''
    v = v.reshape(1, -1)/norm(v)
    vs = null_space(v).conj().T
    return concatenate([v, vs], 0).T


def environment_from_unitary(u):
    '''matrix out of form
              ↑ ↑
              | |
              ___
               v   
              ___
              | |
      '''
    return (u@array([1, 0, 0, 0])).reshape(2, 2)


def tensor_to_unitary(A, testing=False):
    """given a left isometric tensor A, put into a unitary.
       NOTE: A should be left canonical: No checks!
    """
    d, D, _ = A.shape
    iso = A.transpose([1, 0, 2]).reshape(D*d, D)
    U = unitary_extension(iso)
    if testing:
        passed = allclose(cT(iso)@iso, eye(2)) and \
                 allclose(U@cT(U), eye(4)) and \
                 allclose(cT(U)@U, eye(4)) and \
                 allclose(U[:iso.shape[0], :iso.shape[1]], iso) and\
                 allclose(tensordot(U.reshape(2, 2, 2, 2), array([1, 0]), [2, 0]).reshape(4, 2), 
                                iso)
        return U, passed


    #  ↑ j
    #  | |
    #  ---       
    #   u  = i--A--j
    #  ---      |
    #  | |      σ
    #  i σ 

    return U


def unitary_to_tensor(U):
    n = int(log2(U.shape[0]))
    return tensordot(U.reshape(*2 * n * [2]), array([1, 0]), [n, 0]).reshape(2 ** (n - 1), 2, 2 ** (n - 1)).transpose(
    [1, 0, 2])


def cirq_qubits(num):
    return cirq.LineQubit.range(num)


def split_2s(x):
    """split_2s: take a list: [β, γ, β, γ, ...], return [[β, γ], [β, γ], ...]
    """
    return [x[i:i+2] for i in range(len(x)) if not i%2]


class OptimizerCircuit:
    def __init__(self, circuit=None, total_qubits=None, aux_qubits=None):
        self.circuit = circuit
        self.total_qubits = total_qubits
        self.aux_qubits = aux_qubits
        self.qubits = None


class Optimizer:
    def __init__(self, u, v):
        self.u = u
        self.v = v

        self.initial_guess = None
        self.iters = 0
        self.optimized_result = None
        self.obj_fun_values = []
        self.settings = {
            'maxiter': 100,
            'verbose': False,
            'method': 'Powell',
            'tol': 1e-8,
            'store_values': False
        }
        self.is_verbose = self.settings['verbose']
        self.circuit = OptimizerCircuit()

    def _settings_(self, new_settings):
        return self.settings.update(new_settings)

    def gate_from_params(self, params):
        pass

    def update_state(self):
        pass

    def callback_store_values(self, xk):
        val = self.objective_function(xk)
        self.obj_fun_values.append(val)
        if self.is_verbose:
            print(f'{self.iters}:{val}')
        self.iters += 1

    # def get_circuit(self, params):
    #     pass
    #
    # def calc_objective_function(self, circuit):
    #     pass

    def objective_function(self, params):
        pass

    def optimize(self):
        options = {'maxiter': self.settings['maxiter'],
                   'disp': self.settings['verbose']}

        kwargs = {'fun': self.objective_function,
                  'x0': self.initial_guess,
                  'method': self.settings['method'],
                  'tol': self.settings['tol'],
                  'options': options,
                  'callback': self.callback_store_values if self.settings['store_values'] else None}

        self.optimized_result = minimize(**kwargs)
        # maybe implement genetic evolution algorithm or particle swarm?
        # self.optimized_result = differential_evolution(self.objective_function)
        self.update_state()
        if self.is_verbose:
            print(f'Reason for termination is {self.optimized_result.message}')

    def plot_convergence(self, file, exact_value=None):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.figure()
        x = range(self.iters)
        plt.plot(x, self.obj_fun_values)
        if exact_value is not None:
            plt.axhline(exact_value, c='r')
        plt.xlabel('iterations')
        plt.ylabel('objective function value')
        plt.show()
        if file:
            plt.savefig(dir_path + '/' + file)


class QAOAOptimizer(Optimizer):
    def __init__(self, u, v, depth, initial_guess):
        super().__init__(u, v)
        self.bond_dim = int(2**(u.num_qubits()-1))
        self.depth = depth
        self.initial_guess = initial_guess if initial_guess is not None else np.random.rand(2 * depth)

    def gate_from_params(self, params):
        return ShallowStateTensor(self.bond_dim, params)


class FullOptimizer(Optimizer):
    def __init__(self, u, v, initial_guess):
        super().__init__(u, v)
        self.initial_guess = initial_guess if initial_guess is not None else np.random.rand(15)

    def gate_from_params(self, params):
        return FullStateTensor(U4(params))


# Environment Optimizers ##################
class EvoQAOAOptimizer(QAOAOptimizer):
    def __init__(self, u, depth, initial_guess=None, v=None):
        super().__init__(u, v, depth, initial_guess)

    def update_state(self):
        self.v = self.gate_from_params(self.optimized_result.x)


class EvoFullOptimizer(FullOptimizer):
    def __init__(self, u, initial_guess=None, v=None):
        super().__init__(u, v, initial_guess)

    def update_state(self):
        self.v = self.gate_from_params(self.optimized_result.x)


#########################################################

# State Optimizers #############################

class StateQAOAOptimizer(QAOAOptimizer):
    def __init__(self, u, v, depth, hamiltonian, initial_guess=None):
        super().__init__(u, v, depth, initial_guess)
        self.hamiltonian = hamiltonian

    def update_state(self):
        self.u = self.gate_from_params(self.optimized_result.x)


class StateFullOptimizer(FullOptimizer):
    def __init__(self, u, v, hamiltonian, initial_guess=None):
        super().__init__(u, v, initial_guess)
        self.hamiltonian = hamiltonian

    def update_state(self):
        self.u = self.gate_from_params(self.optimized_result.x)


############################################################

# Vertical Optimizers #####################

class VerticalEvoQAOASimulate(EvoQAOAOptimizer):

    def objective_function(self, params):
        environment = ShallowStateTensor(self.bond_dim, params)
        state = State(self.u, environment)  # ignore these yellow errors

        aux_qubits = int(environment.num_qubits() / 2)
        self.circuit.aux_qubits = aux_qubits
        self.circuit.total_qubits = state.num_qubits()

        qubits = cirq.LineQubit.range(self.circuit.total_qubits)
        self.circuit.qubits = qubits
        self.circuit.circuit = cirq.Circuit.from_ops([state.on(*qubits),
                                                      cirq.inverse(environment).on(
                                                          *(qubits[:aux_qubits] + qubits[-aux_qubits:]))])

        other_qubits = self.circuit.total_qubits - self.circuit.aux_qubits
        target_strings = int(2 ** other_qubits)

        simulator = cirq.Simulator()
        results = simulator.simulate(self.circuit.circuit)

        # prob_zeros = sum(np.absolute(results.final_simulator_state.state_vector[:target_strings]) ** 2)

        density_matrix_qubits = self.circuit.qubits[:self.circuit.aux_qubits]
        density_matrix = results.density_matrix_of(density_matrix_qubits)
        prob_zeros = np.abs(density_matrix[0, 0])
        return 1 - prob_zeros


class VerticalEvoFullSimulate(EvoFullOptimizer):

    def objective_function(self, params):
        environment = FullStateTensor(U4(params))
        state = State(self.u, environment)  # ignore these yellow errors

        aux_qubits = int(environment.num_qubits() / 2)
        self.circuit.aux_qubits = aux_qubits
        self.circuit.total_qubits = state.num_qubits()

        qubits = cirq.LineQubit.range(self.circuit.total_qubits)
        self.circuit.qubits = qubits
        self.circuit.circuit = cirq.Circuit.from_ops([state.on(*qubits),
                                                      cirq.inverse(environment).on(
                                                          *(qubits[:aux_qubits] + qubits[-aux_qubits:]))])

        other_qubits = self.circuit.total_qubits - self.circuit.aux_qubits

        simulator = cirq.Simulator()
        results = simulator.simulate(self.circuit.circuit)

        density_matrix_qubits = self.circuit.qubits[:self.circuit.aux_qubits]
        density_matrix = results.density_matrix_of(density_matrix_qubits)
        prob_zeros = np.abs(density_matrix[0, 0])

        #prob_zeros = sum(np.absolute(results.final_simulator_state.state_vector[:target_strings]) ** 2)
        return 1 - prob_zeros


class VerticalEvoQAOASample(EvoQAOAOptimizer):
    def __init__(self, u, depth, reps):
        super().__init__(u, depth)
        self.reps = reps

    def objective_function(self, params):
        environment = ShallowStateTensor(self.bond_dim, params)
        state = State(self.u, environment)  # ignore these yellow errors

        aux_qubits = int(environment.num_qubits() / 2)
        self.circuit.aux_qubits = aux_qubits
        self.circuit.total_qubits = state.num_qubits()

        qubits = cirq.LineQubit.range(self.circuit.total_qubits)

        self.circuit.circuit = cirq.Circuit.from_ops([state.on(*qubits),
                                                      cirq.inverse(environment).on(
                                                          *(qubits[:aux_qubits] + qubits[-aux_qubits:])),
                                                      cirq.measure(*qubits[:aux_qubits])])

        simulator = cirq.Simulator()
        result = simulator.run(self.circuit.circuit, repetitions=self.reps)

        key = list(result.measurements.keys())[0]
        counter = result.histogram(key=key, fold_func=lambda e: sum(e))  # sum up the number of 1s in the measurements
        mean = sum(counter.elements()) / self.reps
        return mean


class VerticalEvoFullSample(EvoFullOptimizer):
    def __init__(self, u, reps):
        super().__init__(u)
        self.reps = reps

    def objective_function(self, params):
        environment = FullStateTensor(U4(params))
        state = State(self.u, environment)  # ignore these yellow errors

        aux_qubits = int(environment.num_qubits() / 2)
        self.circuit.aux_qubits = aux_qubits
        self.circuit.total_qubits = state.num_qubits()

        qubits = cirq.LineQubit.range(self.circuit.total_qubits)

        self.circuit.circuit = cirq.Circuit.from_ops([state.on(*qubits),
                                                      cirq.inverse(environment).on(
                                                          *(qubits[:aux_qubits] + qubits[-aux_qubits:])),
                                                      cirq.measure(*qubits[:aux_qubits])])

        simulator = cirq.Simulator()
        result = simulator.run(self.circuit.circuit, repetitions=self.reps)

        key = list(result.measurements.keys())[0]
        counter = result.histogram(key=key, fold_func=lambda e: sum(e))  # sum up the number of 1s in the measurements
        mean = sum(counter.elements()) / self.reps
        return mean


class VerticalStateQAOASimulate(StateQAOAOptimizer):
    def objective_function(self, params):
        target_u = ShallowStateTensor(self.bond_dim, params)
        physical_qubits = self.hamiltonian.num_qubits()
        original_state = State(self.u, self.v, n=physical_qubits)
        target_state = State(target_u, self.v, n=physical_qubits)

        aux_qubits = int(self.v.num_qubits() / 2)
        qubits = cirq.LineQubit.range(original_state.num_qubits())
        self.circuit.circuit = cirq.Circuit.from_ops([original_state(*qubits),
                                                      self.hamiltonian(
                                                          *qubits[aux_qubits:aux_qubits + physical_qubits]),
                                                      cirq.inverse(target_state).on(*qubits)])

        simulator = cirq.Simulator()
        results = simulator.simulate(self.circuit.circuit)

        final_state = results.final_simulator_state.state_vector[0]
        score = np.abs(final_state) ** 2
        return 1 - score


class VerticalStateFullSimulate(StateFullOptimizer):
    def objective_function(self, params):
        target_u = FullStateTensor(U4(params))
        physical_qubits = self.hamiltonian.num_qubits()
        original_state = State(self.u, self.v, n=physical_qubits)
        target_state = State(target_u, self.v, n=physical_qubits)

        aux_qubits = int(self.v.num_qubits() / 2)
        qubits = cirq.LineQubit.range(original_state.num_qubits())
        self.circuit.circuit = cirq.Circuit.from_ops([original_state(*qubits),
                                                      self.hamiltonian(
                                                          *qubits[aux_qubits:aux_qubits + physical_qubits]),
                                                      cirq.inverse(target_state).on(*qubits)])

        simulator = cirq.Simulator()
        results = simulator.simulate(self.circuit.circuit)

        final_state = results.final_simulator_state.state_vector[0]
        score = np.abs(final_state) ** 2
        return 1 - score

#  Horizontal Optimizers #####################

##############################################################


class HorizontalEvoFullSimulate(EvoFullOptimizer):

    def objective_function(self, params):
        environment = FullStateTensor(U4(params))
        state = State(self.u, environment, 1)

        state_qubits = state.num_qubits()
        env_qubits = environment.num_qubits()

        total_qubits = state_qubits+env_qubits
        self.circuit.total_qubits = state_qubits

        qubits = cirq.LineQubit.range(total_qubits)
        self.circuit.qubits = qubits

        aux_qubits = int(environment.num_qubits()/2)
        self.circuit.aux_qubits = aux_qubits

        target_qubits = range(aux_qubits)
        cnots = [cirq.CNOT(qubits[i], qubits[i+state_qubits]) for i in target_qubits]
        hadamards = [cirq.H(qubits[i]) for i in target_qubits]

        circuit = cirq.Circuit.from_ops([state.on(*qubits[:state_qubits]),
                                         environment.on(*qubits[state_qubits:])] + cnots + hadamards)
        self.circuit.circuit = circuit

        simulator = cirq.Simulator()
        results = simulator.simulate(circuit)

        state_qubits = self.circuit.total_qubits
        aux_qubits = self.circuit.aux_qubits
        qubits = self.circuit.qubits

        density_matrix_qubits = list(qubits[:aux_qubits]) + list(qubits[state_qubits:state_qubits+aux_qubits])
        density_matrix = results.density_matrix_of(density_matrix_qubits)
        prob_all_ones = density_matrix[-1, -1]
        return np.abs(prob_all_ones)


class HorizontalEvoQAOASimulate(EvoQAOAOptimizer):
    def objective_function(self, params):
        environment = ShallowStateTensor(self.bond_dim, params)
        state = State(self.u, environment, 1)

        state_qubits = state.num_qubits()
        env_qubits = environment.num_qubits()

        total_qubits = state_qubits + env_qubits
        self.circuit.total_qubits = state_qubits

        qubits = cirq.LineQubit.range(total_qubits)
        self.circuit.qubits = qubits

        aux_qubits = environment.num_qubits() / 2
        self.circuit.aux_qubits = aux_qubits

        target_qubits = range(len(aux_qubits))
        cnots = [cirq.CNOT(qubits[i], qubits[i + state_qubits]) for i in target_qubits]
        hadamards = [cirq.H(qubits[i]) for i in target_qubits]

        circuit = cirq.Circuit.from_ops([state.on(*qubits[:state_qubits]),
                                         environment.on(*qubits[state_qubits:])] + cnots + hadamards)
        self.circuit.circuit = circuit

        simulator = cirq.Simulator()
        results = simulator.simulate(circuit)

        state_qubits = self.circuit.total_qubits
        aux_qubits = self.circuit.aux_qubits
        qubits = self.circuit.qubits

        density_matrix_qubits = list(qubits[:aux_qubits]) + list(qubits[state_qubits:state_qubits + aux_qubits])
        density_matrix = results.density_matrix_of(density_matrix_qubits)
        prob_all_ones = density_matrix[-1, -1]
        return np.abs(prob_all_ones)


class RepresentMPS:
    def __new__(cls, u, vertical='Vertical', ansatz='Full', simulate='Simulate', **kwargs):
        optimizer_choice = {
            'QAOA': {
                'Simulate': {
                    'Vertical': VerticalEvoQAOASimulate,
                    'Horizontal': HorizontalEvoQAOASimulate
                },
                'Sample': {
                    'Vertical': VerticalEvoQAOASample,
                    'Horizontal': None
                }
            },
            'Full': {
                'Simulate': {
                    'Vertical': VerticalEvoFullSimulate,
                    'Horizontal': HorizontalEvoFullSimulate
                },
                'Sample': {
                    'Vertical': VerticalEvoFullSample,
                    'Horizontal': None
                }
            }
        }

        return optimizer_choice[ansatz][simulate][vertical](u, **kwargs)


class TimeEvolveOptimizer:
    def __new__(cls, u, v, hamiltonian, vertical='Vertical', ansatz='Full', simulate='Simulate', **kwargs):
        optimizer_choice = {
            'QAOA': {
                'Simulate': {
                    'Vertical': VerticalStateQAOASimulate,
                    'Horizontal': None
                },
                'Sample': {
                    'Vertical': None,
                    'Horizontal': None
                }
            },
            'Full': {
                'Simulate': {
                    'Vertical': VerticalStateFullSimulate,
                    'Horizontal': None
                },
                'Sample': {
                    'Vertical': None,
                    'Horizontal': None
                }
            }
        }
        return optimizer_choice[ansatz][simulate][vertical](u, v, hamiltonian=hamiltonian, **kwargs)


class MPSTimeEvolve:
    def __init__(self, u_initial: cirq.Gate, hamiltonian: cirq.Gate, v_initial: cirq.Gate = None, depth: int=0,
                 settings=None, optimizer_settings=None,
                 reps=0):
        self.u = u_initial
        self.hamiltonian = hamiltonian

        self.kwargs = {}
        if reps:
            self.kwargs.update({'reps': reps})
        if depth:
            self.kwargs.update({'depth': depth})

        self.optimizer_settings = optimizer_settings if optimizer_settings else \
            {'vertical': 'Vertical', 'ansatz': 'Full', 'simulate': 'Simulate'}
        self.evo_optimizer_settings = self.optimizer_settings.copy()
        self.evo_optimizer_settings.update({'vertical': 'Horizontal'})

        self.TimeEvoOptimizer = None
        self.EnvOptimizer = None

        self.settings = settings

        self.initial_guess_u = None
        self.initial_guess_v = None

        self.v = v_initial
        if not v_initial:
            self.v = self.get_v_params().v

    def get_v_params(self):
        self.EnvOptimizer = RepresentMPS(self.u, initial_guess=self.initial_guess_v,
                                         **self.evo_optimizer_settings, **self.kwargs)

        if self.settings:
            self.EnvOptimizer._settings_(self.settings)
        self.EnvOptimizer.optimize()
        self.initial_guess_v = self.EnvOptimizer.optimized_result.x
        return self.EnvOptimizer

    def get_u_params(self):
        self.TimeEvoOptimizer = TimeEvolveOptimizer(self.u, self.v, hamiltonian=self.hamiltonian,
                                                    initial_guess=self.initial_guess_u,
                                                    **self.optimizer_settings,
                                                    **self.kwargs)
        if self.settings:
            self.TimeEvoOptimizer._settings_(self.settings)

        self.TimeEvoOptimizer.optimize()
        self.initial_guess_u = self.TimeEvoOptimizer.optimized_result.x
        return self.TimeEvoOptimizer

    def evolve_single_step(self):
        self.u = self.get_u_params().u
        self.v = self.get_v_params().v

    def evolve_multiple_steps(self, steps):
        for _ in range(steps):
            self.evolve_single_step()

    def simulate_state(self):
        state = State(self.u, self.v, 1)
        qubits = cirq.LineQubit.range(state.num_qubits())
        circuit = cirq.Circuit.from_ops([state.on(*qubits)])
        simulator = cirq.Simulator()
        return simulator.simulate(circuit), qubits

    def evolve_bloch_sphere(self, evo_steps):
        current_step = 0
        n_qubits = self.hamiltonian.num_qubits()
        qubit_1 = []

        results, qubits = self.simulate_state()
        qb1 = results.bloch_vector_of(qubits[1])
        qubit_1.append(qb1)

        while current_step < evo_steps:
            # evolve a single step
            self.evolve_single_step()

            # simulate the new state
            results, qubits = self.simulate_state()

            # get bloch sphere of physical qubit
            qb1 = results.bloch_vector_of(qubits[1])

            # record results
            qubit_1.append(qb1)

            current_step += 1

        x_evo = [step[0] for step in qubit_1]
        y_evo = [step[1] for step in qubit_1]
        z_evo = [step[2] for step in qubit_1]
        return x_evo, y_evo, z_evo

    def loschmidt_echo(self, steps):
        '''
        Search for loshmidt echos in Ising Hamiltonian
        :param steps: time steps

        Value that is being evaluated is the square of the Loschmidt amplitude:
        https://royalsocietypublishing.org/doi/pdf/10.1098/rsta.2015.0160
        '''
        original_state, original_qubits = self.simulate_state()

        original_wavefunction = original_state.final_simulator_state.state_vector

        state_overlap = []
        current_step = 0
        while current_step < steps:
            self.evolve_single_step()
            new_state, _ = self.simulate_state()
            new_wavefunction = new_state.final_simulator_state.state_vector

            overlap = np.abs(np.dot(original_wavefunction, new_wavefunction.conj()))**2
            state_overlap.append(overlap)
            current_step += 1
            print(current_step)
        return state_overlap



def sampled_bloch_vector_of(qubit, circuit, reps=1000000):
    """sampled_bloch_vector_of: get bloch vector of a 
    specified qubit by sampling. 

    :param qubit: qubit to sample bloch vector of 
    :param circuit: circuit to evaluate before sampling
    :param reps: number of measurements on each qubit
    """
    sim = cirq.Simulator()
    C = circuit.copy()
    C.append([cirq.measure(qubit, key='z')])
    meas = sim.run(C, repetitions=reps).measurements['z']
    z = array(list(map(int, meas))).mean()

    C = circuit.copy()
    C.append([cirq.inverse(cirq.S(qubit)), cirq.H(qubit), cirq.measure(qubit, key='y')])
    meas = sim.run(C, repetitions=reps).measurements['y']
    y = array(list(map(int, meas))).mean()

    C = circuit.copy()
    C.append([cirq.H(qubit), cirq.measure(qubit, key='x')])
    meas = sim.run(C, repetitions=reps).measurements['x']
    x = array(list(map(int, meas))).mean()

    return -2*array([x, y, z])+1

def random_sparse_circuit(length, depth=10, p=0.5):
    '''10.1103/PhysRevA.75.062314'''
    qubits = cirq.LineQubit.range(length)
    circuit = cirq.Circuit()

    def U(i):
        """U: Random SU(2) element"""
        ψ = 2*π*rand()
        χ = 2*π*rand()
        φ = arcsin(sqrt(rand()))
        for g in [cirq.Rz(χ+ψ), cirq.Ry(2*φ), cirq.Rz(χ-ψ)]:
            yield g(cirq.LineQubit(i))
    for i in range(depth):
        if rand()>p:
            # one qubit gate
            circuit.append(U(randint(0, length)))
        else:
            # two qubit gate
            i = randint(0, length-1)
            if rand()>0.5:
                circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
            else:
                circuit.append(cirq.CNOT(qubits[i+1], qubits[i]))
    return circuit

def random_circuit(length, depth=10, p=0.5, ψχϕs=None):
    qubits = cirq.LineQubit.range(length)
    circuit = cirq.Circuit()
    ψχϕs = [[(None, None, None) for _ in range(length)]
            for _ in range(depth)] if ψχϕs is None else ψχϕs

    def U(i, ψ=None, χ=None, ϕ=None):
        """U: Random SU(2) element"""
        ψ = 2*π*rand() if ψ is None else ψ
        χ = 2*π*rand() if χ is None else χ
        φ = arcsin(sqrt(rand())) if ϕ is None else ϕ
        for g in [cirq.Rz(χ+ψ), cirq.Ry(2*φ), cirq.Rz(χ-ψ)]:
            yield g(cirq.LineQubit(i))
    for j in range(depth):
        for i in range(length):
            circuit.append(U(i, *ψχϕs[j][i]))
            # two qubit gate
        for i in range(length-1):
            if rand()>0.5:
                circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
            else:
                circuit.append(cirq.CNOT(qubits[i+1], qubits[i]))
    return circuit

def random_qaoa_circuit(length, depth=1, βγs=None):
    """qaoa_circuit: qaoa circuit with qubits - useful for testing.
    """
    βγs = [[(randn(), randn()) for _ in range(length)] for _ in range(depth)] if βγs is None else βγs

    qubits = cirq.LineQubit.range(length)
    c =  cirq.Circuit().from_ops([[[cirq.X(qubit)**β for qubit in qubits]+\
                                   [cirq.ZZ(qubits[i], qubits[i+1])**γ for i in range(len(qubits)-1)]
                                   for β, γ in βγs[i]] for i in range(depth)])
    return c

def random_full_rank_circuit(length, depth, ψχϕs=None):
    ψχϕs = [[(None, None, None) for _ in range(length)]
            for _ in range(depth)] if ψχϕs is None else ψχϕs
    qubits = cirq.LineQubit.range(length)
    circuit = cirq.Circuit()

    def U(i, ψ=None, χ=None, ϕ=None):
        """U: Random SU(2) element"""
        ψ = 2*π*rand() if ψ is None else ψ
        χ = 2*π*rand() if χ is None else χ
        φ = arcsin(sqrt(rand())) if ϕ is None else ϕ
        for g in [cirq.Rz(χ+ψ), cirq.Ry(2*φ), cirq.Rz(χ-ψ)]:
            yield g(cirq.LineQubit(i))
    for j in range(depth):

        # Define a parametrisation of su(2**(N-1))
        # MPS matrices will be U(...), xU(...)
        for i in range(1, length):
            circuit.append(U(i, *ψχϕs[j][i]))
        circuit.append(reversed([cirq.CNOT(qubits[i], qubits[i+1]) for i in range(1, length-1)]))

        # Add on all the rest
        circuit.append(cirq.H(qubits[0]))
        circuit.append(cirq.CNOT(qubits[0], qubits[1]))
        circuit.append(cirq.SWAP(qubits[i], qubits[i+1]) for i in range(length-1))
        circuit.append(cirq.SWAP(qubits[-1], qubits[0]))
    return circuit

