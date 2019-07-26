import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import minimize
import cirq
from qmps.represent import ShallowStateTensor, FullStateTensor
from xmps.spin import U4
from scipy.stats import unitary_group


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
        return [self.v(*qubits[n:n + v_qbs])] + [self.u(*qubits[i:i + u_qbs]) for i in range(n)]

    def num_qubits(self):
        return self.n_phys_qubits + self.v.num_qubits()


class IsingHamiltonian(cirq.TwoQubitGate):
    def __init__(self, lamda, delta):
        self.l = lamda
        self.d = delta
        self.t = -2 * self.d / np.pi

    def num_qubits(self):
        return 2

    def _decompose_(self, qubits):
        return cirq.XPowGate(exponent=self.t).on(qubits[0]), cirq.XPowGate(exponent=self.t).on(qubits[1]), \
               cirq.ZZPowGate(exponent=self.t).on(*qubits)

    def _circuit_diagram_info_(self, args):
        return 'H'


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

# Ansatz Optimizers ##################


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


if __name__ == '__main__':

    U = FullStateTensor(np.identity(4))
    H = IsingHamiltonian(0.5, 0.01)
    _settings = {
        'verbose': True,
        'store_value': True,
        'maxiter': 10000,
        'method': 'Nelder-Mead'
    }

    optimizer_settings = {
        'vertical': 'Vertical', 'ansatz': 'Full', 'simulate': 'Simulate'
    }

    timeEvolve = MPSTimeEvolve(U, hamiltonian=H, optimizer_settings=optimizer_settings, settings=_settings)
    state_overlap = timeEvolve.loschmidt_echo(400)
    # represent = RepresentMPS(U, **optimizer_settings)
    # represent._settings_(settings)
    # represent.optimize()
    #
    # test_state = State(represent.u, represent.v)
    # qubits = cirq.LineQubit.range(test_state.num_qubits())
    # sim = cirq.Simulator()
    #
    # circuit1 = cirq.Circuit.from_ops([test_state.on(*qubits)])
    # circuit2 = cirq.Circuit.from_ops([test_state.on(*qubits),
    #                                   cirq.inverse(represent.v).on(qubits[0], qubits[2])])
    # circuit3 = cirq.Circuit.from_ops([represent.v.on(qubits[0], qubits[1])])
    #
    # results1, results2, results3 = sim.simulate(circuit1), sim.simulate(circuit2), sim.simulate(circuit3)


