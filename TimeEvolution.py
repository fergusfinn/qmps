import cirq
from unitary_iMPS import log2
from SwapTest import Tensor, ShallowStateTensor, ShallowEnvironment, get_circuit, rprint, cirq_qubits,\
    Optimizer, FullSwapOptimizer, text_print, tprint, SingleQubitState
import numpy as np
from scipy.stats import unitary_group
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class TwoQubitState(cirq.Gate):
    def __init__(self, u:ShallowStateTensor, v:ShallowEnvironment):
        self.u = u
        self.v = v
        self.bond_dim = int(2 ** (self.u.num_qubits() - 1))
        assert v.num_qubits() == int(2*log2(self.bond_dim))
        self.n_qubits = int(2*log2(self.bond_dim) + 2)

    def _decompose_(self, qubits):
        u_qubits = self.u.num_qubits()
        v_qubits = self.v.num_qubits()
        return self.v.on(*qubits[2:v_qubits+2]), self.u.on(*qubits[1:u_qubits+1]), self.u.on(*qubits[:u_qubits])

    def num_qubits(self):
        return self.n_qubits


class TwoQubitHamiltonian(cirq.Gate):
    '''
    Class for representing the circuit of the two qubit hamiltonian:
    0   0   0   0
    |   |   |-V-|
    |   |-U-|   |
    |-U-|   |   |
    |   |-H-|   |
    '''
    def __init__(self, u: ShallowStateTensor,v: ShallowEnvironment, hamiltonian: cirq.Gate):
        self.tqs = TwoQubitState(u, v)
        self.hamiltonian = hamiltonian

    def _decompose_(self, qubits):
        bond_dim = self.tqs.bond_dim
        qubit_num = int(log2(bond_dim))
        return self.tqs._decompose_(qubits), self.hamiltonian(*qubits[qubit_num:qubit_num+2])

    def num_qubits(self):
        return self.tqs.num_qubits()


class TQHamSwapState(cirq.Gate):
    '''
        Class for representing the circuit of the two qubit hamiltonian having specified V, U, H
    0   0   0   0 = Initialization
    |   |   |-V-|
    |   |-U-|   |
    |-U-|   |   |
    |   |-H-|   |
    |U*-|   |   |
    |   |U*-|   |
    |   |   |V*-|
    M   M   M   M = Measurement
    '''
    def __init__(self, u_original: ShallowStateTensor, u_target: ShallowStateTensor,
                 environment: ShallowEnvironment, hamiltonian: Tensor):
        self.u_original = u_original
        self.environment = environment
        self.hamiltonian = hamiltonian
        self.u_target = u_target

    def _get_original_gates(self):
        return TwoQubitHamiltonian(self.u_original, self.environment, self.hamiltonian)

    def _get_target_gates(self):
        return cirq.inverse(TwoQubitState(self.u_target, self.environment))

    def num_qubits(self):
        return self._get_original_gates().num_qubits()

    def _decompose_(self, qubits):
        original_gates = self._get_original_gates()
        target_gates = self._get_target_gates()
        return original_gates._decompose_(qubits), target_gates._decompose_(qubits)


class SingleQubitHamiltonian(cirq.Gate):
    '''
    class to initialise gates to look like the following:
    0   0   0
    |   |   |
    |   |-V-|
    |-U-|   |
    |   H   |
    |   |   |
    '''

    def __init__(self, u: ShallowStateTensor, v: ShallowEnvironment, hamiltonian: cirq.Gate):
        self.u = u
        self.v = v
        self.ham = hamiltonian
        self.state = SingleQubitState(u, v)

    def num_qubits(self):
        return self.state.num_qubits()

    def _decompose_(self, qubits):
        aux_qubits = self.u.num_qubits() - 1
        return self.state._decompose_(qubits), self.ham(qubits[aux_qubits])


class SingleQubitHamSwap(cirq.Gate):
    '''
    initialise gates to produce circuit:
    0   0   0
    |   |-V-|
    |-U-|   |
    |   H   |
    |U'-|   |
    |   |V'-|
    M   M   M
    '''

    def __init__(self, u_original: ShallowStateTensor, u_target: ShallowStateTensor,
                 environment: ShallowEnvironment, hamiltonian: cirq.Gate):
        self.u_original = u_original
        self.hamiltonian = hamiltonian
        self.environment = environment
        self.u_target = u_target

    def get_original_gates(self):
        return SingleQubitHamiltonian(self.u_original, self.environment, self.hamiltonian)

    def get_target_gates(self):
        return cirq.inverse(SingleQubitState(self.u_target, self.environment))

    def num_qubits(self):
        return self.get_original_gates().num_qubits()

    def _decompose_(self, qubits):
        return self.get_original_gates()._decompose_(qubits), self.get_target_gates().on(*qubits)


class TimeEvoOptimizer(Optimizer):
    def __init__(self, u_params, v_params, bond_dim, hamiltonian, evo_qubits = 1):
        self.u_params = u_params
        self.v_params = v_params
        self.hamiltonian = hamiltonian
        self.bond_dim = bond_dim
        self.iters = 0
        self.reps = 10000
        self.obj_fun_values = []
        self.noisy = False
        self.store_values = False
        self.optimized_result = None
        self.circuit = None
        self.evo_qubits = evo_qubits

    @staticmethod
    def get_target_qubits(num_qubits, bond_dim):
        target_qbs = log2(bond_dim)
        other_qbs = num_qubits - target_qbs
        return 2 ** other_qbs - 1

    def objective_function(self, target_u_params):
        u_original = ShallowStateTensor(self.bond_dim, self.u_params)
        v = ShallowEnvironment(self.bond_dim, self.v_params)
        ham = Tensor(self.hamiltonian, symbol='H') if not isinstance(self.hamiltonian, cirq.Gate) else self.hamiltonian
        u_target = ShallowStateTensor(self.bond_dim, target_u_params)

        if self.evo_qubits == 2:
            state = TQHamSwapState(u_original=u_original, u_target=u_target, environment=v, hamiltonian=ham)

        if self.evo_qubits == 1:
            state = SingleQubitHamSwap(u_original, u_target, v, ham)

        self.circuit = get_circuit(state)

        simulator = cirq.Simulator()
        results = simulator.simulate(self.circuit)

        bloch_vec = results.bloch_vector_of(cirq_qubits(state.num_qubits())[0])
        score = (bloch_vec[2])
        return 1 - score

    def get_state(self, max_iter):
        options = {'maxiter': max_iter,
                   'disp': self.noisy}  # if noisy else False}

        kwargs = {'fun': self.objective_function,
                  'x0': np.random.rand(len(self.u_params)),
                  'method': 'Nelder-Mead',
                  'tol': 1e-5,
                  'options': options,
                  'callback': self.callback_store_values if self.store_values else None}

        self.optimized_result = minimize(**kwargs)
        if self.noisy:
            print(f'Reason for termination is {self.optimized_result.message}')


class QubitTimeEvolution:
    def __init__(self, u_params=None, hamiltonian=None, bond_dim=2, qaoa_depth=2, evo_qubits=1):
        self.u_params = u_params
        self.hamiltonian = hamiltonian
        self.bond_dim = bond_dim
        self.qaoa_depth = qaoa_depth
        self.v_params = self.get_v_params()
        self.TimeEvoOptimizer = None
        self.EnvOptimizer = None
        self.evo_qubits = evo_qubits

    def get_v_params(self, noisy=False):
        self.EnvOptimizer = FullSwapOptimizer(self.u_params, bond_dim=self.bond_dim, qaoa_depth=self.qaoa_depth)
        self.EnvOptimizer.set_noise(noisy)
        self.EnvOptimizer.get_env(max_iter=100)
        return self.EnvOptimizer.optimized_result.x

    def get_u_params(self, noisy=False):
        self.TimeEvoOptimizer = TimeEvoOptimizer(self.u_params, self.v_params, self.bond_dim, self.hamiltonian,
                                                 evo_qubits=self.evo_qubits)
        self.TimeEvoOptimizer.set_noise(noisy)
        self.TimeEvoOptimizer.get_state(max_iter=100)
        return self.TimeEvoOptimizer.optimized_result.x

    def evolve_single_step(self, noisy=False):
        self.u_params = self.get_u_params(noisy)
        self.v_params = self.get_v_params(noisy)

    def evolve_multiple_steps(self, steps, noisy=False):
        for _ in range(steps):
            self.evolve_single_step(noisy)


class TransverseIsing(cirq.Gate):
    '''
    Cirq gate that implements the transverse Ising Gate U = exp(iΔt(J Z_i . Z_(i+1)) + λXi)
    '''
    def __init__(self, j, time_step, lamda):
        self.J = j
        self.time_step = time_step
        self.lamda = lamda

    def _decompose_(self, qubits):
        return cirq.ZZPowGate(exponent=2 * self.time_step * self.J / np.pi).on(*qubits),\
               cirq.XPowGate(exponent=2 * self.lamda * self.time_step / np.pi).on(qubits[0]),\
               cirq.XPowGate(exponent=2 * self.lamda * self.time_step / np.pi).on(qubits[1])

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ['H']*self.num_qubits()


def main():
    u = [0]*2
    ham = cirq.X
    evo_steps = 10
    current_step = 0
    evolver = QubitTimeEvolution(u_params=u, bond_dim=2, hamiltonian=ham, qaoa_depth=1)
    qubit_0 = []
    qubit_1 = []
    qubit_2 = []
    while current_step < evo_steps:
        print(evolver.u_params, '\n', evolver.v_params)
        evolver.evolve_single_step(False)
        # prepare new optimized state
        u, v = evolver.u_params, evolver.v_params
        state = TwoQubitState(ShallowStateTensor(2, u), ShallowEnvironment(2, v))
        # simulate state
        circuit = get_circuit(state)
        simulator = cirq.Simulator()
        results = simulator.simulate(circuit)

        qubits = cirq_qubits(state.num_qubits())
        qb0 = results.bloch_vector_of(qubits[0])
        qb1 = results.bloch_vector_of(qubits[1])
        qb2 = results.bloch_vector_of(qubits[2])

        qubit_0.append(qb0)
        qubit_1.append(qb1)
        qubit_2.append(qb2)
        current_step += 1
        print(current_step)

    return qubit_0, qubit_1, qubit_2


if __name__ == '__main__':
    qubit0, qubit1, qubit2 = main()

    # plt.plot(range(len(qubit_1)), qubit_1)
    # plt.plot(range(len(qubit2)), qubit2)
    # plt.show()
