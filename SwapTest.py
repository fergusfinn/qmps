import numpy as np
import cirq
from scipy.stats import unitary_group
from unitary_iMPS import log2, split_2s
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from xmps.iMPS import iMPS
import json
import io
from io import StringIO
import webbrowser


def rprint(value):
    print(np.around(value, 2))


def cirq_qubits(num):
    return cirq.LineQubit.range(num)


def get_circuit(state, decomp=None):
    if decomp == 'Full':
        return cirq.Circuit.from_ops(cirq.decompose(state(*cirq_qubits(state.num_qubits()))))
    elif decomp == 'Once':
        return cirq.Circuit.from_ops(cirq.decompose_once(state(*cirq_qubits(state.num_qubits()))))
    else:
        return cirq.Circuit.from_ops(state(*cirq_qubits(state.num_qubits())))


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


class FullStateTensor(Tensor):
    """StateTensor: represent state tensor as a unitary"""

    def __init__(self, unitary, symbol='U'):
        super().__init__(unitary, symbol)

    def raise_power(self, power):
        return PowerCircuit(state=self, power=power)


class FullEnvironment(Tensor):
    """Environment: represents the environment tensor as a unitary"""

    def __init__(self, unitary, symbol='V'):
        super().__init__(unitary, symbol)


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


class SingleQubitState(cirq.Gate):
    """State: takes a StateTensor gate and an Environment gate"""

    def __init__(self, U, V):
        d, D = self.d, self.D = 2, 2 ** (U.num_qubits() - 1)
        assert U.num_qubits() == int(log2(D)) + 1
        assert V.num_qubits() == 2 * int(log2(D))
        self.n_u_qubits = int(log2(D)) + 1
        self.n_v_qubits = 2 * int(log2(D))
        self.n_qubits = 2 * int(log2(D)) + 1
        self.U = U
        self.V = V

    def _decompose_(self, qubits):
        v_qubits = int(2 * log2(self.D))
        u_qubits = int(log2(self.D)) + 1
        return self.V(*qubits[-v_qubits:]), self.U(*qubits[:u_qubits])

    def num_qubits(self):
        return self.n_qubits


class SwapTestState(cirq.Gate):
    def __init__(self, state: SingleQubitState, v: ShallowEnvironment):
        self.d = 2
        self.D = 2 ** (state.n_u_qubits - 1)
        self.n_v_dag_qubits = 2 * int(log2(self.D))
        self.state = state
        self.env_dag = cirq.inverse(v)
        self.n_qubits = 2 * int(log2(self.D)) + 1

    def _decompose_(self, qubits):
        half_qubits = int(self.n_v_dag_qubits/2)
        return (self.state(*qubits),
                self.env_dag(*(qubits[:half_qubits] + qubits[-half_qubits:])))

    def num_qubits(self):
        return self.n_qubits


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


class PowerSwapState(cirq.Gate):
    def __init__(self, power_circuit: PowerCircuit, state):
        self.powerCircuit = power_circuit
        self.state = state
        self.n_qubits = max(state.num_qubits(), power_circuit.num_qubits())
        self.inverse_state = cirq.inverse(state)

    def _decompose_(self, qubits):
        power_qubits = self.powerCircuit.num_qubits()
        state_qubits = self.state.num_qubits()
        return self.powerCircuit(*qubits[:power_qubits]), self.inverse_state(*qubits[:state_qubits])

    def num_qubits(self):
        return self.n_qubits


class Optimizer:
    '''
    Base class for optimizers. To specify a new optimization technique simply define a new objective function
    '''
    def __init__(self, u_params, qaoa_depth, bond_dim):
        self.u_params = u_params
        self.v_params = np.random.randn(qaoa_depth*2)
        self.bond_dim = bond_dim
        self.iters = 0
        self.reps = 10000
        self.obj_fun_values = []
        self.noisy = False
        self.store_values = False
        self.optimized_result = None
        self.circuit = None

    def set_noise(self, _bool):
        self.noisy = _bool
        self.store_values = _bool

    def set_u_params(self, params):
        self.u_params = params

    def set_reps(self, reps):
        self.reps = reps

    def objective_function(self, v_params):
        pass

    def callback_store_values(self, xk):
        val = self.objective_function(xk)
        self.obj_fun_values.append(val)
        if self.noisy:
            print(f'{self.iters}:{val}')
        self.iters += 1

    def get_env(self, max_iter):
        options = {'maxiter': max_iter,
                   'disp': self.noisy}  # if noisy else False}

        kwargs = {'fun': self.objective_function,
                  'x0': self.v_params,
                  'method': 'Nelder-Mead',
                  'tol': 1e-5,
                  'options': options,
                  'callback': self.callback_store_values if self.store_values else None}

        self.optimized_result = minimize(**kwargs)
        if self.noisy:
            print(f'Reason for termination is {self.optimized_result.message}')

    def plot_convergence(self, file):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.figure()
        x = list(range(len(self.obj_fun_values)))
        plt.plot(x, self.obj_fun_values)
        plt.xlabel('iterations')
        plt.ylabel('objective function value')
        plt.savefig(dir_path + '/' + file)


class SampleSwapOptimizer(Optimizer):
    '''
    'Vertical' swap test optimizer
    '''

    def objective_function(self, v_params):
        bond_dim = self.bond_dim
        u_params = self.u_params
        reps = self.reps

        u = ShallowStateTensor(bond_dim=bond_dim, βγs=u_params)  # parameterized circuit U
        v = ShallowEnvironment(bond_dim=bond_dim, βγs=v_params)
        swap_state = SwapTestState(state=SingleQubitState(u, v), v=v)

        n_qubits = swap_state.num_qubits()
        qbs = cirq.LineQubit.range(n_qubits)
        measure_qubits = int((n_qubits - 1) / 2)

        circuit = cirq.Circuit.from_ops([swap_state(*qbs), cirq.measure(*qbs[:measure_qubits])])
        self.circuit = circuit
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=reps)

        key = list(result.measurements.keys())[0]
        counter = result.histogram(key=key, fold_func=lambda e: sum(e))  # sum up the number of 1s in the measurements
        mean = sum(counter.elements()) / reps
        return mean


class FullSwapOptimizer(Optimizer):
    @staticmethod
    def get_target_qubits(num_qubits, bond_dim):
        target_qbs = log2(bond_dim)
        other_qbs = num_qubits - target_qbs
        return 2**other_qbs - 1

    def objective_function(self, v_params):
        bond_dim = self.bond_dim
        u_params = self.u_params

        u = ShallowStateTensor(bond_dim=bond_dim, βγs=u_params)  # parameterized circuit U
        v = ShallowEnvironment(bond_dim=bond_dim, βγs=v_params)
        swap_state = SwapTestState(state=SingleQubitState(u, v), v=v)

        n_qubits = swap_state.num_qubits()
        qbs = cirq.LineQubit.range(n_qubits)

        circuit = cirq.Circuit.from_ops([swap_state(*qbs)])
        self.circuit = circuit

        simulator = cirq.Simulator()
        results = simulator.simulate(circuit)

        target_qubits = self.get_target_qubits(n_qubits, bond_dim)
        prob_zeros = sum(np.absolute(results.final_simulator_state.state_vector[:int(target_qubits)])**2)
        return 1-prob_zeros


def get_isometry(u):
    return np.tensordot(u.reshape(*2 * int(np.log2(u.shape[0])) * [2]),
                        np.array([1, 0]),
                        [2, 0]).transpose([1, 0, 2])


def get_analytic_environment(u_params, bond_dim):
    state = ShallowStateTensor(bond_dim=bond_dim, βγs=u_params)
    state_unitary = cirq.unitary(state)
    state_isometry = get_isometry(state_unitary)

    # why doesnt the below code work
    # state_isometry = state_unitary[:, :2].reshape(2, bond_dim, bond_dim)

    mps = iMPS(data=[state_isometry])
    _, __, c = mps.mixed()
    return c


def test_v_swap(bond_dim, qaoa_depth, noisy=True):
    u_params = np.random.randn(2*qaoa_depth)  # p (QAOA depth) * 2 (to get both gamma and beta)
    c = get_analytic_environment(u_params, bond_dim)

    optimizer = SampleSwapOptimizer(u_params, qaoa_depth, bond_dim)
    if noisy:
        optimizer.set_noise(True)  # get it to print out values

    optimizer.get_env(max_iter=100)

    if noisy:
        file_name = f'qaoa_depth{qaoa_depth}_bond_dim{bond_dim}_vertical_swap3.png'
        optimizer.plot_convergence(file=file_name)

    optimized_params = optimizer.optimized_result.x
    print(f'minimum objective function value:{min(optimizer.obj_fun_values)}')

    optimized_environment = cirq.unitary(ShallowEnvironment(bond_dim=bond_dim, βγs=optimized_params))
    return optimized_environment, optimizer, c


def draw_power_state_swap():
    '''
    :return: print the circuit to compare a power circuit & a state
    '''
    U = unitary_group.rvs(4)
    V = unitary_group.rvs(4)

    u_tens = FullStateTensor(U)
    v_tens = FullEnvironment(V)

    power_circuit = u_tens.raise_power(5)

    state = SingleQubitState(u_tens, v_tens)

    power_swap = PowerSwapState(power_circuit, state)
    print(get_circuit(state=power_swap, decomp='Full'))


def draw_power_environment():
    '''
    :return: print the circuit for both the state and environment being a power circuit
    '''
    u = unitary_group.rvs(4)
    u_tens = FullStateTensor(u)
    power_circuit = u_tens.raise_power(5)
    power_environment = u_tens.raise_power(4)

    power_swap = PowerSwapState(power_circuit, power_environment)
    print(get_circuit(state=power_swap, decomp='Full'))


def power_environment_correct_prob(unitary, power):
    u_tens = FullStateTensor(unitary)
    power_circuit = u_tens.raise_power(power)
    power_environment = u_tens.raise_power(power-1)
    power_swap = PowerSwapState(power_circuit, power_environment)

    circuit = get_circuit(power_swap, decomp='Full')
    simulator = cirq.Simulator()

    qbs = cirq_qubits(power_swap.num_qubits())
    results = simulator.simulate(circuit)
    bloch_vector = results.bloch_vector_of(qbs[0])
    return bloch_vector[2]


def test_environment_convergence(runs, show=True, save=False):
    unitary = unitary_group.rvs(4)
    probs = []

    powers = range(2, runs+2)
    for run in powers:
        result = power_environment_correct_prob(unitary, run)
        probs.append(result)

    results_dic = dict(zip(map(str, powers), map(str, probs)))
    plt.figure()
    plt.plot(powers, probs)
    if save:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.savefig(dir_path + '/' + save + '.png')
        with open(str(save)+'.txt', 'w') as json_file:
            json.dump(results_dic, json_file)

    if show:
        plt.show()

    return results_dic


def full_swap_optimizer():
    u_params = np.random.rand(2)
    optimizer = FullSwapOptimizer(u_params, bond_dim=2, qaoa_depth=1)
    optimizer.set_noise(True)
    optimizer.get_env(max_iter=100)
    optimizer.plot_convergence('full_wavefunc_convergence.png')


class HorizontalSwapTest(cirq.Gate):
    def __init__(self, u_params, v_params, bond_dim):
        self.U = ShallowStateTensor(bond_dim, u_params)
        self.V = ShallowEnvironment(bond_dim, v_params)
        self.state = SingleQubitState(self.U, self.V)
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


def save_circuit_diagram(circuit, file):
    with io.open(file, "w", encoding="utf-8") as f:
        f.write(circuit.__str__())


def save_to_latex(circuit, file):
    latex_circuit = cirq.contrib.qcircuit.qcircuit_diagram.circuit_to_latex_using_qcircuit(circuit)
    with open(file, "w") as f:
        f.write(latex_circuit)


def text_print(circuit):
    file = "printed_circuit.txt"
    with io.open(file, "w", encoding="utf-8") as f:
        f.write(circuit.to_text_diagram(transpose=True))
    command_string = f"notepad.exe {file}"
    os.system(command_string)
    os.remove(file)


def tprint(circuit):
    print(circuit.to_text_diagram(transpose=True))


def main():
    U,V = FullStateTensor(unitary_group.rvs(4)), FullEnvironment(unitary_group.rvs(4))
    V_inverse = V**-1
    U_inverse = U**-1
    H = Tensor(unitary_group.rvs(2), 'H')
    qubits = cirq_qubits(U.num_qubits()+1)
    circuit = cirq.Circuit.from_ops(V(*qubits[1:3]),
                                    U(*qubits[0:2]),
                                    H(qubits[1]),
                                    U_inverse(*qubits[0:2]),
                                    V_inverse(*qubits[1:3]),
                                    cirq.measure(qubits[0]),
                                    cirq.measure(qubits[1]),
                                    cirq.measure(qubits[2])
                                    )
    save_to_latex(circuit, 'time_evolution.txt')


if __name__ == '__main__':
    main()



