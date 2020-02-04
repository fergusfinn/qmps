from numpy import array, real, kron, eye, trace, zeros
from numpy.linalg import qr
from numpy.random import randn
import numpy as np
import cirq

from .represent import State, FullStateTensor, FullEnvironment, get_env
from .represent import full_tomography_env_objective_function
from .represent import ShallowCNOTStateTensor, ShallowCNOTStateTensor3, ShallowEnvironment
from .represent import ShallowQAOAStateTensor, ShallowFullStateTensor

from .tools import environment_from_unitary, Optimizer, to_real_vector, from_real_vector
from .tools import get_env_exact, split_2s

from scipy.optimize import approx_fprime

from typing import Callable, List, Dict
from functools import reduce
from itertools import product

from xmps.spin import spins, swap, paulis

π = np.pi

Sx, Sy, Sz = paulis(0.5)
S = {'I': eye(2), 'X': Sx, 'Y': Sy, 'Z': Sz}

class PauliMeasure(cirq.Gate):
    """PauliMeasure:apply appropriate transformation to 2 qubits
       st. measuring qb[0] in z basis measures string"""
    def __init__(self, string):
        if string=='II':
            raise Exception('don\'t measure the identity')
        assert len(string)==2
        self.string = string

    def _decompose_(self, qubits):

        def single_qubits(string, qubit):
            assert len(string)==1
            if string =='X':
                yield cirq.H(qubit)
            elif string == 'Y':
                yield cirq.inverse(cirq.S(qubit))
                yield cirq.H(qubit)

        i, j = self.string
        if i=='I':
            yield cirq.SWAP(qubits[0], qubits[1])
            j, i = i, j
        yield single_qubits(i, qubits[0])
        yield single_qubits(j, qubits[1])
        if i!='I' and j!='I':
            yield cirq.CNOT(qubits[1], qubits[0])

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return list(self.string)

class Hamiltonian:
    """Hamiltonian: string of terms in local hamiltonian.
       Just do quadratic spin 1/2
       ex. tfim = Hamiltonian({'ZZ': 1, 'X': λ}) = Hamiltonian({'ZZ': 1, 'IX': λ/2, 'XI': λ/2})
       for parity invariant specify can single site terms ('X')
       otherwise 'IX' 'YI' etc."""

    def __init__(self, strings=None):
        self.strings = strings
        if strings is not None:
            for key, val in {key:val for key, val in self.strings.items()}.items():
                if len(key)==1:
                    self.strings['I'+key] = val/2
                    self.strings[key+'I'] = val/2
                    self.strings.pop(key)

    def to_matrix(self):
        assert self.strings is not None
        h_i = zeros((4, 4))+0j
        for js, J in self.strings.items():
            h_i += J*reduce(kron, [S[j] for j in js])
        self._matrix = h_i
        return h_i

    def from_matrix(self, mat):
        xyz = list(S.keys())
        strings = list(product(xyz, xyz))
        self.strings = {a+b:trace(kron(a, b)@mat) for a, b in strings}
        del self.strings['II']
        return self

    def measure_energy(self, circuit, qubits, reps=300000):
        assert self.strings is not None
        ev = 0
        for string, g in self.strings.items():
            c = circuit.copy()
            c.append(PauliMeasure(string)(*qubits))
            c.append(cirq.measure(qubits[0], key=string))

            sim = cirq.Simulator()
            meas = sim.run(c, repetitions=reps).measurements[string]
            ev += g*array(list(map(lambda x: 1-2*int(x), meas))).mean()
        return ev

    def calculate_energy(self, circuit, loc=0):
        c = circuit.copy()
        sim = cirq.Simulator()
        ψ = sim.simulate(c).final_state
        H = self.to_matrix()

        I = eye(2)
        H = reduce(kron, [I]*loc+[H]+[I]*(len(c.all_qubits())-loc-2))
        return real(ψ.conj().T@H@ψ)

class SparseFullEnergyOptimizer(Optimizer):
    def __init__(self,
                 H,
                 D=2,
                 depth=2,
                 state_tensor=ShallowCNOTStateTensor,
                 optimize_environment=False,
                 env_depth=4,
                 initial_guess = None,
                 settings: Dict = None):
        self.optimize_environment = optimize_environment
        self.env_depth = env_depth
        self.state_tensor=state_tensor
        self.H = H
        self.D = D
        self.d = 2
        if self.optimize_environment:
            initial_guess = randn(30) if initial_guess is None else initial_guess
            self.objective_function = self.objective_function_opt_environment
        else:
            initial_guess = array([randn(), randn()]*depth) if initial_guess is None else initial_guess
            self.objective_function = self.objective_function_exact_environment
        self.p = len(initial_guess)
        u_original = self.state_tensor(D, initial_guess)
        v_original = None

        super().__init__(u_original, v_original, initial_guess)


    def objective_function_exact_environment(self, u_params):
        U = self.state_tensor(self.D, u_params)
        #V = self.env_optimizer(U, self.env_depth).get_env().v
        try:
            V = FullEnvironment(get_env_exact(cirq.unitary(U))) # for testing
        except np.linalg.LinAlgError:
            print('LinAlgError')
            return self.f

        qbs = cirq.LineQubit.range(2+V.num_qubits())
        sim = cirq.Simulator()

        C =  cirq.Circuit().from_ops(State(U, V, 2)(*qbs))
        H = kron(kron(eye(self.D), self.H), eye(self.D))

        ψ = sim.simulate(C).final_state

        self.f =  real(ψ.conj().T@H@ψ)
        return self.f

    def objective_function_opt_environment(self, params):
        def op_H(H):
            #H = np.eye(4)
            return reduce(np.kron, [np.eye(2), H, np.eye(2)])
        H = op_H(self.H)
        assert len(params)==30
        def gate(v):
            return ShallowFullStateTensor(2, v)

        def state(params, which='energy'):
            p2, p1 = np.split(params, 2)
            if which=='energy':
                qbs = cirq.LineQubit.range(4)
                C = cirq.Circuit.from_ops([gate(p1)(*qbs[2:]),
                                           gate(p2)(*qbs[1:3]),
                                           gate(p2)(*qbs[:2])])
                s = cirq.Simulator()
                return s.simulate(C).final_state
            elif which=='v_purity':
                qbs = [[cirq.GridQubit(y, x) for x in range(2)]
                        for y in range(2)]
                C = cirq.Circuit.from_ops([gate(p1)(*qbs[0][:2]),
                                           gate(p1)(*qbs[1][:2]),
                                           cirq.SWAP(*qbs[0][:2])])
                s = cirq.Simulator()
                return s.simulate(C).final_state
            elif which=='u_purity':
                qbs = [[cirq.GridQubit(y, x) for x in range(3)]
                        for y in range(2)]
                C = cirq.Circuit.from_ops([gate(p1)(*qbs[0][1:]),
                                           gate(p2)(*qbs[0][:2]),
                                           gate(p1)(*qbs[1][1:]),
                                           gate(p2)(*qbs[1][:2]),
                                           cirq.SWAP(*qbs[0][:2]),
                                           cirq.SWAP(*qbs[0][1:])])
                s = cirq.Simulator()
                return s.simulate(C).final_state
            elif which=='uv_purity':
                qbs = cirq.LineQubit.range(5)
                C = cirq.Circuit.from_ops([gate(p1)(*qbs[3:]),
                                           gate(p2)(*qbs[2:4]),
                                           gate(p1)(*qbs[:2]),
                                           cirq.SWAP(*qbs[:2])])
                s = cirq.Simulator()
                return s.simulate(C).final_state
        def ϵ(x):
            uv_state, u_state, v_state, e_state = (state(x, 'uv_purity'),
                                                   state(x, 'u_purity'),
                                                   state(x, 'v_purity'),
                                                   state(x, 'energy'))
            v_purity = np.real(v_state.conj().T@np.kron(np.eye(2), np.kron(swap(), np.eye(2)))@v_state)
            u_purity = np.real(u_state.conj().T@np.kron(np.eye(4), np.kron(swap(), np.eye(4)))@u_state)
            uv_purity = np.real(uv_state.conj().T@np.kron(np.kron(np.eye(2), swap()), np.eye(4))@uv_state)
            energy = np.real(e_state.conj().T@H@e_state)

            k = 1
            return sum([energy,+k*u_purity,+k*v_purity,-2*k*uv_purity])

        return ϵ(params)

class NonSparseFullEnergyOptimizer(Optimizer):
    """NonSparseFullEnergyOptimizer

    NonSparse: not a low depth variational optimizer
    Full: simulates the full wavefunction i.e. not via sampling"""
    def __init__(self,
                 H,
                 D=2,
                 get_env_function=get_env_exact,
                 initial_guess=None,
                 settings: Dict = None):
        self.env_function = get_env_function
        self.H = H
        self.D = D
        self.d = 2
        initial_guess = (randn((2*D)**2-1) if initial_guess is None else initial_guess)
        u_original = FullStateTensor(SU(initial_guess, 2*D))
        v_original = None

        super().__init__(u_original, v_original, initial_guess)

    def objective_function(self, u_params):
        self.U = U = SU(u_params, 2*self.D)

        V = self.env_function(U)
        #assert abs(full_tomography_env_objective_function(FullStateTensor(U), FullEnvironment(V)))<1e-6

        qbs = cirq.LineQubit.range(int(2+2*log2(self.D)))
        sim = cirq.Simulator()

        C =  cirq.Circuit().from_ops(State(FullStateTensor(U), FullEnvironment(V), 2)(*qbs))
        H = kron(kron(eye(self.D), self.H), eye(self.D))

        ψ = sim.simulate(C).final_state

        f =  real(ψ.conj().T@H@ψ)
        return f

    def update_state(self):
        self.U = SU(self.optimized_result.x, 2*self.D)

class NoisyNonSparseFullEnergyOptimizer(Optimizer):
    """NonSparseFullEnergyOptimizer

    NonSparse: not a low depth variational optimizer
    Full: simulates the full wavefunction i.e. not via sampling"""
    def __init__(self,
                 H,
                 depolarizing_prob,
                 D=2,
                 get_env_function=get_env_exact,
                 initial_guess=None,
                 settings: Dict = None):
        self.get_env = get_env_function
        if D!=2:
            raise NotImplementedError('D>2 not implemented')
        self.H = H
        self.D = D
        self.d = 2
        self.depolarizing_prob = depolarizing_prob
        initial_guess = (randn(15) if initial_guess is None else initial_guess)
        u_original = FullStateTensor(U4(initial_guess))
        v_original = None

        super().__init__(u_original, v_original,
                         initial_guess=initial_guess)

    def objective_function_density_matrix(self, u_params):
        U = U4(u_params)
        V = self.get_env(U)
        assert abs(full_tomography_env_objective_function(FullStateTensor(U), FullEnvironment(V)))<1e-6

        qbs = cirq.LineQubit.range(4)

        C =  cirq.Circuit().from_ops(cirq.decompose(State(FullStateTensor(U), FullEnvironment(V), 2)(*qbs)))

        noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(self.depolarizing_prob))

        system_qubits = sorted(C.all_qubits())
        noisy_circuit = cirq.Circuit()
        for moment in C:
            noisy_circuit.append(noise.noisy_moment(moment, system_qubits))

        sim = cirq.DensityMatrixSimulator(noise=noise)
        ρ = sim.simulate(noisy_circuit).final_density_matrix

        f =  real(trace(ρ@H))
        return f

    def objective_function_monte_carlo(self, u_params):
        U = U4(u_params)
        V = self.get_env(U)
        assert abs(full_tomography_env_objective_function(FullStateTensor(U), FullEnvironment(V)))<1e-6

        qbs = cirq.LineQubit.range(4)

        C =  cirq.Circuit().from_ops(cirq.decompose(State(FullStateTensor(U), FullEnvironment(V), 2)(*qbs)))

        noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(self.depolarizing_prob))

        system_qubits = sorted(C.all_qubits())
        noisy_circuit = cirq.Circuit()
        for moment in C:
            noisy_circuit.append(noise.noisy_moment(moment, system_qubits))

        sim = cirq.Simulator()
        ψ = sim.simulate(noisy_circuit).final_state
        H = kron(kron(eye(2), self.H), eye(2))
        f = real(ψ.conj().T@H@ψ)

        #sim = cirq.DensityMatrixSimulator(noise=noise)
        #ρ = sim.simulate(noisy_circuit).final_density_matrix

        #f =  real(trace(ρ@H))
        return f

    def objective_function(self, u_params):
        return self.objective_function_monte_carlo(u_params)

    def update_state(self):
        self.U = U4(self.optimized_result.x)

class NoisySparseFullEnergyOptimizer(Optimizer):
    """NonSparseFullEnergyOptimizer

    NonSparse: not a low depth variational optimizer
    Full: simulates the full wavefunction i.e. not via sampling"""
    def __init__(self,
                 H,
                 depolarizing_prob,
                 D=2,
                 depth=2,
                 env_optimizer=None,
                 env_depth=4,
                 state_tensor=ShallowCNOTStateTensor,
                 initial_guess = None,
                 settings: Dict = None):
        self.env_optimizer = env_optimizer
        self.env_depth = env_depth
        self.state_tensor=state_tensor
        self.H = H
        self.D = D
        self.d = 2
        initial_guess = array([randn(), randn()]*depth) if initial_guess is None else initial_guess
        self.p = len(initial_guess)
        u_original = self.state_tensor(D, initial_guess)
        v_original = None

        self.depolarizing_prob = depolarizing_prob

        super().__init__(u_original, v_original, initial_guess)

    def objective_function_density_matrix(self, u_params):
        pass

    def objective_function_monte_carlo(self, u_params):
        U = self.state_tensor(self.D, u_params)
        V = FullEnvironment(get_env_exact(cirq.unitary(U))) # for testing

        qbs = cirq.LineQubit.range(int(2*np.log2(self.D)+2))

        C =  cirq.Circuit().from_ops(cirq.decompose(State(U, V, 2)(*qbs)))

        noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(self.depolarizing_prob))

        system_qubits = sorted(C.all_qubits())
        noisy_circuit = cirq.Circuit()
        for moment in C:
            noisy_circuit.append(noise.noisy_moment(moment, system_qubits))

        sim = cirq.Simulator()
        ψ = sim.simulate(noisy_circuit).final_state
        H = kron(kron(eye(self.D), self.H), eye(self.D))
        f = real(ψ.conj().T@H@ψ)

        #sim = cirq.DensityMatrixSimulator(noise=noise)
        #ρ = sim.simulate(noisy_circuit).final_density_matrix

        #f =  real(trace(ρ@H))
        return f

    def objective_function(self, u_params):
        return self.objective_function_monte_carlo(u_params)

class NoisySparseSampledEnergyOptimizer(Optimizer):
    def __init__(self, H: Hamiltonian,
                 n_state_params=2,
                 n_env_params=2,
                 n_samples=1000,
                 initial_guess=None,
                 optimize_env=False,
                 state_ansatz=ShallowCNOTStateTensor,
                 env_ansatz=ShallowCNOTStateTensor):
        self.n_state_params = n_state_params
        self.n_env_params = n_env_params
        self.n_samples = n_samples
        self.initial_guess = initial_guess if initial_guess is not None else randn(n_state_params+n_env_params)
        self.optimize_env = optimize_env
        self.state_ansatz = state_ansatz
        self.env_ansatz = env_ansatz
        self.u_params, self.v_params = (initial_guess[:self.n_state_params],
                                        initial_guess[self.n_state_params:])

        u_original = self.state_ansatz(D, self.u_params)
        v_original = self.environment_ansatz(D, self.u_params)
        super().__init__(u_original, v_original, initial_guess)

    def objective_function(params):
        self.u_params, self.v_params = (params[:self.n_state_params],
                                        params[self.n_state_params:])

