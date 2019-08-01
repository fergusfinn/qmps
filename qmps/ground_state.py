import cirq
from .represent import State, FullStateTensor, FullEnvironment, get_env
from .represent import get_env_exact, full_tomography_env_objective_function
from .represent import HorizontalSwapOptimizer, ShallowStateTensor, ShallowEnvironment
from .tools import environment_from_unitary, Optimizer, to_real_vector, from_real_vector
from .tools import split_2s
from numpy import array, real, kron, eye, trace, zeros
from numpy.linalg import qr
from numpy.random import randn
from numpy import log2, trace

from xmps.spin import N_body_spins, U4, spins

from scipy.optimize import approx_fprime

from typing import Callable, List, Dict
from functools import reduce
from itertools import product

Sx, Sy, Sz = spins(0.5)

Sx, Sy, Sz = 2*Sx, 2*Sy, 2*Sz
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
        if D!=2:
            raise NotImplementedError('D>2 not implemented')
        self.H = H
        self.D = D
        self.d = 2
        initial_guess = (randn(15) if initial_guess is None else initial_guess)
        u_original = FullStateTensor(U4(initial_guess))
        v_original = None

        super().__init__(u_original, v_original, initial_guess)

    def objective_function(self, u_params):
        U = U4(u_params)
        V = self.env_function(U)
        assert abs(full_tomography_env_objective_function(FullStateTensor(U), FullEnvironment(V)))<1e-6

        qbs = cirq.LineQubit.range(4)
        sim = cirq.Simulator()

        C =  cirq.Circuit().from_ops(State(FullStateTensor(U), FullEnvironment(V), 2)(*qbs))
        H = kron(kron(eye(self.D), self.H), eye(self.D))

        ψ = sim.simulate(C).final_state

        f =  real(ψ.conj().T@H@ψ)
        return f

    def update_state(self):
        self.U = U4(self.optimized_result.x)

class SparseFullEnergyOptimizer(Optimizer):
    def __init__(self, 
                 H, 
                 D=2, 
                 env_optimizer=HorizontalSwapOptimizer,
                 env_depth=2,
                 depth=3,
                 initial_guess = None, 
                 settings: Dict = None):
        self.env_optimizer = env_optimizer
        self.env_depth = env_depth
        self.H = H
        self.D = D
        self.d = 2
        initial_guess = array([randn(), randn()]*depth) if initial_guess is None else initial_guess
        self.p = len(initial_guess)
        u_original = ShallowStateTensor(D, initial_guess)
        v_original = None

        super().__init__(u_original, v_original, initial_guess)

    def objective_function(self, u_params):
        U = ShallowStateTensor(self.D, u_params)
        #V = self.env_optimizer(U, self.env_depth).get_env().v
        V = FullEnvironment(get_env_exact(cirq.unitary(U))) # for testing

        qbs = cirq.LineQubit.range(2+V.num_qubits())
        sim = cirq.Simulator()

        C =  cirq.Circuit().from_ops(State(U, V, 2)(*qbs))
        H = kron(kron(eye(self.D), self.H), eye(self.D))

        ψ = sim.simulate(C).final_state

        f =  real(ψ.conj().T@H@ψ)
        return f

class NoisyNonSparseFullEnergyOptimizer(Optimizer):
    """NonSparseFullEnergyOptimizer

    NonSparse: not a low depth variational optimizer
    Full: simulates the full wavefunction i.e. not via sampling"""
    def __init__(self, 
                 H, 
                 D=2, 
                 depolarizing_prob=0.2,
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

    def objective_function(self, u_params):
        U = U4(u_params)
        V = self.get_env(U)
        assert abs(full_tomography_env_objective_function(FullStateTensor(U), FullEnvironment(V)))<1e-6

        qbs = cirq.LineQubit.range(4)

        C =  cirq.Circuit().from_ops(State(FullStateTensor(U), FullEnvironment(V), 2)(*qbs))

        noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(self.depolarizing_prob))

        system_qubits = sorted(C.all_qubits())
        noisy_circuit = cirq.Circuit()
        for moment in C:
            noisy_circuit.append(noise.noisy_moment(moment, system_qubits))
        H = kron(kron(eye(2), self.H), eye(2))

        sim = cirq.DensityMatrixSimulator(noise=noise)
        ρ = sim.simulate(noisy_circuit).final_state

        f =  real(trace(ρ@H))
        return f

    def update_state(self):
        self.U = U4(self.optimized_result.x)

def optimize_ising_D_2(J, λ, sample=False, reps=10000, testing=False):
    """optimize H = -J*ZZ+gX
    """
    def sampled_energy(U, V, J, λ, reps=reps):
        qbs = cirq.LineQubit.range(4)
        sim = cirq.Simulator()

        # create the circuit for 2 local measurements
        C =  cirq.Circuit().from_ops(State(FullStateTensor(U), FullEnvironment(V), 2))

        C_ = C.copy()
        # measure ZZ
        C_.append([cirq.CNOT(qbs[2], qbs[1]), cirq.measure(qbs[1], key='zz')]) 
        meas = sim.run(C_, repetitions=reps).measurements['zz']
        zz = array(list(map(lambda x: 1-2*int(x), meas))).mean()

        C_ = C.copy()
        # measure X
        C_.append([cirq.H(qbs[2]), cirq.measure(qbs[2], key='x')])
        meas = sim.run(C_, repetitions=reps).measurements['x']
        x = array(list(map(lambda x: 1-2*int(x), meas))).mean()
        return J*zz+λ*x

    def full_energy(U, V, λ):
        qbs = cirq.LineQubit.range(4)
        sim = cirq.Simulator()

        C =  cirq.Circuit().from_ops(State(FullStateTensor(U), FullEnvironment(V), 2)(*qbs))
        IZZI = 4*N_body_spins(0.5, 2, 4)[2]@N_body_spins(0.5, 3, 4)[2]
        IIXI = 2*N_body_spins(0.5, 3, 4)[0]
        IXII = 2*N_body_spins(0.5, 2, 4)[0]
        ψ = sim.simulate(C).final_state
        return real(ψ.conj().T@(J*IZZI+λ*(IXII+IIXI)/2)@ψ)

    def optimize_energy(N=400, env_update=1, ϵ=1e-1, e_fun = sampled_energy if sample else full_energy):
        """minimizes ising energy in a full parametrisation of SU(4)
           u at each time step is a 15d real vector, with U4(u) = exp(-iu⋅σ), 
           σ the vector of generators of SU(4).

        :param N: how many steps to take
        :param env_update: update the environment every env_update steps
        :param ϵ: time step (learning rate)
        :param e_fun: whether to use the sampled or full energy function
        """
        def f(u, V, λ): return full_energy(U4(u), V, λ)
        
        u = randn(15) # initial value of u
        V = get_env(U4(u)) # get initial value of V

        for n in range(N):
            du = ϵ*approx_fprime(u, f, ϵ, V, 1)
            u -= du
            if not n%env_update:
                print('\nupdating environment\n')
                V = get_env(U4(u), environment_from_unitary(V))
            print(f(u, V, λ))

        U = U4(u)
        return U, V

    U, V = optimize_energy()
    return U, V
