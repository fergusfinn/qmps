import cirq
from .represent import State, FullStateTensor, FullEnvironment, get_env
from .represent import get_env_exact, full_tomography_env_objective_function
from .tools import environment_from_unitary, Optimizer, to_real_vector, from_real_vector
from numpy import array, real, kron, eye
from numpy.linalg import qr
from numpy.random import randn

from xmps.spin import N_body_spins, U4

from scipy.optimize import approx_fprime

from typing import Callable, List, Dict

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
        self.get_env = get_env_function
        if D!=2:
            raise NotImplementedError('D>2 not implemented')
        self.H = H
        self.D = D
        self.d = 2
        initial_guess = (randn(15) if initial_guess is None else initial_guess)
        u_original = FullStateTensor(U4(initial_guess))
        v_original = None

        super().__init__(u_original, v_original,
                         initial_guess=initial_guess, settings=None)

    def objective_function(self, u_params):
        U = U4(u_params)
        V = self.get_env(U)
        assert abs(full_tomography_env_objective_function(FullStateTensor(U), FullEnvironment(V)))<1e-6

        qbs = cirq.LineQubit.range(4)
        sim = cirq.Simulator()

        C =  cirq.Circuit().from_ops(State(FullStateTensor(U), FullEnvironment(V), 2)(*qbs))
        H = kron(kron(eye(2), self.H), eye(2))

        ψ = sim.simulate(C).final_state

        f =  real(ψ.conj().T@H@ψ)
        return f

    def update_final_circuits(self):
        self.U = U4(self.optimized_result.x)

class SparseFullEnergyOptimizer(Optimizer):
    def __init__(self, 
                 H, 
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
        initial_guess = (randn(15) if initial_guess is None else initial_guess)
        u_original = FullStateTensor(U4(initial_guess))
        v_original = None

        super().__init__(u_original, v_original,
                         initial_guess=initial_guess, settings=None)


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
