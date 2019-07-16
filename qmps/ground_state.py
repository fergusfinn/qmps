import cirq
from .represent import State, FullStateTensor, FullEnvironment, get_env
from .tools import environment_from_unitary
from numpy import array, real
from numpy.random import randn

from xmps.spin import N_body_spins, U4

from scipy.optimize import approx_fprime

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
        return -J*zz+λ*x

    def full_energy(U, V, λ):
        qbs = cirq.LineQubit.range(4)
        sim = cirq.Simulator()

        C =  cirq.Circuit().from_ops(State(FullStateTensor(U), FullEnvironment(V), 2)(*qbs))
        IZZI = 4*N_body_spins(0.5, 2, 4)[2]@N_body_spins(0.5, 3, 4)[2]
        IIXI = 2*N_body_spins(0.5, 3, 4)[0]
        IXII = 2*N_body_spins(0.5, 2, 4)[0]
        ψ = sim.simulate(C).final_state
        return real(ψ.conj().T@(-J*IZZI+λ*(IXII+IIXI)/2)@ψ)

    def optimize_energy(N=400, env_update=100, ϵ=1e-1, e_fun = sampled_energy if sample else full_energy):
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
