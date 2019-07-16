import unittest

from xmps.tensor import unitary_extension, embed, H as cT, C as c, T, haar_unitary
from xmps.tensor import deembed, eye_like

from numpy import transpose, prod, array, sum, sqrt, mean, real, imag, concatenate
from numpy import array, allclose, kron, tensordot, trace as tr, eye
from numpy.random import randn
import numpy as np

from math import log as mlog
def log2(x): return mlog(x, 2)

from cirq import TwoQubitMatrixGate, LineQubit, H, S, measure, inverse as inv, Circuit
from cirq import CSWAP, X 
from cirq import Simulator 
import cirq 

from scipy.linalg import norm
from scipy.optimize import minimize
from scipy.optimize import approx_fprime

from xmps.spin import U4 # 15d real parametrisation of SU(2)
from xmps.spin import N_body_spins
from scipy.linalg import null_space

#########################################################################
#########################################################################

# Notebook code

#########################################################################
#########################################################################

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

def left_orthogonal_tensor_to_unitary(A):
    """given a left isometric tensor A, put into a unitary
    """
    d, D, _ = A.shape
    iso = A.transpose([1, 0, 2]).reshape(D*d, D)
    assert allclose(cT(iso)@iso, eye(2)) # left isometry
    U = unitary_extension(iso)
    assert allclose(U@cT(U), eye(4)) # unitary
    assert allclose(cT(U)@U, eye(4)) # unitary
    assert allclose(U[:iso.shape[0], :iso.shape[1]], iso) # with the isometry in it
    assert allclose(tensordot(U.reshape(2, 2, 2, 2), array([1, 0]), [2, 0]).reshape(4, 2), 
                    iso)

    #  ↑ j
    #  | |
    #  ---       
    #   u  = i--A--j
    #  ---      |
    #  | |      σ
    #  i σ 

    return U

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
    U, V = U._unitary_(), V._unitary_()
    qbs = cirq.LineQubit.range(3)
    r = 0

    LHS, RHS = Circuit(), Circuit()
    LHS.append([State2(U, V)(*qbs)])
    RHS.append([Environment2(V)(*qbs[:2])])

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
    U, V = U._unitary_(), V._unitary_()
    qbs = cirq.LineQubit.range(3)
    r = 0

    LHS, RHS = Circuit(), Circuit()
    LHS.append([State2(U, V)(*qbs)])
    RHS.append([Environment2(V)(*qbs[:2])])

    sim = Simulator()
    LHS = sim.simulate(LHS).bloch_vector_of(qbs[0])
    RHS = sim.simulate(RHS).bloch_vector_of(qbs[0])
    return norm(LHS-RHS)

#########################################################################
#########################################################################

# D=2 Case

#########################################################################
#########################################################################

class StateTensor2(cirq.TwoQubitGate):
    """Gate taking full two qubit unitary U, representing the mps tensor"""
    def __init__(self, U):
        self.U = U
    def _unitary_(self):
        return self.U
    def _circuit_diagram_info_(self, args):
        return 'U', 'U'

class Environment2(cirq.TwoQubitGate):
    """Environment2: represents the environment tensor as a two qubit unitary"""
    def __init__(self, V):
        self.V = V
    def _unitary_(self):
        return self.V
    def _circuit_diagram_info_(self, args):
        return 'V', 'V'
 
class State2(cirq.ThreeQubitGate):
    """State: combines the StateTensor and Environment into a State"""
    def __init__(self, U, V):
        self.U = U
        self.V = V
    def _decompose_(self, qubits):
        return (Environment2(self.V)(*qubits[1:3]),
                StateTensor2(self.U)(*qubits[:2])) 
    def _circuit_diagram_info_(self, args):
      return 'I-U', 'V-U', 'V-I'

def mat(v):
    '''helper function - put list of elements (real, imaginary) in a square matrix'''
    re, im = np.split(v, 2)
    C = (re+im*1j).reshape(int(sqrt(len(v)/2)), -1)
    return C

def demat(A):
    '''takes a matrix, breaks it down into a real vector'''
    re, im = real(A).reshape(-1), imag(A).reshape(-1)  
    return concatenate([re, im], axis=0)

def to_unitaries_l(AL):
    """given a left isometric tensor AL, put into a unitary

    :param AL: tensor
    """
    Us = []
    for A in AL:
        d, D, _ = A.shape
        iso = A.transpose([1, 0, 2]).reshape(D*d, D)
        assert allclose(cT(iso)@iso, eye(2)) # left isometry
        U = unitary_extension(iso)
        assert allclose(U@cT(U), eye(4)) # unitary
        assert allclose(cT(U)@U, eye(4)) # unitary
        assert allclose(U[:iso.shape[0], :iso.shape[1]], iso) # with the isometry in it
        assert allclose(tensordot(U.reshape(2, 2, 2, 2), array([1, 0]), [2, 0]).reshape(4, 2), 
                        iso)

        #  ↑ j
        #  | |
        #  ---       
        #   u  = i--A--j
        #  ---      |
        #  | |      σ
        #  i σ 
        Us.append(U)

    return Us

def from_unitaries_l(Us):
    As = []
    for U in Us: 
        A = tensordot(U.reshape(*2*int(log2(U.shape[0]))*[2]), array([1, 0]), [2, 0]).transpose([1, 0, 2])
        As.append(A)
    return As

def sampled_bloch_vector_of(qubit, circuit, reps=1000000):
    """sampled_bloch_vector_of: get bloch vector of a 
    specified qubit by sampling. 

    :param qubit: qubit to sample bloch vector of 
    :param circuit: circuit to evaluate before sampling
    :param reps: number of measurements on each qubit
    """
    sim = cirq.Simulator()
    C = circuit.copy()
    C.append([measure(qubit, key='z')])
    meas = sim.run(C, repetitions=reps).measurements['z']
    z = array(list(map(int, meas))).mean()

    C = circuit.copy()
    C.append([inv(S(qubit)), H(qubit), measure(qubit, key='y')])
    meas = sim.run(C, repetitions=reps).measurements['y']
    y = array(list(map(int, meas))).mean()

    C = circuit.copy()
    C.append([H(qubit), measure(qubit, key='x')])
    meas = sim.run(C, repetitions=reps).measurements['x']
    x = array(list(map(int, meas))).mean()

    return -2*array([x, y, z])+1

def full_env_obj_fun(U, V):
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

    LHS, RHS = Circuit(), Circuit()
    LHS.append([State2(U, V)(*qbs)])
    RHS.append([Environment2(V)(*qbs[:2])])

    sim = Simulator()
    LHS = sim.simulate(LHS).bloch_vector_of(qbs[0])
    RHS = sim.simulate(RHS).bloch_vector_of(qbs[0])
    return norm(LHS-RHS)

def sampled_env_obj_fun(U, V, reps=10000):
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
        j | |   j | |  

    """
    qbs = cirq.LineQubit.range(3)
    r = 0

    LHS, RHS = Circuit(), Circuit()
    LHS.append([State2(U, V)(*qbs)])
    RHS.append([Environment2(V)(*qbs[:2])])

    LHS = sampled_bloch_vector_of(qbs[0], LHS, reps)
    RHS = sampled_bloch_vector_of(qbs[0], RHS, reps)
    return norm(LHS-RHS)

def get_env(U, C0=randn(2, 2)+1j*randn(2, 2), sample=False, reps=100000):
    ''' return v satisfying

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
        """f_obj: take an 8d real vector, use mat to turn it into a 4d complex vector, embed to 
           turn it into a unitary, then calculate the objective function.
        """
        r = full_env_obj_fun(U, embed(mat(v)))
        return r

    def s_obj(v, U=U):
        """s_obj: take an 8d real vector, use mat to turn it into a 4d complex vector, embed to 
           turn it into a unitary, then calculate the (sampled) objective function.
        """
        r = sampled_env_obj_fun(U, embed(mat(v)))
        return r

    obj = s_obj if sample else f_obj

    res = minimize(obj, demat(C0), method='Powell')
    return embed(mat(res.x))

def optimize_ising_D_2(J, λ, sample=False, reps=10000, testing=False):
    """optimize H = -J*ZZ+gX
    """
    def sampled_energy(U, V, J, λ, reps=reps):
        qbs = cirq.LineQubit.range(4)
        sim = cirq.Simulator()

        # create the circuit for 2 local measurements
        C =  Circuit().from_ops([Environment2(V)(*qbs[2:4]), StateTensor2(U)(*qbs[1:3]), StateTensor2(U)(*qbs[0:2])])

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

        C = Circuit().from_ops([Environment2(V)(*qbs[2:4]), StateTensor2(U)(*qbs[1:3]), StateTensor2(U)(*qbs[0:2])])
        IZZI = 4*N_body_spins(0.5, 2, 4)[2]@N_body_spins(0.5, 3, 4)[2]
        IIXI = 2*N_body_spins(0.5, 3, 4)[0]
        IXII = 2*N_body_spins(0.5, 2, 4)[0]
        ψ = sim.simulate(C).final_state
        return np.real(ψ.conj().T@(-J*IZZI+λ*(IXII+IIXI)/2)@ψ)

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
                V = get_env(U4(u), deembed(V))
            print(f(u, V, λ))

        U = U4(u)
        return U, V

    U, V = optimize_energy()
    return U, V

#########################################################################
#########################################################################

# General Case

#########################################################################
#########################################################################

class StateTensor(cirq.Gate):
    pass

class Environment(cirq.Gate):
    pass

class FullStateTensor(StateTensor):
    """StateTensor: represent state tensor as a unitary"""
    def __init__(self, U):
        self.U = U
        self.n_qubits = int(log2(U.shape[0]))

    def _unitary_(self):
        return self.U

    def num_qubits(self):
        return int(self.n_qubits)

    def _circuit_diagram_info_(self, args):
        return ['U']*self.n_qubits

class FullEnvironment(Environment):
    """Environment: represents the environment tensor as a unitary"""
    def __init__(self, V):
        self.V = V
        self.n_qubits = int(log2(V.shape[0]))

    def _unitary_(self):
        return self.V

    def num_qubits(self):
        return self.n_qubits

    def _circuit_diagram_info_(self, args):
        return ['V']*self.n_qubits
 
class ShallowStateTensor(StateTensor):
    """ShallowStateTensor: shallow state tensor based on the QAOA circuit"""
    def __init__(self, D, βγs):
        self.βγs = βγs
        self.p = len(βγs)
        self.n_qubits = int(log2(D))+1

    def num_qubits(self):
        return self.n_qubits

    def _decompose_(self, qubits):
        return [[cirq.X(qubit)**β for qubit in qubits]+\
                [cirq.ZZ(qubits[i], qubits[i+1])**γ for i in range(self.n_qubits-1)]
                for β, γ in split_2s(self.βγs)]

class ShallowEnvironment(Environment):
    """ShallowEnvironmentTensor: shallow environment tensor based on the QAOA circuit"""
    def __init__(self, D, βγs):
        self.βγs = βγs
        self.p = len(βγs)
        self.n_qubits = 2*int(log2(D))

    def num_qubits(self):
        return self.n_qubits

    def _decompose_(self, qubits):
        return [[cirq.X(qubit)**β for qubit in qubits]+\
                [cirq.ZZ(qubits[i], qubits[i+1])**γ for i in range(self.n_qubits-1)]
                for β, γ in split_2s(self.βγs)]

class State(cirq.Gate):
    """State: takes a StateTensor gate and an Environment gate"""
    def __init__(self, U: StateTensor, V: Environment):
        d, D = self.d, self.D = 2, 2**(U.num_qubits()-1)
        assert U.num_qubits() == int(log2(D))+1
        assert V.num_qubits() == 2*int(log2(D))
        self.n_u_qubits = int(log2(D))+1
        self.n_v_qubits = 2*int(log2(D))
        self.n_qubits = 2*int(log2(D))+1
        self.U = U
        self.V = V

    def _decompose_(self, qubits):
        v_qubits = int(2*log2(self.D))
        u_qubits = int(log2(self.D))+1
        return (self.V(*qubits[-v_qubits:]),
                self.U(*qubits[:u_qubits])) 

    def num_qubits(self):
        return self.n_qubits

    def _circuit_diagram_info_(self, args):
      d, D = self.d, self.D
      return ('I\n|\nU', *['V\n|\nU']*int(log2(D)), *['V\n|\nI']*int(log2(D)))

def shallow_env_obj_fun(U_params, V_params, n):
    qbs = cirq.LineQubit.range(2*n+1)
    r = 0

    LHS, RHS = Circuit(), Circuit()
    LHS.append([State(ShallowStateTensor(2**n, U_params), 
                ShallowEnvironment(2**n, V_params))(*qbs)])
    RHS.append([ShallowEnvironment(2**n, V_params)(*qbs[:2*n])])

    sim = Simulator()
    LHS = sim.simulate(LHS).bloch_vector_of(qbs[0])
    RHS = sim.simulate(RHS).bloch_vector_of(qbs[0])
    return norm(LHS-RHS)

def shallow_sampled_env_obj_fun(U_params, V_params, n, reps=100000):
    qbs = cirq.LineQubit.range(2*n+1)
    r = 0

    LHS, RHS = Circuit(), Circuit()
    LHS.append([State(ShallowStateTensor(2**n, U_params), 
                ShallowEnvironment(2**n, V_params))(*qbs)])
    RHS.append([ShallowEnvironment(2**n, V_params)(*qbs[:2*n])])

    LHS = sampled_bloch_vector_of(qbs[0], LHS, reps)
    RHS = sampled_bloch_vector_of(qbs[0], RHS, reps)
    return norm(LHS-RHS)

def split_2s(x):
    """split_2s: take a list: [β, γ, β, γ, ...], return [[β, γ], [β, γ], ...]
    """
    return [x[i:i+2] for i in range(len(x)) if not i%2]

def get_shallow_env(U_params, n, cut_off_if_less_than=1e-2, p=2, max_iters=500, ϵ=1e-1, schedule=0.5, noisy=False):
    """get_shallow_env: get the environment with the qaoa parametrisation

    U_params: parameters determining the state
    n: number of qubits
    cut_off_if_less_than: stop optimizing if objective less than this
    p: QAOA depth
    max_iters: maximum number of iterations
    ϵ: initial ϵ
    schedule: if the objective function increases, multiply ϵ by schedule
    noisy: print update info
    """
    if noisy:
        print('getting environment: n={}, p={}'.format(n, p))

    def f(βγ): return shallow_env_obj_fun(U_params, βγ, n)
    βγ = randn(p*2)
    w = schedule
    candidate = βγ
    for t in range(1, max_iters+1):
        x = f(βγ)
        if noisy:
            print(x)
        if x < f(candidate):
            candidate = βγ
        else:
            if noisy: 
                print('decreasing ϵ')
            ϵ = ϵ*w
            βγ = candidate
        if cut_off_if_less_than>x:
            break

        dβγ = ϵ*approx_fprime(βγ, f, ϵ)
        βγ = βγ - dβγ

    return βγ

def optimize_ising(D, J, λ, p=1, max_iters=1000, env_update=10, ϵ=1e-1, w=0.5, sample=False, reps=10000, testing=False):
    """optimize H = -J*ZZ+gX
      | | | | | | | | 
      | | -----------       
      | |      v         
      | | -----------  
      | | | | | | | |           (2)
      | ------- | | |  
      |    u    | | |  
      | ------- | | |  
      | | | | | | | |             
      ------- | | | |
         u    | | | |
      ------- | | | |
      | | | | | | | |
      | | | σ ρ | | | 

        ⏟         ⏟ 
    (log2(D)) (log2(D))
        """
    def sampled_energy(U_βγ, V_βγ, J, λ, reps=reps):
        n = int(log2(D))
        n_qubits = 2*n+2
        qbs = cirq.LineQubit.range(n_qubits)
        sim = cirq.Simulator()

        C = Circuit().from_ops([ShallowEnvironment(D, V_βγ)(*qbs[-2*n:]), 
                                ShallowStateTensor(D, U_βγ)(*qbs[1:-n]), 
                                ShallowStateTensor(D, U_βγ)(*qbs[:n+1])])

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

    def full_energy(U_βγ, V_βγ, J, λ):
        n = int(log2(D))
        n_qubits = 2*n+2
        qbs = cirq.LineQubit.range(n_qubits)
        sim = cirq.Simulator()

        C = Circuit().from_ops([ShallowEnvironment(D, V_βγ)(*qbs[-2*n:]), 
                                ShallowStateTensor(D, U_βγ)(*qbs[1:-n]), 
                                ShallowStateTensor(D, U_βγ)(*qbs[:n+1])])
        IZZI = 4*N_body_spins(0.5, n+1, n_qubits)[2]@N_body_spins(0.5, n+2, n_qubits)[2]
        IIXI = 2*N_body_spins(0.5, n+2, n_qubits)[0]
        IXII = 2*N_body_spins(0.5, n+1, n_qubits)[0]
        ψ = sim.simulate(C).final_state
        return np.real(ψ.conj().T@(-J*IZZI+λ*(IXII+IIXI)/2)@ψ)

    u = randn(2*p)
    print(len(u))
    print('initial_environment_update')
    V_βγ = get_shallow_env(u, int(log2(D)), p=p, noisy=True)
    print('found environment')

    def f(u): return full_energy(u, V_βγ, J, λ)

    candidate = u
    x = f(u)
    for t in range(1, max_iters+1):
        if x < f(candidate):
            candidate = u
        if not t%env_update:
            print('\nupdating environment\n')
            u = candidate
            x = f(u)
            print('Best invalid guess: ', x)
            V_βγ = get_shallow_env(u, int(log2(D)), p=2*p)
            x = f(u)
            print('Best valid guess: ', x)
        else:
            x = f(u)
            print(x)

        du = ϵ*approx_fprime(u, f, ϵ)
        u -= du

    return u, V_βγ
