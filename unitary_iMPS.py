import unittest

from pymps.tensor import unitary_extension, embed, H as cT, C as c, T, haar_unitary
from pymps.tensor import deembed, eye_like

from numpy import transpose, prod, array, sum, sqrt, mean, real, imag, concatenate
from numpy import array, allclose, kron, tensordot, trace as tr, eye
from numpy.random import randn
import numpy as np
from pymps.spin import N_body_spins

from math import log as mlog
def log2(x): return mlog(x, 2)

from cirq import TwoQubitMatrixGate, LineQubit, H, S, measure, inverse as inv, Circuit
from cirq import CSWAP, X
from cirq import Simulator
import cirq 

from scipy.linalg import norm
from scipy.optimize import minimize
from scipy.optimize import approx_fprime
from pymps.spin import U4 # 15d real parametrisation of SU(2)
from tools import svals

class StateTensor(cirq.TwoQubitGate):
    """StateTensor: represent state tensor as a unitary"""
    def __init__(self, U):
        self.U = U
    def _unitary_(self):
        return self.U
    def _circuit_diagram_info_(self, args):
        return 'U', 'U'

class Environment(cirq.TwoQubitGate):
    """Environment: represents the environment tensor as a unitary"""
    def __init__(self, V):
        self.V = V
    def _unitary_(self):
        return self.V
    def _circuit_diagram_info_(self, args):
        return 'V', 'V'
 
class State(cirq.ThreeQubitGate):
    """State: combination of Environment and StateTensor"""
    def __init__(self, U, V):
        self.U = U
        self.V = V
    def _decompose_(self, qubits):
        return (Environment(self.V)(*qubits[1:3]),
                StateTensor(self.U)(*qubits[:2])) 
    def _circuit_diagram_info_(self, args):
      return 'I\n|\nU', 'V\n|\nU', 'V\n|\nI'

def mat(v):
    '''helper function - put list of elements (real, imaginary) in a square matrix'''
    re, im = np.split(v, 2)
    C = (re+im*1j).reshape(int(sqrt(len(v)/2)), -1)
    return C

def demat(A):
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
    Adds measurements to existing circuit

    :param qubit: qubit to sample bloch vector of 
    :param circuit: circuit to evaluate before sampling
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
    """
    qbs = cirq.LineQubit.range(3)
    r = 0

    LHS, RHS = Circuit(), Circuit()
    LHS.append([State(U, V)(*qbs)])
    RHS.append([Environment(V)(*qbs[:2])])

    sim = Simulator()
    LHS = sim.simulate(LHS).bloch_vector_of(qbs[0])
    RHS = sim.simulate(RHS).bloch_vector_of(qbs[0])
    return norm(LHS-RHS)

def sampled_env_obj_fun(U, V, reps=10000):
    """sampled_env_obj_fun
    """
    qbs = cirq.LineQubit.range(3)
    r = 0

    LHS, RHS = Circuit(), Circuit()
    LHS.append([State(U, V)(*qbs)])
    RHS.append([Environment(V)(*qbs[:2])])

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

        to precision k/100000
        '''

    def f_obj(v, U=U):
        r = full_env_obj_fun(U, embed(mat(v)))
        return r

    def s_obj(v, U=U):
        r = sampled_env_obj_fun(U, embed(mat(v)))
        return r

    obj = s_obj if sample else f_obj

    res = minimize(obj, demat(C0), method='Powell')
    return embed(mat(res.x))

def optimize_ising(J, λ, sample=False, reps=10000, testing=False):
    """optimize H = -J*ZZ+gX
    """
    def sampled_energy(U, V, J, λ, reps=reps):
        qbs = cirq.LineQubit.range(4)
        sim = cirq.Simulator()

        # create the circuit for 2 local measurements
        C =  Circuit().from_ops([Environment(V)(*qbs[2:4]), StateTensor(U)(*qbs[1:3]), StateTensor(U)(*qbs[0:2])])

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

        C = Circuit().from_ops([Environment(V)(*qbs[2:4]), StateTensor(U)(*qbs[1:3]), StateTensor(U)(*qbs[0:2])])
        IZZI = 4*N_body_spins(0.5, 2, 4)[2]@N_body_spins(0.5, 3, 4)[2]
        IIXI = 2*N_body_spins(0.5, 3, 4)[0]
        ψ = sim.simulate(C).final_state
        return np.real(ψ.conj().T@(-J*IZZI+λ*IIXI)@ψ)

    def optimize_energy(N=200, env_update=5, ϵ=1e-1, e_fun = sampled_energy if sample else full_energy):
        """minimizes ising energy in a full parametrisation of SU(4)

        :param N: how many steps to take
        :param env_update: update the environment every env_update steps
        :param ϵ: time step
        :param e_fun: whether to use the sampled or full energy function
        """
        def f(u, V, λ): return full_energy(U4(u), V, λ)
        
        u = randn(15)
        V = get_env(U4(u))

        for n in range(N):
            du = ϵ*approx_fprime(u, f, 0.1, V, 1)
            u -= du
            if not n%env_update:
                print('\nupdating environment\n')
                V = get_env(U4(u), deembed(V))
            print(f(u, V, λ))

        U = U4(u)
        return U, V

    U, V = optimize_energy()
    return U, V
