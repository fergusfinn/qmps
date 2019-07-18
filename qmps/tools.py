import cirq

from numpy import eye, concatenate, allclose, swapaxes, tensordot
from numpy import array, pi as π, arcsin, sqrt, real, imag, split
from numpy import zeros, block, diag, log2

from numpy.random import rand, randint, randn
from numpy.linalg import svd
import numpy as np

from scipy.linalg import null_space, norm, svd
from scipy.optimize import minimize

import matplotlib.pyplot as plt

import os
from typing import Callable, List, Dict


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


class Optimizer:
    '''
    Base class for optimizers. To specify a new optimization technique simply define a new objective function
    '''
    def __init__(self, u_original: cirq.Gate, v_original: cirq.Gate,
                 objective_function: Callable, qaoa_depth: int = 1,
                 initial_guess: List = None, settings: Dict = None):
        self.u = u_original
        self.v = v_original
        self.obj_fun = objective_function
        self.iters = 0
        self.obj_fun_values = []
        self.store_values = False
        self.optimized_result = None
        self.circuit = None
        self.initial_guess = initial_guess if initial_guess is not None else np.random.random(2*qaoa_depth)
        self.bond_dim = 2**(self.u.num_qubits()-1)

        self._settings_ = settings if settings else{
                                            'maxiter': 100,
                                            'verbose': False,
                                            'method': 'Nelder-Mead',
                                            'tol': 1e-5,
                                            'store_values': False
                                            }

    def settings(self, new_settings):
        self._settings_.update(new_settings)

    def callback_store_values(self, xk):
        val = self.objective_function(xk)
        self.obj_fun_values.append(val)
        if self._settings_['verbose']:
            print(f'{self.iters}:{val}')
        self.iters += 1

    def objective_function(self, params):
        pass

    def get_env(self):
        options = {'maxiter': self._settings_['maxiter'],
                   'disp': self._settings_['verbose']}

        kwargs = {'fun': self.objective_function,
                  'x0': self.initial_guess,
                  'method': self._settings_['method'],
                  'tol': self._settings_['tol'],
                  'options': options,
                  'callback': self.callback_store_values if self._settings_['store_values'] else None}

        self.optimized_result = minimize(**kwargs)
        self.update_final_circuits()
        if self._settings_['verbose']:
            print(f'Reason for termination is {self.optimized_result.message}')
        return self

    def plot_convergence(self, file, exact_value=None):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.figure()
        x = range(len(self.obj_fun_values))
        plt.plot(x, self.obj_fun_values)
        if exact_value is not None:
            plt.axhline(exact_value, c='r')
        plt.xlabel('iterations')
        plt.ylabel('objective function value')
        plt.show()
        if file:
            plt.savefig(dir_path + '/' + file)

    def update_final_circuits(self):
        pass

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
