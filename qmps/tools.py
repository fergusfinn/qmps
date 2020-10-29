from scipy.optimize import minimize_scalar

from numpy import eye, concatenate, allclose, swapaxes, tensordot
from numpy import array, pi as π, arcsin, sqrt, real, imag, split
from numpy import zeros, block, diag, log2
from numpy.random import rand, randint, randn
from numpy.linalg import svd, qr
import numpy as np

from xmps.spin import U4
from xmps.iMPS import TransferMatrix, iMPS

from scipy.linalg import null_space, norm, svd, cholesky
from scipy.optimize import minimize

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
#    from skopt import gp_minimize

import matplotlib.pyplot as plt
import os
from typing import Callable, List, Dict
import cirq
from tqdm import tqdm


def get_circuit(state, decomp=None):
    if decomp == 'Full':
        return cirq.Circuit.from_ops(cirq.decompose(state(*cirq_qubits(state.num_qubits()))))
    elif decomp == 'Once':
        return cirq.Circuit.from_ops(cirq.decompose_once(state(*cirq_qubits(state.num_qubits()))))
    else:
        return cirq.Circuit.from_ops(state(*cirq_qubits(state.num_qubits())))

def random_unitary(*args):
    return qr(randn(*args))[0]

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

def split_3s(x):
    """split_3s: take a list: [β, γ, β, γ, ...], return [[β, γ, β], [γ, β, γ], ...]
    """
    return [x[i:i+3] for i in range(len(x)) if not i%3]

def split_ns(x, n):
    """split_ns: take a list: [β, γ, β, γ, ...], return [[β, γ, β], [γ, β, γ], ...]
    """
    return [x[i:i+n] for i in range(len(x)) if not i%n]

def get_env_exact(U):
    """get_env_exact: v. useful for testing. Much faster than variational optimization of the env.

    :param U:
    """
    η, l, r = TransferMatrix(unitary_to_tensor(U)).eigs()
    return environment_to_unitary(cholesky(r).conj().T)

def get_env_exact_alternative(U):
    AL, AR, C = iMPS([unitary_to_tensor(U)]).mixed()
    return environment_to_unitary(C)

def sqrtm(X):
    Λ, V = eig(X)
    return V@csqrt(diag(X))

###################
# Optimizers
###################
class OptimizerCircuit:
    def __init__(self, circuit=None, total_qubits=None, aux_qubits=None):
        self.circuit = circuit
        self.total_qubits = total_qubits
        self.aux_qubits = aux_qubits
        self.qubits = None


class Optimizer:
    def __init__(self, u = None, v = None, initial_guess=None, obj_fun = None, args = None):
        self.u = u
        self.v = v

        self.initial_guess = initial_guess
        self.iters = 0
        self.optimized_result = None
        self.obj_fun_values = []
        self.settings = {
            'maxiter': 10000,
            'verbose': True,
            'method': 'Nelder-Mead',
            'tol': 1e-8,
            'store_values': True,
            'bayesian': False,
        }
        self.is_verbose = self.settings['verbose']
        self.obj_fun = obj_fun
        self.args = args
        self.circuit = OptimizerCircuit()
        # self.gate = U4 if gate is None else gate

    def change_settings(self, new_settings):
        return self.settings.update(new_settings)

    def gate_from_params(self, params):
        pass

    def update_state(self):
        pass

    def callback_store_values(self, xk):
        val = self.objective_function(xk)
        self.obj_fun_values.append(val)
        if self.settings['verbose']:
            print(f'{self.iters}:{val}')
        self.iters += 1

    def objective_function(self, params):
        if self.obj_fun is not None:
            return self.obj_fun(params, *self.args)
        else:
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

        if self.settings['bayesian']:
            self.optimized_result = gp_minimize(self.objective_function, [(-np.pi, np.pi)]*(len(self.initial_guess)))
        elif self.settings['method'] == 'Rotosolve':
            self.optimized_result = double_rotosolve(self.objective_function, self.initial_guess, options['maxiter'], options['disp'])
        else:
            self.optimized_result = minimize(**kwargs)

        self.update_state()
        if self.is_verbose and not self.settings['bayesian']:
            print(f'Reason for termination is {self.optimized_result.message} ' +
                  f'\nObjective Function Value is {self.optimized_result.fun}')
        return self.optimized_result

    def plot_convergence(self, file, exact_value=None):
        import os
        dir_path = os.path.abspath('')
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


class GuessInitialFullParameterOptimizer(Optimizer):    
    def objective_function(self, params):
        target_u = FullStateTensor(U4(params).conj())
        num_qubits = 2*self.u.num_qubits()
        qubits = cirq.LineQubit.range(num_qubits)
        self.circuit.circuit = cirq.Circuit.from_ops([cirq.H.on(qubits[0]), cirq.H.on(qubits[1]),
                                                      cirq.CNOT.on(qubits[0], qubits[2]),
                                                      cirq.CNOT.on(qubits[1], qubits[3]),
                                                      self.u.on(*qubits[0:2]),
                                                      target_u.on(*qubits[2:4]),
                                                      cirq.CNOT.on(qubits[0], qubits[2]),
                                                      cirq.CNOT.on(qubits[1], qubits[3]),
                                                      cirq.H.on(qubits[0]), cirq.H.on(qubits[1])])

        simulator = cirq.Simulator()
        results = simulator.simulate(self.circuit.circuit)
        final_state = results.final_simulator_state.state_vector[0]
        score = np.abs(final_state)**2
        return 1 - score


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

def random_full_rank_circuit(length, depth, ψχϕs=None):
    ψχϕs = [[(None, None, None) for _ in range(length)]
            for _ in range(depth)] if ψχϕs is None else ψχϕs
    qubits = cirq.LineQubit.range(length)
    circuit = cirq.Circuit()

    def U(i, ψ=None, χ=None, ϕ=None):
        """U: Random SU(2) element"""
        ψ = 2*π*rand() if ψ is None else ψ
        χ = 2*π*rand() if χ is None else χ
        φ = arcsin(sqrt(rand())) if ϕ is None else ϕ
        for g in [cirq.Rz(χ+ψ), cirq.Ry(2*φ), cirq.Rz(χ-ψ)]:
            yield g(cirq.LineQubit(i))
    for j in range(depth):

        # Define a parametrisation of su(2**(N-1))
        # MPS matrices will be U(...), xU(...)
        for i in range(1, length):
            circuit.append(U(i, *ψχϕs[j][i]))
        circuit.append(reversed([cirq.CNOT(qubits[i], qubits[i+1]) for i in range(1, length-1)]))

        # Add on all the rest
        circuit.append(cirq.H(qubits[0]))
        circuit.append(cirq.CNOT(qubits[0], qubits[1]))
        circuit.append(cirq.SWAP(qubits[i], qubits[i+1]) for i in range(length-1))
        circuit.append(cirq.SWAP(qubits[-1], qubits[0]))
    return circuit

def double_rotosolve(ϵ, initial_parameters, N_iters=100, disp=True):
    S = []
    ss = []
    es = []
    params = initial_parameters
    I = np.eye(len(params))
    for w in range(N_iters):
        if disp:
            print(w, ', ', sep='', end='', flush=True)
        for i, _ in tqdm(enumerate(params)):
            def M(x):
                return np.sum(ϵ(params+I[i]*x))
            A = (M(0)+M(np.pi))
            B = (M(0)-M(np.pi))
            C = (M(np.pi/2)+M(-np.pi/2))
            D = (M(np.pi/2)-M(-np.pi/2))
            E = (M(np.pi/4)-M(-np.pi/4))

            a, b, c, d = 1/4*(2*E-np.sqrt(2)*D), 1/4*(A-C), 1/2*D, 1/2*B

            P = np.sqrt(a**2+b**2)
            u = np.arctan2(b, a)
            
            Q = np.sqrt(c**2+d**2)
            v = np.arctan2(d, c)

            def f(x): return (P*np.sin(2*x+u)+Q*np.sin(x+v))

            θ_ = minimize_scalar(f, bounds = [-np.pi, np.pi]).x
            #θ_ = (-np.pi/2-np.arctan2(2*ϵ(params)-ϵ(params+I[i]*π/2)-ϵ(params-I[i]*π/2), ϵ(params+I[i]*π/2)-ϵ(params-I[i]*π/2)))
            params[i] += np.arctan2(np.sin(θ_), np.cos(θ_))
            #params[i] = np.arctan2(np.sin(params[i]), np.cos(params[i]))
        print('\n', sep='', end='', flush=True)
        #double_sinusoids(H, state_function, params)
        es.append(ϵ(params))
    return RotosolveResult(es, es[-1], params, '')

class RotosolveResult(object):
    def __init__(self, history, fun, x, message):
        self.history = history 
        self.fun = fun
        self.x = x
        self.message = message
