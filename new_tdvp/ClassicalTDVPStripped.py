#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:02:05 2020

@author: jamie
"""
#from numba import jit
import jax.numpy as jnp
from jax import device_put, jit
import numpy as np
from scipy.linalg import null_space, eig, expm
from scipy.optimize import approx_fprime, minimize
from math import cos, sin
from numpy import pi
from cmath import exp
from functools import partial
from xmps.spin import U4, lambdas
from scipy.stats import unitary_group
from functools import reduce
from qmps.ground_state import Hamiltonian
import matplotlib.pyplot as plt
from math import isclose
from tqdm import tqdm

@jit
def ry(x):
    return jnp.eye(2) * jnp.cos(x / 2) - 1j * jnp.sin(x / 2) * jnp.array([[0, -1j], [1j, 0]])
@jit
def rz(x):
    return jnp.eye(2) * jnp.cos(x / 2) - 1j * jnp.sin(x / 2) * jnp.array([[1, 0],[0, -1]])
@jit
def rx(x):
    return jnp.eye(2) * jnp.cos(x / 2) - 1j * jnp.sin(x / 2) * jnp.array([[0, 1],[1, 0]])

@jit
def U(p1,p2,p3):
    return rz(p1) @ ry(p2) @ rz(p3)


CNOT = jnp.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,0,1],
            [0,0,1,0]
        ])

CNOTr = jnp.array([
            [1,0,0,0],
            [0,0,0,1],
            [0,0,1,0],
            [0,1,0,0]
        ])


@jit
def matrix(params):
    """
     Return the following parametrised 2 qubit unitary:
          
      |      |
    Rz(0)  Rz(3)
    Ry(1)  Ry(4)
    Rz(2)  Rz(5)
      |      |
      @------x
    Rz(6)  Ry(7)
      x------@
      |    Ry(8)
      @------x
      |      |
    Rz(9)  Rz(12)
    Ry(10) Ry(13)
    Rz(11) Rz(14)
      |      |
      
      """
    
    p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14 = params
    
    U1 = U(p2, p1, p0)
    U2 = U(p5, p4, p3)
    U3 = U(p11, p10, p9)
    U4 = U(p12, p13, p14)
    
    return jnp.kron(U3, U4) @ CNOTr @ jnp.kron(np.eye(2), ry(p8)) @ CNOT @ jnp.kron(rz(p6), ry(p7)) @ CNOTr @ jnp.kron(U1, U2)
  
def OO_lambdas():
    """
    keep only those lambdas which have non-zero elements in the first column
    such that we have 7 independant parameters for the 4 complex values.
    """
    OO_lambda_index = [0,1,2,3,4,8,9]
    return lambdas()[OO_lambda_index]


def OO_unitary(p):
    """
    Im not sure this is correct, but we want to parametrise a unitary which 
    has guaranteed inputs as 00. So if this is the case we do not need a full 
    parametrisation of SU(4) as we are not accessing the other elements.
    
    Instead we only need 7 of the generators of SU(4)??
    """
    
    return expm(-1j * np.tensordot(p, OO_lambdas(), [0,0]))


def gradient_descent(cf, gf, init, lr = 0.01, tol = 1e-8, miter = 10000, atol = 1e-6):
    converged = False
    
    iter_ = 0
    v0 = cf(init)
    th0 = np.array(init)

    while not converged:
        th1 = th0 - lr * gf(th0)   
        
        v1 = cf(th1)
        
        if np.abs(v0) < atol:
            return ResultObject(th0, v0, iter_ * 16, "Answer Reached")
        
        if np.abs(v1) < atol:
            return ResultObject(th1, v1, iter_ * 16, "Answer reached")

        if v1 > v0:
            lr /= 2
            iter_ += 1
            continue

        if np.abs(v0 - v1) < tol:
            return ResultObject(th1, v1, iter_ * 16, "CF stopped changing")
        
        if iter_ == miter:
            return ResultObject(th1, v1, iter_ * 16, "Max Iter Reached")
        
        th0 = th1
        v0 = cf(th0)
        
        iter_ += 1

class ResultObject():
    def __init__(self, x, fun, nfev, message):
        self.x = x
        self.fun = fun
        self.nfev = nfev
        self.message = message

class CircuitSolver():
    def __init__(self):
        pass        
    @staticmethod
    def D(theta): 
        return np.array([
                [cos(theta)**2, sin(theta)**2],
                [sin(theta)**2, cos(theta)**2]
                ])
    
    @staticmethod
    def D2(theta):
        return np.array([
                [cos(theta)**2, 0],
                [0, sin(theta)**2]
            ])
    
    @staticmethod
    def X(theta):
        return np.array([
                [cos(pi * theta / 2), -1j * sin(pi * theta / 2)],
                [-1j * sin(pi * theta / 2), cos(pi * theta / 2)]
                ])
        
    @staticmethod
    def Z(theta): 
        return np.array([        
                [1, 0],
                [0, exp(1j * pi * theta)]
                ])
        
    
    @staticmethod
    def D1(theta): 
        return np.array([
                [cos(theta), 0],
                [0, -1j*sin(theta)]
                ])
    
    @staticmethod
    def D3(theta):
        return np.array([
                [cos(theta), 0],
                [0, sin(theta)]
            ])
    
    def M(self, params):
        a,b,c,d,e,f = params 
        M = self.Z(b) @ self.X(c) @ self.Z(d) @ self.D3(a) @ self.X(e) @ self.Z(f)
        return M   
    
    
    def paramU(self, params):
        """
        Return unitaries U1 and U2 from 22 params (15 params for the fully
        parametrised U1 and 7 for the single column of the unitary U2)
        """
        p1 = params[15:]
        p2 = params[:15]
        
        U1 = matrix(p1) 
        # find a unit norm column that is going to be accessed by the circuit
        #   and embed this column into a larger unitary matrix.
        
        ##################################
        # u2 =  (self.Z(a) @ self.X(b) @ self.D1(c) @ self.X(-b) @ self.Z(-a)).reshape(4,1)
        #U2 = np.concatenate((u2, null_space(u2.conj().T)), axis = 1)
        ##################################
        # Doesnt look like it works
        
        U2 = matrix(p2)
        
        return U1, U2

    
class ManifoldOverlap():
    """
    This class holds the jitted jax functions that do the tensor network 
    contractions for time evolving the MPS unitary state
    """
    # the partial decorator works like this:
    # parial(jit, static_argnums = (0,))(ciruit)
    # Jit cannot take classes as an input but we can define this as a static 
    #   argument and jit will ignore it.
    # Then this new function gets applied to the circuit function.
    
    def circuit(self, U1, U2, U1_, U2_, Mr, Ml, W, path = "greedy"):
                
        """
        
        0    0    0    0    0    0   
        |-U2-|    |-U2-|    |-U2-|
        |    |-U1-|    |-U1-|    |
        |    |    |    |    |    |
      |Ml|   |------W-------|   |Mr|    W = exp(i*H*t)
        |    |    |    |    |    |
        |    |-U1'|    |-U1'|    |
        |-U2'|    |-U2'|    |-U2'|
        0    0    0    0    0    0
        
        """                 
        overlap = np.einsum(
                
                U2_, [6,7,26,27],
                U2_, [8,9,28,29],
                U2_, [10,11,30,31],
                U1_, [27,28,22,23],
                U1_, [29,30,24,25],
                W,[22,23,24,25,18,19,20,21],
                Ml, [26,12],
                Mr, [31,17],
                U1, [18,19,13,14],
                U1, [20,21,15,16],
                U2, [12,13,0,1],
                U2, [14,15,2,3],
                U2, [16,17,4,5],
                
                [0,1,2,3,4,5,6,7,8,9,10,11],
                
                optimize = path
            )[0,0,0,0,0,0 ,0,0,0,0,0,0]
        
        return overlap
    
    def path(self):
        
        U1, U2, U1_, U2_ = [unitary_group.rvs(4).reshape(2,2,2,2) for _ in range(4)]
        M = unitary_group.rvs(2)
        W = unitary_group.rvs(16).reshape(2,2,2,2,2,2,2,2)
        
        path = np.einsum_path(
                U2_, [6,7,26,27],
                U2_, [8,9,28,29],
                U2_, [10,11,30,31],
                U1_, [27,28,22,23],
                U1_, [29,30,24,25],
                W,[22,23,24,25,18,19,20,21],
                M, [26,12],
                M, [31,17],
                U1, [18,19,13,14],
                U1, [20,21,15,16],
                U2, [12,13,0,1],
                U2, [14,15,2,3],
                U2, [16,17,4,5],
                [0,1,2,3,4,5,6,7,8,9,10,11],
                optimize = "greedy"
            )[0]
        
        return path


class LeftEnvironment():
    
    def exact_environment_circuit(self, U1, U2, U1_, U2_):
        """
        Find the (left) eigenvalue of the matrix:
            
        0     0     0
        \-U2- \     \
        \     \-U1- \
        i     \     j
              \      
        i'    \     j'
        \     \-U1'-\
        \-U2'-\     \
        0     0     0
        """
        
        M_ij = np.einsum(
                U2_, [3,4,7,8],
                U1_, [8,5,9,10],
                U1,  [9,10,11,2],
                U2,  [6,11,0,1],
                [0,1,4,3,2,5,6,7]
            )[0,0,0,0,:,:,:,:].reshape(4,4)
        
        return M_ij
    
    def exact_environment(self, U1, U2, U1_, U2_):
                
        M_ij = self.exact_environment_circuit(U1, U2, U1_, U2_)
        
        eta, l = eig(M_ij)
        l0 = l[:,np.argmax(eta)].reshape(2,2)
        return eta[np.argmax(eta)], l0
        

class RightEnvironment():
    """
    This class holds the jitted jax functions that do the tensor network 
    contractions for calculating the environment of the MPS unitary state
    """        
    def circuit(self, U1, U2, U1_, U2_, M, path = "greedy"):

        """
        Find the value of the circuit:
              0     0
        i     |-U2- |
        |-U1- |     |
        |     |    |M|
        |-U1'-|     |
        |     |-U2'-|
        j     |     |
              0     0
        """
        
        M_ij = np.einsum(
            U2_, [11,12,10,9],
            U1_, [2,10,4,5],
            M, [9,8],
            U1,  [4,5,1,3],
            U2,  [3,8,6,7],
            [2,1,11,12,6,7],
            optimize = path
        )[:,:,0,0,0,0]
        
        return M_ij
    
    
    def path(self):
        
        U1, U2, U1_, U2_ = [unitary_group.rvs(4).reshape(2,2,2,2) for _ in range(4)]
        M = unitary_group.rvs(2)
        
        path = np.einsum_path(            
            U2_, [11,12,10,9],
            U1_, [2,10,4,5],
            M ,  [9,8],
            U1,  [4,5,1,3],
            U2,  [3,8,6,7],
            [2,1,11,12,6,7],
            optimize = "greedy"
            )[0]
        return path
        
    
    def exact_environment_circuit(self, U1, U2, U1_, U2_):
        """
        Find the eigenvalue of the matrix:
            
        i     0     0
        \     \-U2--\
        \-U1--\     \
        \     \     j
        \     \     
        \     \     j'
        \-U1'-\     \
        \     \-U2'-\
        i'    0     0
        """
        M_ij = np.einsum(
                
                U2_, [4,5,8,7],
                U1_, [3,8,9,10],
                U1,  [9,10,0,11],
                U2,  [11,6,1,2],
                [1,2,4,5,0,3,6,7]
            )[0,0,0,0,:,:,:,:].reshape(4,4)
        
        return M_ij
    
    def exact_environment(self, U1, U2, U1_, U2_):
        
        M_ij = self.exact_environment_circuit(U1, U2, U1_, U2_)
        
        eta, r = eig(M_ij)
        r0 = r[:,np.argmax(eta)].reshape(2,2)
        
        return eta[np.argmax(eta)], r0 
   
    
class OverlapCalculator(CircuitSolver):
    def __init__(self):
        super().__init__()
    """
    This class holds the jitted jax functions that do the tensor network 
    contractions for calculating expectation values of the MPS unitary state
    """
    
    def expectation_value(self, U1, U2, O, path = "greedy"):
        if len(O.shape) == 4:
            return self.qbt2_exp_val(U1, U2, O, path)
        
        if len(O.shape) == 8:
            return self.qbt4_exp_val(U1, U2, O, path)
    
    def path(self, O):
        if len(O.shape) == 4:
            return self.qbt2_path()
        
        if len(O.shape) == 8:
            return self.qbt4_path()

    
    def qbt4_exp_val(self, U1, U2, O, path = "greedy"):
        """
        Caclulate expectation value of an operator given a state U1, U2:
            
        0    0    0    0    0    0
        \-U2-\    \-U2-\    \-U2-\
        \    \-U1-\    \-U1-\    \
        \    \-------O------\    \
        \    \-U1-\    \-U1-\    \
        \-U2-\    \-U2-\    \-U2-\
        0    0    0    0    0    0
        
        """
        U2_ = U2.reshape(4,4).conj().T.reshape(2,2,2,2)
        U1_ = U1.reshape(4,4).conj().T.reshape(2,2,2,2)
        
        exp_val = np.einsum(
                U2_, [6,7,12,13],
                U2_, [8,9,14,15],
                U2_, [10,11,16,17],
                U1_, [13,14,18,19],
                U1_, [15,16,20,21],
                O,   [18,19,20,21,22,23,24,25],
                U1,  [22,23,26,27],
                U1,  [24,25,28,29],
                U2,  [12,26,0,1],
                U2,  [27,28,2,3],
                U2,  [29,17,4,5],
                [0,1,2,3,4,5,6,7,8,9,10,11],
                optimize = path
            )[0,0,0,0,0,0 ,0,0,0,0,0,0]
        
        return exp_val.real
    
    
    def qbt2_exp_val(self, U1, U2, O, path = "greedy"):
        """
        
        Calculate the expectation value of an operator 
        given a state U1, U2
        
        0    0    0    0
        |-U2-|    |-U2-|
        |    |-U1-|    |
        |    |    |    |
        |    |-O--|    |
        |    |    |    |
        |    |-U1'|    |
        |-U2'|    |-U2'|
        0    0    0    0
        
        """

        U2_ = U2.reshape(4,4).conj().T.reshape(2,2,2,2)
        U1_ = U1.reshape(4,4).conj().T.reshape(2,2,2,2)
                
        exp_value = np.einsum(
                U2_, [4,5,8,9],
                U2_, [6,7,10,11],
                U1_, [9,10,12,13],
                O,   [12,13,14,15],
                U1,  [14,15,16,17],
                U2,  [8,16,0,1],
                U2,  [17,11,2,3],
                [4,5,6,7,0,1,2,3],
                optimize = path
            )[0,0,0,0, 0,0,0,0]
        
        return exp_value.real

    def qbt2_path(self):
        
        U1, U2, U1_, U2_, O = [unitary_group.rvs(4).reshape(2,2,2,2) for _ in range(5)]
        
        path = np.einsum_path(
                U2_, [4,5,8,9],
                U2_, [6,7,10,11],
                U1_, [9,10,12,13],
                O,   [12,13,14,15],
                U1,  [14,15,16,17],
                U2,  [8,16,0,1],
                U2,  [17,11,2,3],
                [4,5,6,7,0,1,2,3],
                optimize = "greedy"
            )[0]
        
        return path
    
    def qbt4_path(self):
        U1, U2, U1_, U2_ = [unitary_group.rvs(4).reshape(2,2,2,2) for _ in range(4)]
        O = unitary_group.rvs(16).reshape(2,2,2,2,2,2,2,2)
        
        path = np.einsum_path(
                U2_, [6,7,12,13],
                U2_, [8,9,14,15],
                U2_, [10,11,16,17],
                U1_, [13,14,18,19],
                U1_, [15,16,20,21],
                O,   [18,19,20,21,22,23,24,25],
                U1,  [22,23,26,27],
                U1,  [24,25,28,29],
                U2,  [12,26,0,1],
                U2,  [27,28,2,3],
                U2,  [29,17,4,5],
                [0,1,2,3,4,5,6,7,8,9,10,11],
                optimize = "greedy"
            )[0]
        
        return path

        
class Represent(CircuitSolver):
    """
    This class performs the variational search for calculating the environment
    of the MPS
    """
    def __init__(self):
        super().__init__()
        self.RE = RightEnvironment()
        self.LE = LeftEnvironment()
        self.right_params = None
        self.path = self.RE.path()
        self.convergence = []
        self.gradients = []
        self.params_updates = []
        
    def cost_function(self, params):
        eta, *p = params
        M = self.M(p)
        return np.linalg.norm(eta * M - self.RE.circuit(self.U1, self.U2,
                                                       self.U1_, self.U2_, M, 
                                                       self.path))

    def optimize(self, U1, U2, U1_, U2_):
        self.U1 = U1
        self.U2 = U2
        self.U1_ = U1_
        self.U2_ = U2_
        
        res = minimize(self.cost_function, 
                       x0 = [1.0,np.pi/4,0,0,0,0,0],
                       method = "Nelder-Mead",
                       options = {"disp":False,
                                  "xatol":1e-8,
                                  "fatol":1e-8,
                                  "maxiter":10000})
        
        self.right_params = res
        return res
    
    def grad(self, params):
        return approx_fprime(params, self.cost_function, epsilon = 1e-8)
    
    
    def optimize_by_hand(self, Us, init_params = np.array([1.0,np.pi/4,0,0,0,0,0]), 
                         atol = 1e-4, alpha = 0.1, 
                         tol = 1e-6, maxiter = 10000):
        
        self.U1, self.U2, self.U1_, self.U2_ = Us
        
        res = gradient_descent(self.cost_function, self.grad, init_params)
        return res
                       
    
    def exact_env(self, U1, U2, U1_, U2_):
        _, Mr = self.RE.exact_environment(U1, U2, U1_, U2_)
        _, Ml = self.LE.exact_environment(U1, U2, U1_, U2_)
        return Mr, Ml


class Optimize(CircuitSolver):
    """
    This class performs the variational search to minimize the energy of the 
    MPS for a given Hamiltonian.
    
    Methods:
    - cost_function
    - optimize
    """

    def __init__(self):
        super().__init__()
        self.OC = OverlapCalculator()
        self.RE = Represent()
        self.path = None
        self.energy_opt = []
        
    def cost_function(self, params):
        U1, U2 = self.paramU(params)
        U1 = U1.reshape(2,2,2,2)
        U2 = U2.reshape(2,2,2,2)
    
        exp_val = self.OC.expectation_value(U1, U2, self.O, self.path)
        return exp_val
        
    def optimize(self, O, initial_params = None):
        
        if initial_params is None:
            initial_params = np.random.rand(30)
        
        self.O = O
        if self.path is None:
            self.path = self.OC.path(O)
        
        res = minimize(self.cost_function, 
                       x0 = initial_params,
                       callback = self.callback,
                       method = "Nelder-Mead")

        return res
    
    def callback(self, xk):
        self.energy_opt.append(self.cost_function(xk))
        
    
class Evolve(CircuitSolver):
    """
    This class performs the variational search that implements the TDVP 
    projection onto the manifold of fixed bond dimension MPS.
    
    Methods:
        
    - cost_function: CF using variational search for environment
    - exact_cost_function: CF using exact environment using eigenvectors
    - optimize: variationally find the maximum overlap state
    - exact_optimize: optimize using the exact cost function
    """
    def __init__(self):
        super().__init__()
        self.MO = ManifoldOverlap()
        self.RE = Represent()
        self.path = self.MO.path()
        self.cf_convergence = []
        
    def cost_function(self, params):
        U1_, U2_ = self.paramU(params)
        U1_ = U1_.conj().T.reshape(2,2,2,2)
        U2_ = U2_.conj().T.reshape(2,2,2,2)        
        
        env_params = self.RE.optimize(self.U1, self.U2, U1_, U2_)
        M = self.M(env_params.x[1:])
        
        overlap = self.MO.circuit(self.U1, self.U2, 
                                  U1_, U2_, 
                                  M, M,
                                  self.W, 
                                  self.path)
        
        return -np.abs(overlap)**2
        
    def exact_cost_function(self, params):
        U1_, U2_ = self.paramU(params)
        U1_ = U1_.conj().T.reshape(2,2,2,2)
        U2_ = U2_.conj().T.reshape(2,2,2,2)        
        
        Mr, Ml = self.RE.exact_env(self.U1, self.U2, U1_, U2_)

        overlap = self.MO.circuit(self.U1, self.U2, 
                                  U1_, U2_, 
                                  Mr, Mr.conj().T, 
                                  self.W, 
                                  self.path)
        
        return -np.abs(overlap)**2
    
    def optimize(self, W, U1, U2, initial_params = None):
        if initial_params is None:
            initial_params = np.random.rand(30)
            
        self.W = W
        self.U1 = U1
        self.U2 = U2
        res = minimize(self.cost_function, 
                       x0 = initial_params,
                       callback = self.callback,
                       method = "Nelder-Mead",
                       options = {"maxiter":len(initial_params)*1000})
        
        return res
    
    def callback(self, xk):
        self.cf_convergence.append(self.cost_function(xk))
    
    def exact_callback(self, xk):
        self.cf_convergence.append(self.exact_cost_function(xk))
    
    def exact_optimize(self, W, U1, U2, initial_params = None, record = False):
        if initial_params is None:
            initial_params = np.random.rand(30)
            
        if record is True:
            self.cf_convergence = []
            callback = self.exact_callback
            
        else:
            callback = None
            
        self.W = W
        self.U1 = U1
        self.U2 = U2
        
        res = minimize(self.exact_cost_function, 
                       x0 = initial_params,
                       callback = callback,
                       options = {"fatol":1e-8,
                                  "xatol":1e-8},
                       method = "Nelder-Mead")
        
        return res
    
    def time_evolve(self, steps, W, init_params = None, show_convergence = False):
        """
        Time evolve up to a time T = dt * steps. 
        """
        if init_params is None:
            init_params = np.random.rand(30)
        
        
        results = []
        
        for i in tqdm(range(steps)):
            U1, U2 = self.paramU(init_params)
            
            res_e = self.exact_optimize(W,
                                        U1.reshape(2,2,2,2),
                                        U2.reshape(2,2,2,2),
                                        initial_params = init_params,
                                        record = show_convergence)
            
            if show_convergence:
                flag = np.random.rand(1)
                if flag < 0.5:
                    plt.plot(self.cf_convergence)
                    plt.title(f"Convergence of step {i}")
                    plt.show()
                
            results.append(res_e)
            
            init_params = res_e.x
            
        return results
    
    
class Optimizer(CircuitSolver): 
    """
    Class to implement representation, optimization, and time evolution.
    
    Use the methods
        
        self.optimize
        self.represent
        self.evolve
        
    to access three classes that implement the code
    """
       
    def __init__(self):
        super().__init__()
        self.optimize = Optimize()
        self.represent = Represent()
        self.evolve = Evolve()
             
         
X0 = np.array([
        [0,1],
        [1,0]
    ])


Z0 = np.array([
        [1,0],
        [0,-1]
    ])


I = np.array([
        [1,0],
        [0,1]
    ])


def tensor(tensors):
    return reduce(lambda t1,t2: np.kron(t1,t2), tensors)


    
    

    
    
    