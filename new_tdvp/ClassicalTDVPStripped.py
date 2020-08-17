#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:02:05 2020

@author: jamie
"""

import jax.numpy as jnp
from jax import device_put, devices, jit
import numpy as np
from scipy.linalg import null_space, eig, expm
from scipy.optimize import approx_fprime, minimize
from math import cos, sin
from numpy import pi
from cmath import exp
from functools import partial
from xmps.spin import U4
from scipy.stats import unitary_group
from functools import reduce

class CircuitSolver():
    
    @staticmethod
    def D(theta): 
        return np.array([
                [cos(theta)**2, sin(theta)**2],
                [sin(theta)**2, cos(theta)**2]
                ])
    @staticmethod
    def X(theta):
        return np.array([
                [cos(pi * theta / 2), sin(pi * theta / 2)],
                [sin(pi * theta / 2), cos(pi * theta / 2)]
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
    
    def M(self, params):
        a, b, c, d = params 
        M = self.Z(a) @ self.X(b) @ self.Z(d) @ self.D(c) @ self.Z(-d) @ self.X(-b) @ self.Z(-a)
        return M / np.linalg.norm(M)
    

class RightEnvironmentFinder(CircuitSolver):
    def __init__(self, U1=None, U2=None, U1_=None, U2_=None, dt=0.01):
        self.U1 = U1 if U1 else unitary_group.rvs(4)
        self.U2 = U2 if U2 else unitary_group.rvs(4)
        self.U1_ = U1_ if U1_ else unitary_group.rvs(4)
        self.U2_ = U2_ if U2_ else unitary_group.rvs(4)
        self.path = None
        self.grad = partial(approx_fprime, f = self.cost_function, epsilon = 0.001)
        self.dt = dt # need dt to set effective bounds on the change in the eigenvalue of right environment
        self.rightParams = None 
        
    def circuit(self, params):
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
        
        # we want to calculate the path once and only once, so the path finding 
        #   cost is minimized
        if self.path is None:
            self.path = np.einsum_path(
                self.U2_, [11,12,10,9],
                self.U1_, [2,10,4,5],
                self.M(params), [9,8],
                self.U1, [4,5,1,3],
                self.U2, [3,8,6,7],
                [2,1,11,12,6,7],
                optimize = "optimal"
                )[0]
        
        M_ij = np.einsum(
            self.U2_, [11,12,10,9],
            self.U1_, [2,10,4,5],
            self.M(params), [9,8],
            self.U1, [4,5,1,3],
            self.U2, [3,8,6,7],
            [2,1,11,12,6,7],
            optimize = self.path
        )[:,:,0,0,0,0]
        
        return M_ij / np.linalg.norm(M_ij)

    def exact_right_env(self):
        """
        
        Find the eigenvalues of the matrix:
            
                0    0
           i    |-U2-|
           |-U1-|    |
           |    |    j
           |    |    
           |    |    j'
           |-U1-|    |
           i'   |-U2-|
                0    0
           
           in order to find the right environment. This is faster than 
           finding the environment variationally
            
        """
        
        
        M_ij = np.einsum(
            self.U2_, [3,4,7,11],
            self.U1_, [11,5,9,10],
            self.U1, [9,10,8,2],
            self.U2, [6,8,0,1],
            [0,1,3,4,6,7,2,5]
        )[0,0,0,0,:,:,:,:].reshape(4,4)
        
        eta, r = eig(M_ij)
        
        r = r[:,0].reshape(2,2)
        
        n = np.trace(r.conj().T @ r)
    
        return eta[0], r / np.sqrt(n)
        
            
    def updateUs(self, new_Us):
        """
        Update unitaries in in order U1,U2,U1_,U2_
        """
        self.U1, self.U2, self.U1_, self.U2_ = new_Us
        
        
    def cost_function(self, params):
        eta, *p = params
        return np.linalg.norm(eta * self.M(p) - self.circuit(p))
    
    def optimize(self):
        res = minimize(self.cost_function,
                       x0 = [1.0,0,0,0,0],
                       method = "TNC",
                       jac = self.grad,
                       bounds = ((1 - 5 * (self.dt**2), 1),
                                 (None, None),
                                 (None, None),
                                 (None, None),
                                 (None, None))
                       )
        
        self.rightParams = res.x
        return res.x[1:]
    
    def exact_environment(self):
        """
        Find the eigenvalue of the matrix:
            
        0     0     j
        |-U2- |     |
        |     |-U1- |
        i     |     |
              |     |
        i'    |     |
        |     |-U1'-|
        |-U2'-|     |
        0     0     j'
        """
        
        M_ij = np.einsum(
            self.U2_, [3,4,7,11],
            self.U1_, [11,5,9,10],
            self.U1, [9,10,8,2],
            self.U2, [6,8,0,1],
            [0,1,3,4,6,7,2,5]
            )[0,0,0,0,:,:,:,:].reshape(4,4)
    
        eta, r = eig(M_ij)
        
        r = r[:,0].reshape(2,2)
    
        n = np.trace(r.conj().T @ r)

        return eta[0], r / np.sqrt(n)
    
    
    def full_right_environment(self):
        """
        Return the right environment matrix:
        
        0   0
        |U2 |
       i|   |
           |M|
       j|   |
        |U2'|
        0   0

        """
        
        r = np.einsum(
            self.U2_,[2,3,5,7],
            self.M(self.rightParams),  [7,6],
            self.U2, [4,6,0,1],
            [0,1,2,3,4,5]
            )[0,0,0,0,:,:]
    
        return r

class TDVPCircuitCalculator(CircuitSolver):
    def __init__(self, U1, U2, H, dt, exact = True):
        self.U1 = device_put(U1)
        self.U2 = device_put(U2)
        self.H = H
        self.dt = dt
        self.W = expm(1j * H * dt).reshape(*[2]*8)
        self.init_params = np.random.rand(18)
        self.path = None
        self.rCalc = RightEnvironmentFinder(dt = self.dt)
        self.exact = exact
        self.path_info = None

        
    def paramU(self, params):
        """
        Return unitaries U1 and U2 from 18 params (15 params for the fully
        parametrised U1 and 3 for the single column of the unitary U2)
        """
        a,b,c = params[:3]
        
        U1_params = params[3:]
        U1 = U4(U1_params) 
        # find a unit norm column that is going to be accessed by the circuit
        #   and embed this column into a larger unitary matrix.
        u2 =  (self.Z(a) @ self.X(b) @ self.D1(c) @ self.X(-b) @ self.Z(-a)).reshape(4,1)
        U2 = np.concatenate((u2, null_space(u2.conj().T)), axis = 1)
        
        return U1, U2
    
    
    def circuit(self, params):
        
        U1_, U2_ = self.paramU(params)
        U1_ = U1_.conj().T.reshape(2,2,2,2)
        U2_ = U2_.conj().T.reshape(2,2,2,2)
        
        self.rCalc.updateUs([self.U1,self.U2,U1_,U2_])
        
        if self.exact:
            _, M = self.rCalc.exact_right_env()
            
        if not self.exact:
            params = self.rCalc.optimize()
            M = self.M(params)
        
        if self.path is None:
            p = np.einsum_path(
                U2_, [6,7,26,27],
                U2_, [8,9,28,29],
                U2_, [10,11,30,31],
                U1_, [27,28,22,23],
                U1_, [29,30,24,25],
                self.W,[22,23,24,25,18,19,20,21],
                M, [26,12],
                M, [31,17],
                self.U1, [18,19,13,14],
                self.U1, [20,21,15,16],
                self.U2, [12,13,0,1],
                self.U2, [14,15,2,3],
                self.U2, [16,17,4,5],
                [0,1,2,3,4,5,6,7,8,9,10,11],
                optimize = "greedy"
                )
            self.path = p[0]
            self.path_info = p[1]

4r5    
    
    def optimize(self):
        
        res = minimize(self.circuit, 
                       x0 = self.init_params, 
                       method = "Nelder-Mead")
        
        self.init_params = res.x
        
        self.U1, self.U2 = self.paramU(res.x)
        self.U1 = self.U1.reshape(2,2,2,2)
        self.U2 = self.U2.reshape(2,2,2,2)
        
        return self.U1, self.U2


class ManifoldOverlap():

    def circuit(U1, U2, U1_, U2_, M, W, path):
        
        overlap = jnp.einsum(
                
                U2_, [6,7,26,27],
                U2_, [8,9,28,29],
                U2_, [10,11,30,31],
                U1_, [27,28,22,23],
                U1_, [29,30,24,25],
                self.W,[22,23,24,25,18,19,20,21],
                M, [26,12],
                M, [31,17],
                self.U1, [18,19,13,14],
                self.U1, [20,21,15,16],
                self.U2, [12,13,0,1],
                self.U2, [14,15,2,3],
                self.U2, [16,17,4,5],
                
                [0,1,2,3,4,5,6,7,8,9,10,11],
                
                optimize = path
            )[0,0,0,0,0,0 ,0,0,0,0,0,0]
        
        return -np.abs(overlap)**2
    
    
class RightEnvironment():
    
    
    def circuit(U1, U2, U1_, U2_, M, path):
        M_ij = jnp.einsum(
            U2_, [11,12,10,9],
            U1_, [2,10,4,5],
            M  , [9,8],
            U1,  [4,5,1,3],
            U2,  [3,8,6,7],
            [2,1,11,12,6,7],
            optimize = path
        )[:,:,0,0,0,0]
        
        return M_ij / np.linalg.norm(M_ij)

    
    































###########################################
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

H0 = lambda J, g: -J * ( np.kron(Z0,Z0) + (g/2)*( np.kron(I,X0) + np.kron(X0,I) ) )

H = lambda J, g: tensor([H0(J,g), I, I]) + tensor([I, H0(J,g), I]) + tensor([I, I, H0(J,g)])

if __name__ == "__main__":
    # Ham = H(-1,1)
    # U1 = unitary_group.rvs(4).reshape(2,2,2,2)    
    # U2 = unitary_group.rvs(4).reshape(2,2,2,2)
    # TDVP = TDVPCircuitCalculator(U1, U2, Ham, 0.01, False)

    # start_1 = time.time()
    # a = TDVP.circuit(np.random.rand(18))
    # end_1 = time.time()
    # diff_1 = start_1 - end_1
    # print("Time Including Path Calculation: ", diff_1)
    
    # start_1 = time.time()
    # a = TDVP.circuit(np.random.rand(18))
    # end_1 = time.time()
    # diff_1 = start_1 - end_1
    # print("Time Excluding Path Calculation: ", diff_1)
    print(devices())
    