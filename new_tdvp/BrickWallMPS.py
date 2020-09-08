#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 09:13:34 2020

@author: jamie
"""

import numpy as np
from functools import reduce
from ClassicalTDVPStripped import CircuitSolver, tensor
from scipy.optimize import minimize
from xmps.spin import U4, lambdas
from scipy.stats import unitary_group
from scipy.linalg import eig, expm

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

def paramU(params):
        """
        Return unitaries U1 and U2 from 22 params (15 params for the fully
        parametrised U1 and 7 for the single column of the unitary U2)
        """
        p1 = params[7:]
        p2 = params[:7]
        
        U1 = U4(p1) 
        # find a unit norm column that is going to be accessed by the circuit
        #   and embed this column into a larger unitary matrix.
        
        ##################################
        # u2 =  (self.Z(a) @ self.X(b) @ self.D1(c) @ self.X(-b) @ self.Z(-a)).reshape(4,1)
        #U2 = np.concatenate((u2, null_space(u2.conj().T)), axis = 1)
        ##################################
        # Doesnt look like it works
        
        U2 = OO_unitary(p2)
        
        return U1, U2
    
class bwMPS():
    def __init__(self, Us, l):
        self.layers = len(Us)
        self.Us = Us 
        self.l = l # length of unitaries needed - depends on location of 
                        # operator
                    
    @staticmethod
    def tensor(tensors):
        return reduce(lambda t1,t2: np.kron(t1,t2), tensors)

    def circuit(self, Ml, Mr):
        C = tensor([Ml,[np.eye(4)]*(self.l-1), Mr])
        psi = self.state()
        return C @ psi

    def state(self):
        I = np.eye(2)
        psi = np.array([1] + [0]*( (2**(2*self.l)) - 1) )
        
        for i, u in enumerate(self.Us):
            
            if i % 2 == 0:
                psi = self.tensor([u]*self.l) @ psi
                
            else:
                psi = self.tensor([I] + ([u]*(self.l - 1)) + [I]) @ psi 
        
        return psi

    def to_matrix_left(self):
        """
        0    0    j
        |-U2-|    |
        |    |-U1-|    -->  i--A--j
        |    |    |            ||
        i    d    d'         (d,d')
        """
        assert len(self.Us) == 2 
        U2, U1 = self.Us
        return np.tensordot(U2[...,0,0], U1, (1,2)).reshape(2,4,2)

    def to_matrix_right(self):
        """
        i    0    0
        |    |-U2-|
        |-U1-|    |    -->  i--A--j
        |    |    |            ||
        d    d'   j          (d,d')
        """
        assert len(self.Us) == 2 
        U2,U1 = self.Us
        return np.transpose(np.tensordot(U2[...,0,0], U1, (0,3)).reshape(2,4,2), [2,1,0])

class bwOverlap():
    def __init__(self, bws):
        self.bw1 = bws[0]
        self.bw2 = bws[1]

    def right_exact_circuit(self):
        A1 = self.bw1.to_matrix_right()
        A2 = self.bw2.to_matrix_right().conj().T
        return np.transpose(np.tensordot(A1, A2, (1,1)), [0,2,1,3]).reshape(4,4)  
    
def state_from_params(p,l):
    U1,U2 = CircuitSolver().paramU(p)
    return bwMPS([U2,U1], l).state()

    
def optimize_2layer_bwmps(H, test = False):
    
    def obj(p):
        psi1 = state_from_params(p, 2)
        H1 = np.kron(np.kron(np.eye(2), H), np.eye(2))
        E1 = np.real(psi1.conj().T @ H1 @ psi1)
        
        psi2 = state_from_params(p, 3)
        H2 = np.kron(np.kron(np.kron(np.eye(2),np.eye(2)), H), np.kron(np.eye(2), np.eye(2)))
        E2 = np.real(psi2.conj().T @ H2 @ psi2)
        
        return (E1 + E2) / 2
    
    opt = []
    def cb(xk):
        f = obj(xk)
        #print(f)
        opt.append(f)
        
    
    init_p = np.random.rand(22)
    
    res = minimize(obj, init_p, 
                   method = "Nelder-Mead", 
                   options = {"maxiter":10000}, 
                   tol = 1e-8,
                   callback = cb)

    if test:
        return opt
    
    else:
        return res


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
        l0 = l[:,np.argmax(np.abs(eta))].reshape(2,2)
        return eta[np.argmax(np.abs(eta))], l0

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
        r0 = r[:,np.argmax(np.abs(eta))].reshape(2,2)
        
        return eta[np.argmax(np.abs(eta))], r0 


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
                           
    
    def exact_env(self, U1, U2, U1_, U2_):
        _, Mr = self.RE.exact_environment(U1, U2, U1_, U2_)
        _, Ml = self.LE.exact_environment(U1, U2, U1_, U2_)
        return Mr, Ml

