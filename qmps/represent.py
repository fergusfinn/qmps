import cirq

from xmps.iMPS import iMPS, TransferMatrix

from .tools import cT, direct_sum, unitary_extension, sampled_bloch_vector_of, Optimizer, cirq_qubits, log2, split_2s
from .tools import from_real_vector, to_real_vector, environment_to_unitary
from .tools import unitary_to_tensor
from xmps.spin import U4
from typing import List, Callable, Dict
from .States import State, FullStateTensor, FullEnvironment
from .tools import RepresentMPS
from numpy import concatenate, allclose, tensordot, swapaxes, log2
from numpy.linalg import eig
from numpy import diag
from numpy.random import randn

from scipy.linalg import null_space, norm, cholesky
from scipy.optimize import minimize
from scipy.linalg import polar

import numpy as np

def sqrtm(X):
    Λ, V = eig(X)
    return V@csqrt(diag(X))

def get_env(U, C0=randn(2, 2)+1j*randn(2, 2), sample=False, reps=100000):
    '''NOTE: just here till we can refactor optimize.py
       return v satisfying

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
        """f_obj: take an 8d real vector, use mat to turn it into a 4d complex vector, environment_to_unitary to 
           turn it into a unitary, then calculate the objective function.
        """
        r = full_tomography_env_objective_function(FullStateTensor(U), 
                FullEnvironment(environment_to_unitary(from_real_vector(v))))
        return r

    def s_obj(v, U=U):
        """s_obj: take an 8d real vector, use mat to turn it into a 4d complex vector, environment_to_unitary to 
           turn it into a unitary, then calculate the (sampled) objective function.
        """
        r = sampled_env_objective_function(FullStateTensor(U), 
                FullEnvironment(environment_to_unitary(from_real_vector(v))))
        return r

    obj = s_obj if sample else f_obj

    res = minimize(obj, to_real_vector(C0.reshape(-1)), method='Nelder-Mead')
    return environment_to_unitary(from_real_vector(res.x))

def get_env_exact(U):
    η, l, r = TransferMatrix(unitary_to_tensor(U)).eigs()
    return environment_to_unitary(cholesky(r).conj().T)

def get_env_exact_alternative(U):
    AL, AR, C = iMPS([unitary_to_tensor(U)]).mixed()
    return environment_to_unitary(C)


#######################
# Objective Functions #
#######################

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
    qbs = cirq.LineQubit.range(3)
    r = 0

    LHS, RHS = cirq.Circuit(), cirq.Circuit()
    LHS.append([State(U, V)(*qbs)])
    RHS.append([V(*qbs[:2])])

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
    qbs = cirq.LineQubit.range(3)
    r = 0

    LHS, RHS = cirq.Circuit(), cirq.Circuit()
    LHS.append([State(U, V)(*qbs)])
    RHS.append([V(*qbs[:2])])

    sim = cirq.Simulator()
    LHS = sim.simulate(LHS).bloch_vector_of(qbs[0])
    RHS = sim.simulate(RHS).bloch_vector_of(qbs[0])
    return norm(LHS-RHS)

############################################
# Tensor, StateTensor, Environment, State  #
############################################ 

def get_env_swap_test(u, vertical='Vertical', ansatz='Full', simulate='Simulate', **kwargs):
    return RepresentMPS(u, vertical='Vertical', ansatz='Full', simulate='Simulate', **kwargs)