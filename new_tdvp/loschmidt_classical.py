#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:15:09 2020

@author: jamie

A classical implementation of the brick wall style TDVP requiring fewer qubits 
and more efficient optimization. We will optimize the right environment at 
each step, and compare to the exact version.

Structure of the work:
    
1) define an initial state A_0

2) Translate this into the unitary U1 and U2 

for step in steps:

    3) Find Right environment of U1 and U2 variationally

    4) Find right environment exactly and compare
    
    5) Find overlap of U and U' with time evo operator
    
    6) Select the maximum overlap U'
    

Things to do:
    
a) function that takes in A and produces U1 and U2
b) function that takes in U1 and U2 and produces a right environment
c) function that takes U1, U2, U1', U2' and H and finds overlap
d) Function that takes U1, U2 and turns into an A (using cholesky decomp)?
e) function that takes 15 params and returns a U. This will be an 
        overparametrisation but its a good start.

"""
import numpy as np
from xmps.iMPS import iMPS
from scipy.linalg import qr, polar, null_space, eig, expm
from qmps.tools import tensor_to_unitary
from scipy.optimize import approx_fprime, minimize
from math import cos, sin
from numpy import pi
from cmath import exp
from functools import partial
from xmps.spin import U4
from scipy.stats import unitary_group
import matplotlib.pyplot as plt 

D = lambda theta: np.array([
        [cos(theta)**2, sin(theta)**2],
        [sin(theta)**2, cos(theta)**2]
    ])

D1 = lambda theta: np.array([
        [cos(theta), 0],
        [0, -1j*sin(theta)]
    ])
X = lambda theta: np.array([
        [cos(pi * theta / 2), sin(pi * theta / 2)],
        [sin(pi * theta / 2), cos(pi * theta / 2)]
    ])

Z = lambda theta: np.array([        
    [1, 0],
    [0, exp(1j * pi * theta)]
    ])

R = lambda a, b, c, d: Z(a) @ X(b) @ Z(d) @ D(c) @ Z(-d) @ X(-b) @ Z(-a)


def env_from_Us_exact(U1, U2, U1_, U2_):

    M_ij = np.einsum(
        U2_, [3,4,7,11],
        U1_, [11,5,9,10],
        U1, [9,10,8,2],
        U2, [6,8,0,1],
        [0,1,3,4,6,7,2,5]
    )[0,0,0,0,:,:,:,:].reshape(4,4)
    
    eta, r = eig(M_ij)
        
    r = r[:,0].reshape(2,2)
    
    n = np.trace(r.conj().T @ r)

    return eta[0], r / np.sqrt(n)
    
    
def Us_from_A(A):
    """
    N.B. I don't know how useful this will prove
    
    Process to get the brick U1 and U2 from translationally invariant
    single site A mps.
    
    Multiple the two As together:
        
    1)  --A-A--  -> --B--
          | |        ||
      
     Use QR decomp to turn B into:
         
    2)    -- C --D --
                ||
          
    3) Reshape D into 4 x 4
    
    Use Polar Decomp to turn D into a unitary:
        
                      | |
                       H
     4) --D--   =     | |
         ||           U_d    
                      | |    
                     
    5) Then multiply H into C and embed in a unitary
    """
    
    # 1)
    B = np.transpose(np.tensordot(A,A, axes = (2,1)), [1,0,3,2]).reshape(2,8)
    
    #2)
    C,D = qr(B, overwrite_a=True) # overwrite_a can give better performance
    
    #3)
    D = np.transpose(D.reshape(2,2,2,2),[1,2,0,3]).reshape(4,4)
    
    #4)
    U_d, H = polar(D)
    
    #5)
    H = H.reshape(2,2,2,2)
    C_ = np.tensordot(H, C, axes = ((2,3), (1,0))).reshape(4,1)
    C_ = C_ / np.linalg.norm(C_)
    U_c = np.concatenate((C_, null_space(C_.conj().T)),axis=1)
    
    return U_c, U_d


def get_path():
    
    U2 = unitary_group.rvs(4).reshape(2,2,2,2)
    U1 = unitary_group.rvs(4).reshape(2,2,2,2)
    U1_ = unitary_group.rvs(4).reshape(2,2,2,2)
    U2_ = unitary_group.rvs(4).reshape(2,2,2,2)
    a,b,c,d = np.random.rand(4)
    
    path = np.einsum_path(
        U2_, [11,12,10,9],
        U1_, [2,10,4,5],
        R(a,b,c,d), [9,8],
        U1, [4,5,1,3],
        U2, [3,8,6,7],
        [2,1,11,12,6,7],
        optimize = "greedy")[0]
    
    return path


def simulate_circuit(a,b,c,d, U1, U2, U1_, U2_, R, path):        
    """
    produce the circuit:
        
   |0> ------------|D|------------ |0>
          |U2|             |U2`|
   |0> --------------------------- |0>
              |U1|   |U1`|
        i ----------------- j
    
    
    Where D is given exactly here but would normally be embedded in a unitary
    as:
        
    |0> -------------------------- |0>
                    |R|
            i ------------- j
    """
        
    M_ij = np.einsum(
            U2_, [11,12,10,9],
            U1_, [2,10,4,5],
            R(a,b,c,d), [9,8],
            U1, [4,5,1,3],
            U2, [3,8,6,7],
            [2,1,11,12,6,7],
            optimize = path
        )[:,:,0,0,0,0]

    return M_ij
    

def env_from_Us_var(U1, U2, U1_, U2_, dt, e = 0.0001):                
    """
    Variationally return the right environment of the brick MPS structure
    """
    param_circ = partial(simulate_circuit, U1=U1, U2=U2, U1_=U1_, 
                         U2_=U2_, R = R, path = get_path())
    
    def cost_func(params):
        eta, a, b, c, d = params
        return np.linalg.norm(eta * R(a,b,c,d) - param_circ(a,b,c,d))
    
    grad = partial(approx_fprime, f = cost_func, epsilon = e)
    
    res = minimize(cost_func, x0 = [1.0,0,0,0,0], 
                   method = "TNC", 
                   jac = grad, 
                   bounds = ((1 - 5 * (dt**2), 1), 
                             (None, None), 
                             (None, None), 
                             (None, None), 
                             (None, None ))
                   )
    
    return res.x

def parametrised_Us(params):
    a,b,c = params[:3]
    U1_params = params[3:]
    
    U1 = U4(U1_params)    
    u2 =  (Z(a) @ X(b) @ D1(c) @ X(-b) @ Z(-a)).reshape(4,1)
    U2 = np.concatenate((u2, null_space(u2.conj().T)), axis = 1)
    
    return U1, U2

def get_tdvp_path():
    U1, U2, U1_, U2_,W = [unitary_group.rvs(4).reshape(2,2,2,2)]*5

    path = np.einsum_path(
            U2_, [4,5,16,17],
            U2_, [6,7,18,19],
            U1_, [17,18,14,15],
            W,   [14,15,12,13],
            U1,  [12,13,9,10],
            U2,  [8,9,0,1],
            U2,  [10,11,2,3],
            [0,1,2,3,4,5,6,7],
            optimize = "greedy"
        )[0]
    
    return path
    
def tdvp_projection_circuit(U1, U2, U1_, U2_, W, M, path):
    """
    Produce a circuit that calculates the overlap between a time evolved mps
    and the fixed bond dimension manifold.
    
            0   0   0   0   
            |U2 |   |U2 |   
            |   |U1 |   |   
           |M|  |-W-|  |M|  
            |   |U1'|   |   
            |U2'|   |U2'|
            0   0   0   0
            
    """
    overlap = np.einsum(
            U2_, [4,5,16,17],
            U2_, [6,7,18,19],
            U1_, [17,18,14,15],
            W,   [14,15,12,13],
            U1,  [12,13,9,10],
            U2,  [8,9,0,1],
            U2,  [10,11,2,3],
            [0,1,2,3,4,5,6,7],
            optimize = path
        )[0,0,0,0,0,0,0,0]
    
    return overlap

def updateU_cost_func(params, U1, U2, W, path, dt):
    U1_, U2_ = parametrised_Us(params)
    U1_ = U1_.conj().T.reshape(2,2,2,2)
    U2_ = U2_.conj().T.reshape(2,2,2,2)
    
    # get the parameters for the environment, need dt to set bounds on 
    #   the optimization
    params_for_env = env_from_Us_var(U1, U2, U1_, U2_, dt)
    R0 = R(*params_for_env[1:]) 
    M = R0/np.linalg.norm(R0)
    
    # use these params to find M
    
    # claculate the overlap of U with the MPS manifold
    overlap = tdvp_projection_circuit(U1, U2, U1_, U2_, W, M, path)
    return -np.abs(overlap)
    

def update_U(U1, U2, W, path, dt, init):
    
    """
    given a U1, U2 and H we want to find an update to U1 and U2, U1' and U2'
    that maximise overlap with a fixed bond dimension MPS.
    
    
    """
    
    cost_func = partial(updateU_cost_func, U1 = U1, U2 = U2, W = W, path = path, dt=dt)
    # grad = partial(approx_fprime , f=cost_func, epsilon = 0.001)
    res = minimize(cost_func, x0 = init, method = "Nelder-Mead")
    
    U1, U2 = parametrised_Us(res.x)
    
    print(f"Overlap with MPS manifold is {res.fun}")
    return U1, U2, res.x
    
    
def find_env_from_M(M, U2, U2_):
    """
    Our parametrised right environmnet involves finding |M| as a small 
    correction to the Right env :
    
    0   0
    |U2 |
   i|   |
       |M|
   j|   |
    |U2'|
    0   0
                            
    Now we want something that will take M and U and produce that 
    environment for testing purposes:
    """
    
    r = np.einsum(
        U2_,[2,3,5,7],
        M,  [7,6],
        U2, [4,6,0,1],
        [0,1,2,3,4,5])[0,0,0,0,:,:]
    
    return r
    
def test_var_eigs(reps = 100):
    """
    
    Compare how effectively the bounded L-BFGS algorithm works to identify
    the largest eigenvalue of the mixed transfer matrix. It works based on the 
    knowledge that for time evolution over a time dt the right eigenvalue can 
    only change by at most O(dt^2) so we search a region bounded by 
    eta in [1 - k*dt^2, 1] where k is a constant we choose bsaed on
    observation. We note that k = 1 tends to make the region too small
    to contain the eigenvalue so we set k = 5.
    
    Result - this seems to work quite well; over 100 runs we get the eigenvalues 
    differ by an average O(10-4) and the Forbenius norm of the difference
    between the exact right environment and variationally discovered right
    environment is O(10-3)
    
    """
    
    eigs_exact = []
    eigs_var = []    
    r_exact = []
    r_var = []
    for i in range(reps):
        U1 = unitary_group.rvs(4)
        U2 = unitary_group.rvs(4)
        
        # generate random complex matrices
        A1 = np.random.rand(4,4) + 1j*np.random.rand(4,4)
        A2 = np.random.rand(4,4) + 1j*np.random.rand(4,4)
    
        # turn these random matrices into hermitian matrices
        H1 = 0.5 * (A1 + A1.conj().T)
        H2 = 0.5 * (A2 + A2.conj().T)
        
        # generate time evolution operators that are close to the identitiy,
        #   generated from the random hermitian matrices
        dt = 0.01
        dU1 = expm(1j * dt * H1)
        dU2 = expm(1j * dt * H2)
        
        # evololve U1 and U2 to get two matrices close to U1 and U2
        U1_ = U1 @ dU1
        U2_ = U2 @ dU2
        
        # calculate exact eigenvalue of transfer matrix
        eta_e, r_e = env_from_Us_exact(U1.reshape(2,2,2,2), 
                                   U2.reshape(2,2,2,2), 
                                   U1_.conj().T.reshape(2,2,2,2), 
                                   U2_.conj().T.reshape(2,2,2,2))
        eigs_exact.append(eta_e)
        r_exact.append(r_e)
        # Do the saem calculation variationally
        params = env_from_Us_var(U1.reshape(2,2,2,2), 
                        U2.reshape(2,2,2,2), 
                        U1_.conj().T.reshape(2,2,2,2),
                        U2_.conj().T.reshape(2,2,2,2),
                        dt)
        
        eigs_var.append(params[0])
        r_v = R(*params[1:])
        Rv = find_env_from_M(r_v, 
                             U2.reshape(2,2,2,2), 
                             U2_.conj().T.reshape(2,2,2,2))
        
        Rv = Rv / np.linalg.norm(Rv)
        r_var.append(Rv)
        
    diff = np.array(eigs_var) - np.array(eigs_exact)
    print(np.average(diff.real))
    
    plt.hist(diff.real)
    plt.plot()
    
    diff_total = []
    for e,v in zip(r_exact, r_var):
        diff =  np.sqrt(np.sum(np.abs((e.reshape(2,2) - v.reshape(2,2)))**2))
        diff_total.append(diff)
        
    print("The mean Frob distance between right environment is " , sum(diff_total) / 100)
    
    plt.hist(diff_total, 30)
    plt.title("Frob norm diff of env")
    plt.show()
    
    return eigs_exact, eigs_var, r_exact, r_var


def overlapUs(U1, U2, U1_, U2_):
    """
    Calculate the circuit:
        
            
           0   0   0     
           |U2 |   | 
           |   |U1 | 
           |   |   |  
           |   |U1'|      
           |U2'|   |
           0   0   0
           
    to confirm that our parametrisation can effectively compile to random
    unitary matrices.
    
    """
    overlap = np.einsum(
            U2_, [3,4,6,10],
            U1_, [10,5,8,9],
            U1,  [8,9,7,2],
            U2,  [6,7,0,1],
            [0,1,2,3,4,5]
            )[0,0,0,0,0,0]
    
    return overlap

def params_overlaps(params, U1, U2):
    U1_, U2_ = parametrised_Us(params)
    U1_ = U1_.conj().T.reshape(2,2,2,2)
    U2_ = U2_.conj().T.reshape(2,2,2,2)

    return -(np.abs(overlapUs(U1.reshape(2,2,2,2),U2.reshape(2,2,2,2),U1_,U2_))**2)    
    
    
def test_U_parametrisation_effectiveness(reps = 100):
    """
    
    We want to test the effectiveness of this parametrisation of 
    unitary matrices. We specify random unitaries and then use the 
    parametrisation to find a U1' and U2' that give maximal overlap with 
    the initial matrices. In every case the overlap with the 0s was within
    1e-4 of 1.0. However U1' did not compile to U1. I suggest this might
    be because of the freedom to insert a resolution of the identity along the 
    closed legs of the overlap circuit.

    """
    results = []
    U1s = []
    U1_s = []
    for _ in range(reps):
        U1, U2 = unitary_group.rvs(4), unitary_group.rvs(4)
        cost_func = partial(params_overlaps, U1 = U1, U2 = U2)
        grad = partial(approx_fprime, f = cost_func, epsilon = 0.001)
        res = minimize(cost_func, x0 = np.random.rand(18), method = "BFGS", jac = grad)
        
        results.append(res.fun)
        U1s.append(U1)
        U1_s.append(U4(res.x[3:]))
        
    return results, U1s, U1_s


if __name__ =="__main__":
    X0 = X(1)
    Z0 = Z(1)
    I = np.eye(2)
    dt = 0.01
    steps = 100
    
    H = lambda J,g: -J * (np.kron(Z0, Z0) +(g/2)*(np.kron(I,X0) + np.kron(X0,I)))
    W = expm(1j * dt * H(-1,1)).reshape(2,2,2,2)
    U1s = []
    U2s = []
    
    U1,U2 = unitary_group.rvs(4), unitary_group.rvs(4)
    U1s.append(U1)
    U2s.append(U2)
    
    overlap_path = get_tdvp_path()
    initial_guess = np.random.rand(18)
    for _ in range(steps):
        U1_, U2_, new_params = update_U(U1.reshape(2,2,2,2), 
                                        U2.reshape(2,2,2,2), 
                                        W, 
                                        overlap_path, 
                                        dt,
                                        initial_guess )
        U1s.append(U1_)
        U2s.append(U2_)
        initial_guess = new_params
        U1 = U1_
        U2 = U2_
        
    