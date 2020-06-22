#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:06:24 2020

@author: jamie
"""

import numpy as np
from scipy.linalg import expm
from math import cos, sin
from numpy import pi
from scipy.stats import unitary_group
from cmath import exp
from scipy.optimize import minimize, approx_fprime
from functools import partial
import matplotlib.pyplot as plt

D = lambda theta: 0.5*np.array([
        [cos(theta)**2, sin(theta)**2],
        [sin(theta)**2, cos(theta)**2]
    ])

X = lambda theta: np.array([
        [cos(pi * theta / 2), -1j * sin(pi * theta / 2)],
        [-1j * sin(pi * theta / 2), cos(pi * theta / 2)]
    ])

Z = lambda theta: np.array([        
    [1, 0],
    [0, exp(1j * pi * theta)]
    ])

A = np.random.rand(4,4)
H = 0.5*(A + A.conj().T)

dU = lambda t: expm(1j * H * t)


U1p = lambda t, U1: dU(t) @ U1
U2p = lambda t, U2: dU(t) @ U2

R = lambda a, b, c, d: Z(a) @ X(b) @ Z(d) @ D(c) @ Z(-d) @ X(-b) @ Z(-a)



def right_env_circuit_1(a,b,c,d, U1, U2, U1_, U2_, path):
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

if __name__ == "__main__":
    t = 0.
    a,b,c,d = [0]*4 # np.random.rand(3)
    U1, U2 = unitary_group.rvs(4), unitary_group.rvs(4)
    U1_ = U1p(t, U1) 
    U2_ = U2p(t, U2)
    
    U1 = U1.reshape(2,2,2,2)
    U2 = U2.reshape(2,2,2,2)
    U1_ = U1_.conj().T.reshape(2,2,2,2)
    U2_ = U2_.conj().T.reshape(2,2,2,2)

    path = np.einsum_path(
        U2_, [11,12,10,9],
        U1_, [2,10,4,5],
        R(a,b,c,d), [9,8],
        U1, [4,5,1,3],
        U2, [3,8,6,7],
        [2,1,11,12,6,7],
        optimize = "optimal"
    )[0]
    
    M_ij = right_env_circuit_1(a, b, c, d, U1, U2, U1_, U2_, path)
    print("The difference between R_if and M_if is: \n")
    
    print(R(a,b,c,d) - M_ij)
    print("\nWe can see that the equation is satisfied for the case where U = U'\n")
    
    print("Now check the case for small change, the parameters don't change much from 0:\n")
    
    

    
    params = {"eta":[], "a":[], "b":[], "c":[], "fun":[], "iters":[], "d":[]}
    pows = range(-2,9)
    t = 4
    e = 0.01
    for i in pows:
        np.random.seed(0)
        U1 = unitary_group.rvs(4)
        np.random.seed(1)
        U2 = unitary_group.rvs(4)

        #t = 10**(-i)    
        U1_ = U1p(t, U1) 
        U2_ = U2p(t, U2)
    
        U1 = U1.reshape(2,2,2,2)
        U2 = U2.reshape(2,2,2,2)
        U1_ = U1_.conj().T.reshape(2,2,2,2)
        U2_ = U2_.conj().T.reshape(2,2,2,2)
        
        param_circuit = partial(right_env_circuit_1, U1 = U1, U1_ = U1_, U2 = U2, U2_ = U2_, path = path)
        
        def cost_func(params):
            eta, a, b, c, d   = params
            return np.linalg.norm(eta * R(a,b,c,d) - param_circuit(a,b,c,d))
        
        grad = partial(approx_fprime, f = cost_func, epsilon = e)
        
        res = minimize(cost_func, x0 = [1.0, 0,0,0,0], method = "L-BFGS-B", jac = grad, bounds = [(max([0,1-t**2]), 1.0), (0,2*np.pi), (0,2*np.pi), (0,2*np.pi), (0,2*np.pi)])
        
        params["eta"].append(res.x[0])
        params["a"].append(res.x[1])
        params["b"].append(res.x[2])
        params["c"].append(res.x[3])
        params["d"].append(res.x[4])

        params["fun"].append(res.fun)
        params["iters"].append(res.nit)
        
        t *= 0.5
    
    x = pows
    plt.style.use('default')
    fig, ax1 = plt.subplots()
    ax1.plot(x, params["eta"],"r", label = "$\\eta$")
    ax1.plot(x, params["a"],'b', label = "$\\theta_{1}$")
    ax1.plot(x, params["b"], 'g', label = "$\\theta_{2}$")
    ax1.plot(x, params["c"], 'k', label = "$\\theta_{3}$")
    ax1.plot(x, params["d"], 'm', label = "$\\theta_{4}$")

    ax1.plot(x, params["fun"], "y", label = "fun")
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("iterations")
    ax2.plot(x, params["iters"], "p", label = "iters")
    
    plt.xticks(pows, labels = ["$2^{%d}$" % (-1*i) for i in pows])
    ax1.set_xlabel("dt")
    ax1.set_ylabel("value of parameters and cost function")
    ax1.legend()
    plt.show()
        
    

    
    
