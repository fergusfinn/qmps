#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:34:12 2020

@author: jamie
"""
import numpy as np
from ClassicalTDVPStripped import tensor, Optimizer
from BrickWallMPS import bwMPS, RightEnvironment, LeftEnvironment, paramU, state_from_params, OO_unitary
from scipy.linalg import expm
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from scipy.optimize import minimize
from xmps.spin import U4
from tqdm import tqdm

Pauli = {"X":np.array([[0,1],[1,0]]),
         "Y":np.array([[0,-1j],[1j,0]]),
         "Z":np.array([[1,0],[0,-1]])}

PauliVec = np.array([Pauli["Z"], Pauli["X"], Pauli["X"]])

def HH(J):
    return J * sum([np.kron(s,s) for s in PauliVec])

def r(m):
    return m.reshape(2,2,2,2)


def env_check():
    Op = Optimizer()
    U2 = np.eye(4)
    U1 = unitary_group.rvs(4)
    h = HH(1)
    Ut = expm(1j * h * 0.1)
    U1_ = (Ut @ U1).conj().T
    
    etar, envr = Op.represent.RE.exact_environment(r(U1), r(U2), r(U1_), r(U2))
    etal, envl = Op.represent.LE.exact_environment(r(U1), r(U2), r(U1_), r(U2))

    print(envr, "\n", envl) 
    print(etar, etal)

def evolve_U2():
    hh = HH(1)
    I = np.eye(4)
    h = tensor([I,hh,I])
    Ut = expm(1j * h * 0.01)
    
    U1 = np.eye(4)
    U2 = np.array([
            [0,1,0,0],
            [1,0,0,0],
            [0,0,0,1],
            [0,0,1,0]
        ])
    
    def obj(p, U2, Ut):
        psi1 = bwMPS([U2, np.eye(4)], 3).state()
        U2_ = OO_unitary(p)
        psi2 = bwMPS([U2_, np.eye(4)], 3).state()
        return np.abs(psi2.conj().T @ Ut @ psi1)**2
    
    STEPS = 500
    init_params = np.random.rand(7)
    results = []
    for i in range(STEPS):
        print(i)
        res = minimize(obj, 
                       x0 = init_params, 
                       args = (U2, Ut),
                       method = "Nelder-Mead",
                       tol = 1e-8,
                       options = {"maxiter":20000, 
                                  "disp":True,
                                  "adaptive":True})
        
        init_params = res.x
        U2 = OO_unitary(res.x)
        results.append(res.x)
    
    return results


def evolve_and_plot_U2():
    res = evolve_U2()    
    states = []
    for r_ in res:
        U2 = OO_unitary(r_)
        state = U2 @ np.array([1,0,0,0])
        states.append(state)
    
    plt.figure()
    plt.plot(np.abs(np.array(states)[:,1])**2, label = "|01>")
    plt.plot(np.abs(np.array(states)[:,2])**2, label = "|10>")
    plt.legend()
    plt.show()

def evolve_U1():
    hh = HH(1)
    U1 = np.array([
            [0,1,0,0],
            [1,0,0,0],
            [0,0,0,1],
            [0,0,1,0]
        ])
    
    
    STEPS = 400
    init_params = np.random.rand(15)
    results = []
    LE, RE = LeftEnvironment(), RightEnvironment()
    
    def obj2(p, U1, h_):
        U1_ = U4(p)
        _, Ml = LE.exact_environment(r(U1), r(np.eye(4)), r(U1_.conj().T), r(np.eye(4)))
        _, Mr = RE.exact_environment(r(U1), r(np.eye(4)), r(U1_.conj().T), r(np.eye(4)))
    
        return -2 * np.abs((U1_ @ expm(1j * h_ * 0.01) @ U1)[0,0] * Mr[0,0] * Ml[0,0])**2
    
    for _ in tqdm(range(STEPS)):
        res = minimize(obj2, 
                       x0 = init_params, 
                       args = (U1, hh),
                       method = "Nelder-Mead",
                       tol = 1e-5,
                       options = {"maxiter":40000, 
                                  "disp":True,
                                  "adaptive":True})

        init_params = res.x
        U1 = U4(res.x)
        results.append(res.x)
        
    return results
    

def evolve_and_plot_U1():
    res = evolve_U1()
    states = []
    for r_ in res:
        U2 = U4(r_)
        state = U2 @ np.array([1,0,0,0])
        states.append(state)
    
    plt.figure()
    plt.plot(np.abs(np.array(states)[:,1])**2, label = "|01>")
    plt.plot(np.abs(np.array(states)[:,2])**2, label = "|10>")
    plt.legend()
    plt.show()


def obj(p, U1, h_):
    LE = LeftEnvironment()
    RE = RightEnvironment()
    
    psi1 = bwMPS([np.eye(4), U1], 2).state()
    U1_ = U4(p)
    
    _, Ml = LE.exact_environment(r(U1), r(np.eye(4)), r(U1_.conj().T), r(np.eye(4)))
    _, Mr = RE.exact_environment(r(U1), r(np.eye(4)), r(U1_.conj().T), r(np.eye(4)))
    
    Ut = expm(1j * h_ * 0.01)
    Ut_m = tensor([Ml, Ut, Mr])
    
    psi2 = bwMPS([np.eye(4), U1_], 2).state()
    return np.abs(psi2.conj().T @ Ut_m @ psi1)**2

def two_cost_functions():
    p = np.random.rand(15)
    U1 = np.array([
            [0,1,0,0],
            [1,0,0,0],
            [0,0,0,1],
            [0,0,1,0]
        ])
    h = HH(1)
    
    print(obj(p, U1, h))
    
    U1_ = U4(p).conj().T
    LE, RE = LeftEnvironment(), RightEnvironment()
    _, Ml = LE.exact_environment(r(U1), r(np.eye(4)), r(U1_), r(np.eye(4)))
    _, Mr = RE.exact_environment(r(U1), r(np.eye(4)), r(U1_), r(np.eye(4)))
    print(Ml, "\n", Mr)
    print((U1_ @ expm(1j * h * 0.01) @ U1)[0,0] * Mr[0,0] * Ml[0,0])

if __name__ == "__main__":
    evolve_and_plot_U1()
        
        
        
    
        
        
        
        
        
        
    
        

