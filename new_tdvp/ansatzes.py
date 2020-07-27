#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 18:36:43 2020

@author: jamie
"""
from ClassicalTDVPStripped import CircuitSolver, Represent
from scipy.stats import unitary_group
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize, approx_fprime
from functools import partial 

def random_RE(dt):
    U1 = unitary_group.rvs(4)
    U2 = unitary_group.rvs(4)
    
    A1 = np.random.rand(4,4)
    A2 = np.random.rand(4,4)
    H1 = 0.5 * (A1 + A1.conj().T)
    rand_evo1 = expm(1j * H1 * dt)
    
    H2 = 0.5 * (A2 + A2.conj().T)
    rand_evo2 = expm(1j * H2 * dt)

    
    U1_ = (U1 @ rand_evo1).conj().T
    U2_ = (U2 @ rand_evo2).conj().T
    
    RE = Represent()
    
    Mexact = RE.exact_env(U1.reshape(2,2,2,2), U2.reshape(2,2,2,2), U1_.reshape(2,2,2,2), U2_.reshape(2,2,2,2))
    return np.array(Mexact)


def compile_ansatz_cost_function(params, exact):
    return np.linalg.norm( exact - CircuitSolver().M(params) )


def gradient_descent(cf, gf, init, lr = 0.1, tol = 1e-6, miter = 10000, atol = 1e-4):
    converged = False
    
    iter_ = 0
    v0 = cf(init)
    th0 = np.array(init)

    while not converged:
        th1 = th0 - lr * gf(th0)   
        
        v1 = cf(th1)
        
        if np.abs(v0) < atol:
            return {"X": th0, "V": v0, "M": "Tolerance Reached", "N":iter_}

        
        if np.abs(v1) < atol:
            return {"X": th1, "V": v1, "M": "Tolerance Reached", "N":iter_}

        if v1 > v0:
            lr /= 2
            iter_ += 1
            continue

        if np.abs(v0 - v1) < tol:
            return {"X": th1, "V": v1, "M": "Answers Converged", "N":iter_}
        
        if iter_ == miter:
            return {"X": th1, "V": v1, "M": "Max Iter Reached", "N": miter}
        
        th0 = th1
        v0 = cf(th0)
        
        iter_ += 1
        
def compile_ansatz(cost_function, grad):
    
    res = minimize(cost_function, 
                   x0 = [np.pi/4,0,0,0,0,0],
                   method = "Newton-CG",
                   jac = grad,
                   options = {"maxiter": 10000})
    
    return res


def compile_ansatz_sgd(cf, grad):
    res = gradient_descent(cf, grad, np.array([np.pi/4,0,0,0,0,0]))

    return res


def compilation_vs_time():
    for dt in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        Me = random_RE(dt)    
        
        cf = partial(compile_ansatz_cost_function, exact = Me)
        grad = partial(approx_fprime, f = cf, epsilon = 1e-8)
        res = compile_ansatz(cf, grad)
        
        M_c = CircuitSolver().M(res.x)
        
        # assert np.allclose(np.linalg.norm(M_c), 1)
        
        print("Time Step = ", dt)
        
        print("\n")
            
        print(res.message)
        
        print("\n")
        
        print("Cost Function: ", res.fun)
        
        print("\n")
        
        print("Number Evaluations: ", res.nfev)
        
        print("\n")
        
        print("Params: ", res.x)
        
        print("\n")
        
        print("Exact:")
        print(Me)
        
        print("\n")
        
        print("Compiled:")
        print(M_c)
        
        print("\n")
        print("################################")


def grad_desc_comp():
    
    for dt in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        Me = random_RE(dt)    
        
        cf = partial(compile_ansatz_cost_function, exact = Me)
        grad = partial(approx_fprime, f = cf, epsilon = 1e-8)
        res = compile_ansatz_sgd(cf, grad)
        
        M_c = CircuitSolver().M(res["X"])
        
        # assert np.allclose(np.linalg.norm(M_c), 1)
        
        print("Time Step = ", dt)
        
        print("\n")
            
        print(res["M"])
        
        print("\n")
        
        print("Cost Function: ", res["V"])
        
        print("\n")
        
        print("Number Evaluations: ", res["N"])
        
        print("\n")
        
        print("Params: ", res["X"])
        
        print("\n")
        
        print("Exact:")
        print(Me)
        
        print("\n")
        
        print("Compiled:")
        print(M_c)
        
        print("\n")
        print("################################")


if __name__ == "__main__":
    
    grad_desc_comp()
    
    



