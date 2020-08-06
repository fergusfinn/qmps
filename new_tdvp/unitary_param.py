#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 08:40:09 2020

@author: jamie
"""

import numpy as np
from numpy import sin, cos, exp
from scipy.optimize import minimize
from scipy.stats import unitary_group

Rx = lambda x: np.array([
                [cos(x / 2), -1j*sin(x / 2)],
                [-1j*sin(x / 2), cos(x / 2)]
                ])

Ry = lambda x: np.array([
                [cos(x / 2), -sin(x / 2)],
                [sin(x / 2), cos(x / 2)]
                ])

Rz = lambda x: np.array([
                [exp(-1j * (x / 2)), 0],
                [0, exp( 1j * (x / 2))]
                ])

X = np.array([
        [0,1],
        [1,0]
    ])

I = np.eye(2)

P0 = np.array([
        [1,0],
        [0,0]
    ])

P1 = np.array([
        [0,0],
        [0,1]
    ])

def CNOT(c, t):
    
    """
    CNOT works the following way:
        
        - c0 = CNOT(0,1):               in1     in2
            c0[out1,out2, in1,in2] =    |       |
                                        x ----- @
                                        |       |
                                      out1    out2
                                      
        - c1 = CNOT(1,0):               in1     in2
            c1[out1,out2, in1,in2] =    |       |
                                        @ ----- x
                                        |       |
                                      out1    out2

    """
    
    if not c and t:    
        return (np.kron(P0, I) + np.kron(P1, X))
        
    else:
        return (np.kron(I, P0) + np.kron(X, P1))

c0 = CNOT(0,1)
c1 = CNOT(1,0)

def U2(p1,p2,p3):
    return Rz(p1) @ Ry(p2) @ Rz(p3)

def U2f(a,b,c,d):
    c1 = exp(1j * (a - (b/2) - (d/2)))
    c2 = exp(1j * (a - (b/2) + (d/2)))
    c3 = exp(1j * (a + (b/2) - (d/2)))
    c4 = exp(1j * (a + (b/2) + (d/2)))

    return np.array([
            [c1 * cos(c / 2), -c2 * sin(c / 2)],
            [c3 * sin(c / 2), c4 * cos(c / 2)]
        ])


def U4State(p):
    """
    This is a parametrisation of a 2 qubit state:
        
        U_00|00> = a|00> + b|01> + c|10> + d|11>
        
        s.t. |a|^2 + |b|^2 + |c|^2 + |d|^2 = 1
        
        
    """

    U = U2f(*p[:4])
    V = U2f(*p[4:8])
    S = np.array([
            [cos(p[8]), 0],
            [0, 1j*sin(p[8])]
        ])
    
    psi = (U @ S @ V).reshape(4)
    return psi

def U4(p):
    """
    Requires 19 parameters
    """
    
    u1 = U2f(*p[:4])
    u2 = U2f(*p[4:8])
    u3 = U2f(*p[8:12])
    u4 = U2f(*p[12:16])
    
    return (np.kron(u3,u4) @ c0) @ np.kron(Ry(p[16]), np.eye(2)) @ (c1 @ np.kron(Ry(p[17]), Rz(p[18]))) @ (c0 @ np.kron(u1, u2))


def test_param(fast = False):
    
    print("Test 1: Single qubit convergence:")
    
    U2_ = unitary_group.rvs(2)
    
    def obj(params):
        return np.linalg.norm(U2_ - U2f(*params))
    
    res = minimize(obj, x0 = np.random.rand(3), method = "Powell")
    assert np.isclose(res.fun, 0)
    
    print("Test 1 passed\n")
    
    print("Test 2: CNOT Gates:")
    
    c0 = CNOT(0,1)
    assert c0[1,1,0,1] == 0.0
    assert c0[1,1,1,0] == 1.0
    assert c0[1,0,1,1] == 1.0
    assert c0[1,0,1,1] == 1.0
    
    c1 = CNOT(1,0)
    assert c1[1,1,0,1] == 1.0
    assert c1[1,1,1,0] == 0.0
    assert c1[1,0,1,1] == 0.0
    assert c1[1,0,1,1] == 0.0


    print("Test 2 Passed\n")
    
    if not fast:
        print("Test 3: Two Qubit Unitary Compilation:\n")
        print("Testing...")
        U4_ = unitary_group.rvs(4)
        
        def obj2(params):
            return np.linalg.norm(U4_ - U4(params).reshape(4,4))
        
        res = minimize(obj2, x0 = np.random.rand(19), method = "Powell")
        
        assert np.isclose(res.fun, 0)
        print("Test 3 Passed\n")
        
    
    print("Test 4: Two qubit state compilation")
    psi = np.random.rand(4) + 1j*np.random.rand(4)
    psi = psi / np.linalg.norm(psi)  
    
    def obj3(p):
        return np.linalg.norm(psi - U4State(p))
    
    res = minimize(obj3, x0 = np.random.rand(7), method = "Powell")
    assert np.isclose(res.fun, 0)
    print("Test 4 Passed")

    
if __name__ == "__main__":
    test_param()    
    

    
    
