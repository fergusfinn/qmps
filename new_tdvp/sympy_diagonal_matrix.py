#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:28:26 2020

@author: jamie
"""

import numpy as np
from scipy.linalg import expm
from sympy import exp, Symbol, simplify, Matrix
from sympy.printing import pprint
from sympy.physics.quantum import TensorProduct
"""

  |i> ----@---- |j>
          |
|0> --Rx--x--Rx^dag-- |0>

The qubit ordering is defined as follows:
    
    |i0> & |j0>
    
    
The cnot performs the following transformation:
    
|00> -> |00>
|01> -> |01>
|10> -> |11>
|11> -> |10>
"""

theta = Symbol('theta')

X = np.array([
        [0,1j],
        [-1j,0]
    ])

CNOT = Matrix(np.array([
        [1,0,0,0],
        [0,0,0,1],
        [0,0,1,0],
        [0,1,0,0]
    ]))

I = Matrix(np.eye(2))
sigmax = Matrix(1j * X * theta)
sigmax_inv = Matrix(-1j * X * theta)


R = simplify(exp(sigmax))
R_ = simplify(exp(sigmax_inv))
IX = simplify(TensorProduct(I, R))
IX_ = simplify(TensorProduct(I,R_))

U = IX_ * CNOT * IX

M_ij = Matrix([
        [U[0,0], U[0,2]],
        [U[2,0], U[2,2]]
    ])

pprint(simplify(M_ij))

