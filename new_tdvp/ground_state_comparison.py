#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 09:57:23 2020

@author: jamie
"""
import matplotlib.pyplot as plt
from qmps.ground_state import NonSparseFullTwoSiteEnergyOptimizer as TSMPS, Hamiltonian, NonSparseFullEnergyOptimizer as MPS
from ClassicalTDVPStripped import Optimizer
import numpy as np


if __name__ == "__main__":
    H = Hamiltonian({"ZZ":1, "X":1}).to_matrix()
    X = np.array([
            [0,1],
            [1,0]
        ])
    
    HX = np.kron(X,X)
    
    O = TSMPS(H)
    O.optimize()
    
    x2 = range(O.iters)
    y2 = O.obj_fun_values
    
    O_new = Optimizer()
    O_new.optimize.optimize(H.reshape(2,2,2,2))
        
    x1 = range(len(O_new.optimize.energy_opt))
    y1 = O_new.optimize.energy_opt

    
    plt.figure()
    plt.plot(x2,y2, label = "qMPS")
    plt.plot(x1,y1, label = "bwMPS")
    plt.legend()
    