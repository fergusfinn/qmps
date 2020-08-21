#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 09:57:23 2020

@author: jamie
"""
import matplotlib.pyplot as plt
from qmps.ground_state import NonSparseFullTwoSiteEnergyOptimizer as TSMPS, Hamiltonian, NonSparseFullEnergyOptimizer as MPS
from ClassicalTDVPStripped import Optimizer
from BrickWallMPS import optimize_2layer_bwmps
import numpy as np
from xmps.spin import SU

def compare_gs_einsum():
    H = Hamiltonian({"ZZ":1, "X":1}).to_matrix()    
    O = TSMPS(H)
    mps_res = O.optimize()
    
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
    

def compare_gs():
    H = Hamiltonian({"ZZ":1, "X":1}).to_matrix()    

    O_bw = optimize_2layer_bwmps(H)
    
    x2 = range(len(O_bw))
    y2 = O_bw

    O = TSMPS(H)
    mps_res = O.optimize()
    
    x1 = range(O.iters)
    y1 = O.obj_fun_values
    

    plt.figure()
    plt.plot(x1,y1, label = "qMPS")
    plt.plot(x2,y2, label = "bwMPS")
    plt.legend()
  
    
if __name__ == "__main__":
    compare_gs()