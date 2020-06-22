#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:35:21 2020

@author: jamie
"""

import numpy as np
from scipy.stats import unitary_group
import pickle

if __name__ == "__main__":
    U1,U2,U1_,U2_ = [unitary_group.rvs(4).reshape(2,2,2,2)]*4
    W = unitary_group.rvs(16).reshape(2,2,2,2,2,2,2,2)
    M = unitary_group.rvs(2)
    
    p = np.einsum_path(
        U2_, [6,7,26,27],
        U2_, [8,9,28,29],
        U2_, [10,11,30,31],
        U1_, [27,28,22,23],
        U1_, [29,30,24,25],
        W,[22,23,24,25,18,19,20,21],
        M, [26,12],
        M, [31,17],
        U1, [18,19,13,14],
        U1, [20,21,15,16],
        U2, [12,13,0,1],
        U2, [14,15,2,3],
        U2, [16,17,4,5],
        [0,1,2,3,4,5,6,7,8,9,10,11],
        optimize = "optimal"
        )
    
    path = p[0]
    path_info = p[1]
    
    info = {"path": path, "path_info": path_info}
    
    with open("path_info.pkl", "wb") as f:
        pickle.dump(info, f)
        
    