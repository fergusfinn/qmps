#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:05:06 2020

@author: jdborin
"""

import numpy as np
from numpy import kron
from functools import reduce

from scipy.linalg import expm
from qmps.states import Tensor
from qmps.tools import tensor_to_unitary

import cirq
from qmps.time_evolve_tools import merge, put_env_on_left_site, put_env_on_right_site
from xmps.iMPS import Map
from scipy.optimize import minimize
import json

def multi_tensor(Ops):
    return reduce(kron, Ops)

### Exact hamiltonian simulation for prof of principle ###
P = np.array([[0,0],[0,1]])
X = np.array([[0,1],[1,0]])
n = np.array([[1,0],[0,0]])
I = np.eye(2)

def H(μ):
    H1 = multi_tensor([P,X,P,I,I,I])
    H2 = multi_tensor([I,P,X,P,I,I])
    H3 = multi_tensor([I,I,P,X,P,I])
    H4 = multi_tensor([I,I,I,P,X,P])
    H5 = multi_tensor([I,I,I,I,I,n])
    H6 = multi_tensor([I,I,I,I,n,I])
    H7 = multi_tensor([I,I,I,n,I,I])
    H8 = multi_tensor([I,I,n,I,I,I])
    H9 = multi_tensor([I,n,I,I,I,I])
    H10 = multi_tensor([n,I,I,I,I,I])
    
    return (1/4)*(H1 + H2 + H3 + H4) + (μ/6) * (H5 + H6 + H7 + H8 + H9 + H10)

class TriSiteScarsAnsatz(cirq.Gate):
    def __init__(self, params):
        self.params = params # this order: [θ, ϕ], setting ϕ = 0 for all cases
        
    def num_qubits(self):
        return 2
    
    def _decompose_(self, qubits):
        q = qubits
        pi = np.pi
        theta, _ = self.params
        return [
            cirq.ZPowGate(exponent=1/2).on(q[1]),  # would be a ϕ term here but it is set to 0
            cirq.X.on(q[0]),
            cirq.CNOT(q[0], q[1]),
            cirq.X.on(q[0]),
            cirq.CNotPowGate(exponent=2*theta/pi).on(q[1], q[0]),  # global_shift is needed to remove erronous complex numbers
            cirq.S.on(q[0]),
            cirq.ZPowGate(exponent=-theta/pi).on(q[1])
        ]

class ScarGate(cirq.Gate):
    def __init__(self, params):
        self.params = params # this order: [θ1, θ2, θ3]

    def num_qubits(self):
        return 4

    def _decompose_(self, qubits):
        q = qubits
        theta1, theta2, theta3 = self.params
        return [
            ScarsAnsatz([theta3, 0]).on(*q[2:4]),
            ScarsAnsatz([theta2, 0]).on(*q[1:3]),
            ScarsAnsatz([theta1, 0]).on(*q[0:2])
        ]
    
    def _circuit_diagram_info_(self, args):
        return ['U']*self.num_qubits()


def scars_time_evolve_cost_function(params, current_params, ham):
    '''
    params are formatted like: [θ1, θ2, θ3
    '''    
    
    A = lambda theta, phi: np.array([[[0, 1j*np.exp(-1j*phi)], 
                                [0,0]],
                                [[np.cos(theta), 0],
                                [np.sin(theta), 0]]])

    theta1, theta2, theta3 = current_params
    theta1_, theta2_, theta3_ = params
        
    A1 = A(theta1, 0)    
    A2 = A(theta2, 0)    
    A3 = A(theta3, 0)    

    A1_= A(theta1_, 0) 
    A2_= A(theta2_, 0) 
    A3_= A(theta3_, 0) 

    
    _, r = Map(merge(merge(A1,A2),A3), merge(merge(A1_,A2_), A3_).right_fixed_point()
    R = Tensor(put_env_on_left_site(r), 'R')
    L = Tensor(put_env_on_right_site(r.conj().T),'L')
    
    U12 = ScarGate(current_params)
    U12_= ScarGate(params)
    q = cirq.LineQubit.range(10)
    circuit = cirq.Circuit.from_ops([
        cirq.H(q[7]),
        cirq.CNOT(q[7],q[8]),
        U12(*q[4:8]),
        U12(*q[1:5]),
        L(*q[0:2]),
        ham(*q[2:8]),
        R(*q[8:10]),
        cirq.inverse(U12_(*q[1:5])),
        cirq.inverse(U12_(*q[4:8])),
        cirq.CNOT(q[7],q[8]),
        cirq.H(q[7])
    ])
    
    # print(circuit.to_text_diagram(transpose = True))
    sim = cirq.Simulator()
    psi = sim.simulate(circuit).final_state[0]
    return -np.abs(psi)*2


def simulate_scars(initial_params, params):
    dt, timesteps = params
    # μ = 0.325

    W = lambda mu, dt: Tensor(expm(1j * dt * H(mu)),'H')
    hamiltonian = W(0, dt) # set μ = 0 automatically for the three site experiment
    
    final_params = []
    current_params = initial_params
    for _ in range(timesteps):
        final_params.append(np.mod(current_params, 2*np.pi))
        res = minimize(scars_time_evolve_cost_function, current_params, args = (current_params, hamiltonian), options = {'disp':True}, method = 'Nelder-Mead')
        current_params = res.x
    
    return np.array(final_params)

if __name__ == "__main__":
    parser = ArgumentParser(description="3 Site Quantum Scarring using TDVP Evolution")
    parser.add_argument('th1')
    parser.add_argument('th2')
    parser.add_argument('th3')
    parser.add_argument('id')
    arguments = parser.parse_args()
    
    steps = 3000
    dt = 0.1
    init_conds = [float(arguments.th1), float(arguments.th2), float(arguments.th3)]
    params = [dt, steps]
    angles = simulate_scars(init_conds, params)
    save_data = {'data':angles}
    
    with open(f"/home/Scratch/output/3_site_results_{arguments.id}.json", 'w') as file:
        json.dump(save_data, file)

