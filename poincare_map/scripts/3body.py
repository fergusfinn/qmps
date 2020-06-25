#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:03:08 2020

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

import tqdm as tq

from numpy import sin, cos
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed

from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

import matplotlib.pyplot as plt
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
    
    return (1/2)*(H1 + H4) + H2 + H3 + (μ/6) * (H5 + H6 + H7 + H8 + H9 + H10)

class ScarsAnsatz(cirq.Gate):
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

    A123 = merge(merge(A1,A2),A3)
    A123_ = merge(merge(A1_,A2_), A3_)
    _, r = Map(A123, A123_).right_fixed_point()
    R = Tensor(put_env_on_left_site(r), 'R')
    L = Tensor(put_env_on_right_site(r.conj().T),'L')
    
    # to go faster do not use ScarGate, use tensor_to_unitary instead:
    #U12 = ScarGate(current_params)
    #U12_= ScarGate(params)
    
    U12 = Tensor(tensor_to_unitary(A123), 'U')
    U12_ = Tensor(tensor_to_unitary(A123_), 'U')
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
    
    #print(circuit.to_text_diagram(transpose = True))
    sim = cirq.Simulator(dtype = np.complex128)
    psi = sim.simulate(circuit).final_state[0]
    return -np.abs(psi)*2


def simulate_scars(initial_params, params):
    dt, timesteps = params
    # μ = 0.325

    W = lambda mu, dt: Tensor(expm(-1j * dt * H(mu)),'H')
    hamiltonian = W(0, dt) # set μ = 0 automatically for the three site experiment
    
    final_params = []
    current_params = initial_params
    for _ in tq.tqdm(range(timesteps)):
        final_params.append(np.mod(current_params, 2*np.pi))
        res = minimize(scars_time_evolve_cost_function, current_params, args = (current_params, hamiltonian), options = {'disp':False, 'xtol':1e-12, 'ftol':1e-12}, method = 'Powell')
        current_params = res.x
    
    return np.array(final_params)


sin2 = lambda x: sin(x)**2
cos2 = lambda x: cos(x)**2
sin3 = lambda x: sin(x)**3
cos3 = lambda x: cos(x)**3

def θ1dot(θ1, θ2, θ3):
    m1 = lambda θ1, θ2, θ3: 4 * (3 + cos(2*θ2) - 2*cos(2*θ3)*sin2(θ2))
    return (4*sin2(θ2)*cos(3*θ3) + 
            2*sin(θ1)*sin(2*θ2)*(-2*cos2(θ1)*cos(2*θ3) + cos(2*θ1) -3 ) - 
            2*cos(θ3)*(3*cos(2*θ2) + 5) ) / m1(θ1, θ2, θ3)

def θ2dot(θ1, θ2, θ3):
    m2 = lambda θ1, θ2, θ3: 64 * (cos2(θ3) + sin2(θ1)*sin2(θ3))
    return (-4*sin(2*θ3)*((cos(2*θ1)+7)*sin(θ2) - 2*sin2(θ1)*sin(3*θ1)) - 
            8*cos(θ1)*(4*cos2(θ1)*cos(2*θ3)+5) + 
            8*cos(3*θ1)) / m2(θ1, θ2, θ3)

def θ3dot(θ1, θ2, θ3):
    m3 = lambda θ1, θ2, θ3: -16*(1 - cos2(θ2)*sin2(θ1))
    return (-2*sin2(θ2)*sin(2*θ1)*sin(3*θ3) + 
            sin(2*θ1)*sin(θ3)*(cos(2*θ2)+7)+
            8*cos(2*θ1)*cos3(θ2) +
            10*cos(θ2) - 
            2*cos(3*θ2)) / m3(θ1, θ2, θ3)



def ode_solver(init_angles, t_eval, t_total):
    
    def func_list(t, angles):
        return[θ1dot(*angles), θ2dot(*angles), θ3dot(*angles)]
    
    def event(t, y):
        return np.mod(y[1] + np.pi, 2*np.pi) - np.pi
    
    event.direction = 1
    
    y0 = init_angles
    return solve_ivp(func_list,(0,t_total), y0, events = event, t_eval= t_eval, method = "Radau")


def simulate_3_site_classical(line_step, steps, dt):
    θ1_line = np.arange(0.01,2*np.pi, line_step)
    θ3_line = np.arange(0.01,2*np.pi, line_step)
    x,y = np.meshgrid(θ1_line, θ3_line)
    list_of_values = []
    for i in range(len(x)):
        for j in range(len(y)):
            list_of_values.append([x[i,j], 0, y[i,j]])

    total_time = dt*steps / 2
    t = np.arange(0,total_time,0.38)
    list_of_angles = Parallel(n_jobs=-1)(delayed(ode_solver)(i, t, total_time) for i in tq.tqdm(list_of_values))
    return list_of_angles

def find_crossing_points(θ2s):
    sign_of_angle = np.sign(θ2s)
    sign_diff = np.diff(sign_of_angle)
    sign_change_location = np.argwhere(sign_diff == 2)
    return sign_change_location

def centre_plane(θ2s):
    return np.mod(θ2s + np.pi, 2*np.pi) - np.pi

def find_zero(x,t,f2):
    # x is the crossing location
    sol = root_scalar(f2, bracket=(t[x], t[x+1]))
    return sol.root

def interpolate_functions(x, t, results):
    mod_results = centre_plane(results[:,1])
    if x > 5:
        f1 = interp1d(t[x-5:x+5], results[x-5:x+5,0], kind='cubic')
        f2 = interp1d(t[x-5:x+5], mod_results[x-5:x+5], kind='cubic')
        f3 = interp1d(t[x-5:x+5], results[x-5:x+5,2], kind='cubic')
    else:
        f1 = interp1d(t[x-1:x+2], results[x-1:x+2,0], kind='linear')
        f2 = interp1d(t[x-1:x+2], mod_results[x-1:x+2], kind='linear')
        f3 = interp1d(t[x-1:x+2], results[x-1:x+2,2], kind='linear')

    return f1, f2, f3

def get_map(results,t):
    mod_θ2 = centre_plane(results[:,1])
    x_points = find_crossing_points(mod_θ2)
    θ1s = []
    θ3s = []
    for x in x_points:
        f1,f2,f3 = interpolate_functions(x[0],t,results)
        t0 = find_zero(x[0],t,f2)
        θ1_0 = f1(t0)
        θ3_0 = f3(t0)

        θ1s.append(θ1_0)
        θ3s.append(θ3_0)

    plt.plot(np.mod(np.array(θ3s),2*np.pi)/np.pi, np.mod(np.array(θ1s),np.pi * 2)/np.pi, '.', ms = 4)

def compare_classical_quantum(init_conds, dt, steps):
    total_time = dt*steps / 2
    t_eval = [i*dt/2 for i in range(steps)]
    classical_angles = ode_solver(init_conds, t_eval, total_time)
    quantum_angles = simulate_scars(init_conds, [dt, steps])
    
    fig, axes = plt.subplots(3,1, sharey = True)
    for i, ax in enumerate(axes):
        ax.plot(np.mod(quantum_angles[:,i], np.pi * 2), 'b')
        ax.plot(np.mod(classical_angles.y.T[:,i], np.pi * 2), 'r')
    plt.show()

def plot_angles(angles):
    fig, axes = plt.subplots(3,1, sharey = True)
    for i, ax in enumerate(axes):
        ax.plot(np.mod(angles[:,i], np.pi*2))
        
    plt.show()

def classical_map(dt, steps, init_conds = [6.01, 0, 0.9099999999999999]):
    t_eval = [i*dt/2 for i in range(steps)]
    total_time = dt*steps / 2
    
    angles = ode_solver(init_conds, t_eval, total_time)
    
    get_map(angles.y.T, t_eval)
    # plt.show()
    
def search_region_around_periodic_orbit(number_points, dt, steps):
    line1 = np.linspace(5.8, 6.2, number_points)
    line2 = np.linspace(0.8,1.2, number_points)
    
    starting_angles = []
    x,y = np.meshgrid(line1, line2)
    for i in range(len(x)):
        for j in range(len(y)):
            starting_angles.append([x[i,j], 0, y[i,j]])
            
    total_time = dt*steps / 2
    t = np.arange(0,total_time,dt/2)
    results = Parallel(n_jobs=-1)(delayed(ode_solver)(i, t, total_time) for i in tq.tqdm(starting_angles))
    
    cm = plt.cm.viridis
    plt.gca().set_prop_cycle(plt.cycler('color', cm(np.linspace(0, 1, len(results)))))

    for r in results:
        get_map(r.y.T, t)
    plt.show()
    
    


if __name__ == "__main__":    
    steps = 160000
    dt = 0.0005
    initial_params = [6.01, 0, 0.909]
    angles = simulate_scars(initial_params, [dt,steps])    
    np.savetxt("160000step_00005.txt", angles)
    
    #W = lambda mu, dt: Tensor(expm(1j * dt * H(mu)),'H')
#scars_time_evolve_cost_function([1,2,3],[4,5,6], W(0,0.1))
