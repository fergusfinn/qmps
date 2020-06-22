# -*- coding: utf-8 -*-
import numpy as np
import yaml
from numpy import kron
from functools import reduce
from scipy.linalg import expm
import cirq
from qmps.tools import tensor_to_unitary
from scipy.optimize import minimize
from argparse import ArgumentParser
## Things I will need to install locally on Myriad ##
from qmps.represent import Tensor
from qmps.time_evolve_tools import merge, put_env_on_left_site, put_env_on_right_site
from xmps.iMPS import Map

def multi_tensor(Ops):
    return reduce(kron, Ops)

P = np.array([[0,0],[0,1]])
X = np.array([[0,1],[1,0]])
n = np.array([[1,0],[0,0]])
I = np.eye(2)

class ScarsAnsatz(cirq.Gate):
    def __init__(self, params):
        self.params = params # this order: [θ, ϕ]
        
    def num_qubits(self):
        return 2
    
    def _decompose_(self, qubits):
        q = qubits
        pi = np.pi
        theta, phi = self.params
        return [
            cirq.ZPowGate(exponent=1/2 - phi/pi).on(q[1]),
            cirq.X.on(q[0]),
            cirq.CNOT(q[0], q[1]),
            cirq.X.on(q[0]),
            cirq.CNotPowGate(exponent=2*theta/pi).on(q[1], q[0]),  # global_shift is needed to remove erronous complex numbers
            cirq.S.on(q[0]),
            cirq.ZPowGate(exponent=-theta/pi).on(q[1])
        ]

class ScarGate(cirq.Gate):
    def __init__(self, params):
        self.params = params # this order: [θ, ϕ, ϕ', θ']

    def num_qubits(self):
        return 3

    def _decompose_(self, qubits):
        q = qubits
        theta, phi, phi_, theta_ = self.params
        return [
            ScarsAnsatz([theta_, phi_]).on(*q[1:3]),
            ScarsAnsatz([theta, phi]).on(*q[0:2])
        ]
    
    def _circuit_diagram_info_(self, args):
        return ['U']*self.num_qubits()

A = lambda theta, phi: np.array([[[0, 1j*np.exp(-1j*phi)], 
                            [0,0]],
                           [[np.cos(theta),0],
                            [np.sin(theta), 0]]])

def scars_cost_fun_alternate(params, current_params, ham):
    '''
    This cost function doesn't use the quantum circuit parameterisation
    
    params are formatted like: [θ1, ϕ1, ϕ2, θ2], for convenience with the classical differential eqn solver
    '''    
    θ1, ϕ1, ϕ2, θ2 = current_params
    θ1_, ϕ1_, ϕ2_, θ2_ = params
    
    A1 = A(θ1, ϕ1)
    A2 = A(θ2, ϕ2)
    A1_= A(θ1_, ϕ1_)
    A2_= A(θ2_, ϕ2_)
    
    A12 = merge(A1,A2)
    A12_= merge(A1_,A2_)
    
    _, r = Map(A12, A12_).right_fixed_point()
    R = Tensor(put_env_on_left_site(r), 'R')
    L = Tensor(put_env_on_right_site(r.conj().T),'L')
    
    U12 = Tensor(tensor_to_unitary(A12),'U')
    U12_= Tensor(tensor_to_unitary(A12_),'U\'')
    
    q = cirq.LineQubit.range(8)
    circuit = cirq.Circuit.from_ops([
        cirq.H(q[5]),
        cirq.CNOT(q[5],q[6]),
        U12(*q[3:6]),
        U12(*q[1:4]),
        L(*q[0:2]),
        ham(*q[2:6]),
        R(*q[6:8]),
        cirq.inverse(U12_(*q[1:4])),
        cirq.inverse(U12_(*q[3:6])),
        cirq.CNOT(q[5],q[6]),
        cirq.H(q[5])
    ])
    
    # print(circuit.to_text_diagram(transpose = True))
    sim = cirq.Simulator(dtype=np.complex128)
    ψ = sim.simulate(circuit).final_state[0]
    return -np.abs(ψ)*2


def scars_time_evolve_cost_function(params, current_params, ham):
    '''
    params are formatted like: [theta1, phi1, phi2, theta2], for convenience with the classical differential eqn solver
    '''    
    theta1, phi1, phi2, theta2 = current_params
    theta1_, phi1_, phi2_, theta2_ = params
    
    A1 = A(theta1, phi1)
    A2 = A(theta2,phi2)
    A1_= A(theta1_, phi1_)
    A2_= A(theta2_, phi2_)
    
    _, r = Map(merge(A1,A2), merge(A1_,A2_)).right_fixed_point()
    R = Tensor(put_env_on_left_site(r), 'R')
    L = Tensor(put_env_on_right_site(r.conj().T),'L')
    
    U12 = ScarGate(current_params)
    U12_= ScarGate(params)
    q = cirq.LineQubit.range(8)
    circuit = cirq.Circuit.from_ops([
        cirq.H(q[5]),
        cirq.CNOT(q[5],q[6]),
        U12(*q[3:6]),
        U12(*q[1:4]),
        L(*q[0:2]),
        ham(*q[2:6]),
        R(*q[6:8]),
        cirq.inverse(U12_(*q[1:4])),
        cirq.inverse(U12_(*q[3:6])),
        cirq.CNOT(q[5],q[6]),
        cirq.H(q[5])
    ])
    
    # print(circuit.to_text_diagram(transpose = True))
    sim = cirq.Simulator(dtype=np.complex128)
    psi = sim.simulate(circuit).final_state[0]
    return -np.abs(psi)*2


def simulate_scars(initial_params, params):
    dt, timesteps = params
    H = lambda mu:(multi_tensor([I,P,X,P]) + multi_tensor([P,X,P,I])) + (mu/2) * (multi_tensor([I,I,I,n]) + multi_tensor([I,I,n,I]) + multi_tensor([I,n,I,I]) + multi_tensor([n,I,I,I]))

    W = lambda mu, dt: Tensor(expm(1j * dt * H(mu)),'H')

    hamiltonian = W(0.325, dt)
    final_params = []
    current_params = initial_params
    for _ in range(timesteps):
        final_params.append(current_params)
        res = minimize(scars_cost_fun_alternate, current_params, args = (current_params, hamiltonian), options = {'disp':False,'xatol':1e-5, 'fatol':1e-5}, method = 'Nelder-Mead')
        current_params = res.x
    
#     if save_file:
#         np.save(save_file, np.array(final_params))
    
    return np.array(final_params)


if __name__ == "__main__":
    parser = ArgumentParser(description="Quantum Scarring using TDVP Evolution")
    parser.add_argument('th0')
    parser.add_argument('th1')
    parser.add_argument('th2')
    parser.add_argument('th3')
    parser.add_argument('id')
    arguments = parser.parse_args()
    
    steps = 10000
    dt = 0.01
    init_conds = [float(arguments.th0), float(arguments.th1), float(arguments.th2), float(arguments.th3)]
    params = [dt, steps]
    angles = simulate_scars(init_conds, params)
    data = {'data':angles}
    filename = "/home/ucapjmd/Scratch/output/results_exact_conds_" + arguments.id + ".yml"
    with open(filename, 'w') as file:
        yaml.dump(data, file)
        
    
    
