import numpy as np
from numpy import kron
from functools import reduce
from scipy.linalg import expm
from qmps.represent import Tensor, Environment
from qmps.tools import tensor_to_unitary
import cirq
from qmps.time_evolve_tools import merge, put_env_on_left_site, put_env_on_right_site
from xmps.iMPS import Map
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def multi_tensor(Ops):
    return reduce(kron, Ops)

### Exact hamiltonian simulation for prof of principle ###
P = np.array([[0,0],[0,1]])
X = np.array([[0,1],[1,0]])
n = np.array([[1,0],[0,0]])
I = np.eye(2)

H = lambda μ:0.5*(multi_tensor([I,P,X,P]) + multi_tensor([P,X,P,I])) + (μ/4) * (multi_tensor([I,I,I,n]) + 
                                                                                multi_tensor([I,I,n,I]) +
                                                                                multi_tensor([I,n,I,I]) + 
                                                                                multi_tensor([n,I,I,I]))

W = lambda μ, dt: Tensor(expm(1j * dt * H(μ)),'H')

class ScarsAnsatz(cirq.Gate):
    def __init__(self, params):
        self.params = params # this order: [θ, ϕ]
        
    def num_qubits(self):
        return 2
    
    def _decompose_(self, qubits):
        q = qubits
        π = np.pi
        θ, ϕ = self.params
        return [
            cirq.ZPowGate(exponent=1/2 - ϕ/π).on(q[1]),
            cirq.X.on(q[0]),
            cirq.CNOT(q[0], q[1]),
            cirq.X.on(q[0]),
            cirq.CNotPowGate(exponent=2*θ/π).on(q[1], q[0]),  # global_shift is needed to remove erronous complex numbers
            cirq.S.on(q[0]),
            cirq.ZPowGate(exponent=-θ/π).on(q[1])
        ]

class ScarGate(cirq.Gate):
    def __init__(self, params):
        self.params = params # this order: [θ, ϕ, ϕ', θ']

    def num_qubits(self):
        return 3

    def _decompose_(self, qubits):
        q = qubits
        π = np.pi
        θ, ϕ, ϕ_, θ_ = self.params
        return [
            ScarsAnsatz([θ_, ϕ_]).on(*q[1:3]),
            ScarsAnsatz([θ, ϕ]).on(*q[0:2])
        ]
    
    def _circuit_diagram_info_(self, args):
        return ['U']*self.num_qubits()

A = lambda θ, ϕ: np.array([[[0, 1j*np.exp(-1j*ϕ)], 
                            [0,0]],
                           [[np.cos(θ), 0],
                            [np.sin(θ), 0]]])


def scars_time_evolve_cost_function(params, current_params, ham):
    '''
    params are formatted like: [θ1, ϕ1, ϕ2, θ2], for convenience with the classical differential eqn solver
    '''    
    θ1, ϕ1, ϕ2, θ2 = current_params
    θ1_, ϕ1_, ϕ2_, θ2_ = params

    A1 = A(θ1, ϕ1)
    A2 = A(θ2, ϕ2)
    A1_= A(θ1_, ϕ1_)
    A2_= A(θ2_, ϕ2_)
    
    _, r = Map(merge(A1,A2), merge(A1_,A2_)).right_fixed_point()
    R = Environment(put_env_on_left_site(r), 'R')
    L = Environment(put_env_on_right_site(r.conj().T),'L')
    
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
    sim = cirq.Simulator()
    ψ = sim.simulate(circuit).final_state[0]
    return -np.abs(ψ)*2

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
    R = Environment(put_env_on_left_site(r), 'R')
    L = Environment(put_env_on_right_site(r.conj().T),'L')
    
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
    sim = cirq.Simulator()
    ψ = sim.simulate(circuit).final_state[0]
    return -np.abs(ψ)*2

def simulate_scars(dt, timesteps, μ, initial_params, save_file = None):
    hamiltonian = W(μ, dt)
    final_params = []
    current_params = initial_params
    for _ in tqdm(range(timesteps)):
        final_params.append(np.mod(current_params, 2*np.pi))
        res = minimize(scars_time_evolve_cost_function, current_params, args = (current_params, hamiltonian), options = {'disp':True}, method = 'Nelder-Mead')
        current_params = res.x
    
    if save_file:
        np.save(save_file, np.array(final_params))
    
    return np.array(final_params)
    
# Using tensor_to_unitary we almost exactly recreate the exact TDVP differential equation solutions.
from tqdm.notebook import tqdm
from numpy import sin, cos, tan, arcsin, pi
import numpy as np
from scipy.integrate import odeint
dθdt = lambda θ1, ϕ1, ϕ2, θ2: tan(θ2)*sin(θ1)*(cos(θ1)**2)*cos(ϕ1) + cos(θ2)*cos(ϕ2)
dϕdt = lambda θ1, ϕ1, ϕ2, θ2: 2*tan(θ1)*cos(θ2)*sin(ϕ2) - 0.5*tan(θ2)*cos(θ1)*sin(ϕ1)*(2*(sin(θ2)**-2) + 
                                                                                                cos(2*θ1) - 5)

def func_list(angles,t,μ):
    return[dθdt(*angles), -μ + dϕdt(*angles), -μ + dϕdt(*reversed(angles)), dθdt(*reversed(angles))]

if __name__ == "__main__":
    np.random.seed(0)

    y0 = np.random.randn(4)
    steps = 10000
    t = np.linspace(0.0, 500, steps)
    dt = 4 * t[1]-t[0]
    μ = 0.325

    msp =np.empty([1,4]) # multiple_starting_points
    for _ in range(100):
        y0 = np.random.randn(4)
        classical_angles = np.mod(odeint(func_list, y0, t, args=(μ,)), np.pi * 2)
        msp = np.concatenate((msp, classical_angles), axis = 0)
    
    #quantum_angles = simulate_scars(dt, steps, μ, y0, save_file=f'{steps}_{dt}_{μ}_random_initial')
    #file = f'{steps}_{dt}_{μ}_random_initial.npy'
    #quantum_angles = np.load(file)
θ1_c = classical_angles[:,0]
ϕ1_c = classical_angles[:,1]
ϕ2_c = classical_angles[:,2]
θ2_c = classical_angles[:,3]


θ1_q = quantum_angles[:,0]
ϕ1_q = quantum_angles[:,1]
ϕ2_q = quantum_angles[:,2]
θ2_q = quantum_angles[:,3]

plt.figure()
plt.plot(θ1_c[:2000])
plt.plot(θ1_q[:2000])
plt.show()