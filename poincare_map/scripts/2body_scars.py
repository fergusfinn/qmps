from numpy import kron
import numpy as np
from functools import reduce
from scipy.linalg import expm
from qmps.represent import Tensor
from qmps.tools import tensor_to_unitary
import cirq
from qmps.time_evolve_tools import merge, put_env_on_left_site, put_env_on_right_site
from xmps.iMPS import Map
from scipy.optimize import minimize
import tqdm as tq
import matplotlib.pyplot as plt
from joblib import Parallel, delayed 
import yaml

def multi_tensor(Ops):
    return reduce(kron, Ops)

### Exact hamiltonian simulation for prof of principle ###
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
        #π = np.pi
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
    
    A1 = A(θ1, ϕ1)    # A(θ1, ϕ1)
    A2 = A(θ2, ϕ2)    # A(θ2, ϕ2)
    A1_= A(θ1_, ϕ1_)   # A(θ1_, ϕ1_)
    A2_= A(θ2_, ϕ2_) # A(θ2_, ϕ2_)
    
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
    sim = cirq.Simulator(dtype = np.complex128)
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

def simulate_scars(initial_params, params):
    dt, timesteps = params
    μ = 0.325
    H = lambda μ:(multi_tensor([I,P,X,P]) + multi_tensor([P,X,P,I])) + (μ/2) * (multi_tensor([I,I,I,n]) +    # I Have multiplied this H by 2, I think I didnt do it correctly
                                                                                multi_tensor([I,I,n,I]) +
                                                                                multi_tensor([I,n,I,I]) + 
                                                                                multi_tensor([n,I,I,I]))

    W = lambda μ, dt: Tensor(expm(1j * dt * H(μ)),'H')
    hamiltonian = W(μ, dt)
    final_params = []
    current_params = initial_params
    for _ in tq.tqdm(range(timesteps)):
        final_params.append(current_params)
        res = minimize(scars_time_evolve_cost_function, current_params, args = (current_params, hamiltonian), options = {'disp':False,'xatol':1e-4, 'fatol':1e-6}, method = 'Nelder-Mead')
        current_params = res.x
        #xatol':1e-6, 'fatol':1e-8, 'adaptive':True
    return np.array(final_params)

def simulate_scars_with_save(initial_params, params):
    dt, timesteps = params
    μ = 0.325
    H = lambda μ:0.5*(multi_tensor([I,P,X,P]) + multi_tensor([P,X,P,I])) + (μ/4) * (multi_tensor([I,I,I,n]) + 
                                                                                multi_tensor([I,I,n,I]) +
                                                                                multi_tensor([I,n,I,I]) + 
                                                                                multi_tensor([n,I,I,I]))

    W = lambda μ, dt: Tensor(expm(1j * dt * H(μ)),'H')
    hamiltonian = W(μ, dt)
    final_params = []
    current_params = initial_params[0]
    ID = initial_params[1]
    for _ in tq.tqdm(range(timesteps)):
        final_params.append(current_params)
        res = minimize(scars_time_evolve_cost_function, current_params, args = (current_params, hamiltonian), options = {'disp':False,'xatol':1e-10, 'fatol':1e-10}, method = 'Nelder-Mead')
        current_params = res.x
        #xatol':1e-6, 'fatol':1e-8, 'adaptive':True
        
    np.savetxt(f"./1D_results_{ID}.txt", np.array(final_params))
    return np.array(final_params)

# -------------------------------------------------------------------------------

################################################################################
# Classical Solver
################################################################################
from numpy import sin, cos, tan, arcsin
from scipy.integrate import solve_ivp



def ode_solver(init_angles, t_final, t_eval = None):
    dθdt = lambda θ1, ϕ1, ϕ2, θ2: tan(θ2)*sin(θ1)*(cos(θ1)**2)*cos(ϕ1) + cos(θ2)*cos(ϕ2)
    dϕdt = lambda θ1, ϕ1, ϕ2, θ2: 2*tan(θ1)*cos(θ2)*sin(ϕ2) - 0.5*tan(θ2)*cos(θ1)*sin(ϕ1)*(2*(sin(θ2)**-2) + cos(2*θ1) -5) 

    def func_list(t, angles):
        return[dθdt(*angles), -0.325 + dϕdt(*angles), -0.325 + dϕdt(*reversed(angles)), dθdt(*reversed(angles))]
    
    y0 = init_angles
    # Radau method needed to see the nice spirals.
    # Interestingly the method used massively changes the height of this map.
    # Radau fixes the height to a narrow range but the standard method has the
    # height vary across the whole range.     
    return solve_ivp(func_list,(0,t_final), y0,t_eval= t_eval, method = "Radau")


# ------------------------------------------------------------------------------


#################################################################################
# Poincare Map Code
#################################################################################
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

def find_crossing_points(θ2s):
    sign_of_angle = np.sign(θ2s)
    sign_diff = np.diff(sign_of_angle)
    sign_change_location = np.argwhere(sign_diff == 2)
    return sign_change_location

def centre_plane(θ2s):
    return np.mod(θ2s - 0.9 + np.pi, 2*np.pi) - np.pi 

def find_zero(x,t,f1):
    # x is the crossing location
    sol = root_scalar(f1, bracket=(t[x], t[x+1]))
    return sol.root

def interpolate_functions(x, t, results):
    mod_results = centre_plane(results[:,0])
    if x > 5:
        f1 = interp1d(t[x-5:x+5], mod_results[x-5:x+5], kind='cubic')
        f2 = interp1d(t[x-5:x+5], results[x-5:x+5,1], kind='cubic')
        f3 = interp1d(t[x-5:x+5], results[x-5:x+5,2], kind='cubic')
        f4 = interp1d(t[x-5:x+5], results[x-5:x+5,3], kind='cubic')
    else:
        f1 = interp1d(t[x-1:x+2], mod_results[x-1:x+2], kind='linear')
        f2 = interp1d(t[x-1:x+2], results[x-1:x+2,1], kind='linear')
        f3 = interp1d(t[x-1:x+2], results[x-1:x+2,2], kind='linear')
        f4 = interp1d(t[x-1:x+2], results[x-1:x+2,3], kind='linear')

    return f1, f2, f3, f4
        
        
def plot_map_variable(angles, show = True):
    cm = plt.cm.twilight_shifted
    # plt.gca().set_prop_cycle(plt.cycler('color', cm(np.linspace(0, 1, len(angles)))))
    allphi1 = []
    allphi2 = []
    allthe2 = []
    for solution in tq.tqdm(angles):
        t = solution.t
        results = solution.y.T
        mod_θ2 = centre_plane(results[:,0])
        x_points = find_crossing_points(mod_θ2)
        ϕ1s = []
        ϕ2s = []
        θ2s = []
        for x in x_points:
            if x[0] <= 1: continue;
            f1,f2,f3,f4 = interpolate_functions(x[0],t,results)
            t0 = find_zero(x[0],t,f1)
            ϕ1_0 = f2(t0)
            ϕ2_0 = f3(t0)
            θ2_0 = f4(t0)

            ϕ1s.append(ϕ1_0)
            allphi1.append(ϕ1_0)
            
            ϕ2s.append(ϕ2_0)
            allphi2.append(ϕ2_0)

            θ2s.append(θ2_0)
            allthe2.append(θ2_0)
        if show:
            plt.figure()
            plt.ylim(4,6)
            plt.scatter(np.mod(np.array(ϕ1s),np.pi*2), np.mod(np.array(θ2s),np.pi*2), c = np.mod(ϕ2s, 2*np.pi),s=0.5,cmap = cm)
    if show:
        plt.colorbar()
        #plt.show()
        plt.savefig("exact_1D_0th_order.png")
    else:
        return np.column_stack(([0.9]*len(allphi1), allphi1, allphi2, allthe2))


def plot_map_by_number(angles, show = True, argsort = False):
    cm = plt.cm.viridis
    plt.gca().set_prop_cycle(plt.cycler('color',cm(np.linspace(0,1,len(angles)))))
    for i, solution in tq.tqdm(enumerate(angles)):
        t = solution.t
        results = solution.y.T
        mod_θ2 = centre_plane(results[:,0])
        x_points = find_crossing_points(mod_θ2)
        ϕ1s = []
        ϕ2s = []
        θ2s = []
        for x in x_points:
            if x[0] <= 1: continue;
            f1,f2,f3,f4 = interpolate_functions(x[0],t,results)
            t0 = find_zero(x[0],t,f1)
            ϕ1_0 = f2(t0)
            ϕ2_0 = f3(t0)
            θ2_0 = f4(t0)

            ϕ1s.append(ϕ1_0)
            ϕ2s.append(ϕ2_0)
            θ2s.append(θ2_0)
        
        if argsort:
            phi1_index = np.mod(np.array(ϕ1s), 2*np.pi).argsort()
            plt.plot(np.mod(np.array(ϕ1s)[phi1_index],np.pi*2), np.mod(np.array(θ2s)[phi1_index],np.pi*2),ms=0.5, marker = 'o')
        else:
            plt.scatter(np.mod(np.array(ϕ1s),np.pi*2), np.mod(np.array(θ2s),np.pi*2),s=0.5)
    if show:
        plt.colorbar()
        plt.show()

def plot_map_files(filename, max_iter, show = True, colour = True, flipped = False):
    cm = plt.cm.twilight_shifted
    #plt.gca().set_prop_cycle(plt.cycler('color', cm(np.linspace(0, 1, len(range(max_iter))))))
    t = [i*0.03/4 for i in range(1500)]
    for i in range(1,max_iter):
        try:
            with open(filename + f"{i}.yml") as f:
                data = yaml.load(f, Loader = yaml.Loader)
        except:
            continue
        
        try:
            result = data['data']
        except:
            print(i)

        mod_θ2 = centre_plane(result[:,0])
        x_points = find_crossing_points(mod_θ2)
        ϕ1s = []
        ϕ2s = []
        θ2s = []
        for x in x_points:
            f1,f2,f3,f4 = interpolate_functions(x[0],t,result)
            t0 = find_zero(x[0],t,f1)
            ϕ1_0 = f2(t0)
            ϕ2_0 = f3(t0)
            θ2_0 = f4(t0)

            ϕ1s.append(ϕ1_0)
            ϕ2s.append(ϕ2_0)
            θ2s.append(θ2_0)

        col = np.mod(ϕ2s, 2*np.pi) if colour else None  
        if flipped:
            x = np.array(ϕ1s)
            plt.scatter(np.mod(x + 2*(2*np.pi - x),np.pi*2), np.mod(np.array(θ2s),np.pi*2),c = col,s=0.5, cmap = cm)
        else:
            plt.scatter(1,5.4,s=8, c = 'r', marker = 'x')
            plt.scatter(np.mod(np.array(ϕ1s),np.pi*2), np.mod(np.array(θ2s),np.pi*2),c = col,s=0.5, cmap = cm)
    if show:
        plt.colorbar()
        plt.show()

def plot_map_txt(filename, max_iter, show = True, colour = True):
    cm = plt.cm.twilight_shifted
    #plt.gca().set_prop_cycle(plt.cycler('color', cm(np.linspace(0, 1, len(range(1200))))))
    t = [i*0.1/4 for i in range(1200)]
    for i in range(1,max_iter):
        try:
            result = np.loadtxt(filename+f"{i}.txt")
        except:
            continue

        mod_θ2 = centre_plane(result[:,0])
        x_points = find_crossing_points(mod_θ2)
        ϕ1s = []
        ϕ2s = []
        θ2s = []
        for x in x_points:
            if x[0] == 0: continue;
            f1,f2,f3,f4 = interpolate_functions(x[0],t,result)
            t0 = find_zero(x[0],t,f1)
            ϕ1_0 = f2(t0)
            ϕ2_0 = f3(t0)
            θ2_0 = f4(t0)

            ϕ1s.append(ϕ1_0)
            ϕ2s.append(ϕ2_0)
            θ2s.append(θ2_0)
        
        col = np.mod(ϕ2s, 2*np.pi)  
        plt.scatter(np.mod(np.array(ϕ1s),np.pi*2), 
                    np.mod(np.array(θ2s),np.pi*2),
                    c = col,s=0.5, cmap = cm)
    if show:
        plt.colorbar()
        plt.show()

def find_sin (θ1, ϕ1, θ2, μ): 
    return ((μ/2) * (cos(2*θ1)*cos(2*θ2) -1) - (sin(2*θ2)*sin(ϕ1)*(cos(θ1)**3))) / (sin(2*θ1)*(cos(θ2)**3))

def const_energy_params(linesteps):
    θ2_line = np.arange(0.01, 2*np.pi, linesteps)
    ϕ1_line = np.arange(0.01, 2*np.pi, linesteps)
    x,y = np.meshgrid(ϕ1_line, θ2_line)
    list_of_values = []
    for i in range(len(x)):
        for j in range(len(y)):
            ϕ1 = x[i,j]
            θ2 = y[i,j]
            sinϕ2 = find_sin(0.9, ϕ1, θ2, 0.325)
            if sinϕ2 < -1 or sinϕ2 > 1:
                continue
            else:
                list_of_values.append([0.9, ϕ1, arcsin(sinϕ2), θ2])
                
    return list_of_values

def const_energy_params_1D(linesteps):
    ϕ1_line = np.arange(0.01, 2*np.pi, linesteps)
    list_of_values = []
    for ϕ1 in ϕ1_line:
            sinϕ2 = find_sin(0.9, ϕ1, 5.41, 0.325) # pick theta2 = 5.41, as this is a good section on this plot
            if sinϕ2 < -1 or sinϕ2 > 1:
                continue
            else:
                list_of_values.append([0.9, ϕ1, arcsin(sinϕ2), 5.41])
                
    return list_of_values


def simulate_params(list_of_values, dt, timesteps):
    total_time = dt/4*timesteps
    t = [dt/4*i for i in range(1,timesteps)]
    list_of_angles = Parallel(n_jobs=-1)(delayed(ode_solver)(i,total_time, t) for i in tq.tqdm(list_of_values))
    return list_of_angles

def const_energy_simulation(dt, timesteps, linesteps, dim_1 = False):
    if dim_1:
        params = const_energy_params_1D(linesteps)
    else:
        params = const_energy_params(linesteps)
    return simulate_params(params, dt, timesteps)

def run_single_fixed_times(dt, timesteps, quantum = True):
    print("\n", dt, timesteps, "\n")
    total_time = dt * timesteps
    initial_params = [0.9, 0.01, arcsin(find_sin(0.9, 0.01, 0.01, 0.325)), 0.01]
    # t_q = [i*dt for i in range(timesteps)]
    t_c = [i*dt/2 for i in range(timesteps)]
    if quantum:
        q_angles = simulate_scars(initial_params, [dt, timesteps])
    
    c_angles = ode_solver(initial_params, t_eval = t_c, t_final = total_time/2)

    fig, axes = plt.subplots(ncols=2, nrows = 2, sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        if quantum:
            ax.plot(np.mod(q_angles[:,i], 2*np.pi), 'b')
        ax.plot(np.mod(c_angles.y.T[:,i], 2*np.pi), 'r')
    
    plt.show()

def plot_single_from_file(n, filename):
    with open(filename + f"{n}.yml") as f:
        data = yaml.load(f, Loader = yaml.Loader)
    
    results = data['data']
    fig, axis = plt.subplots(ncols = 2, nrows = 2, sharex = True, sharey = True)
    for i, ax in enumerate(axis.flatten()):
        ax.plot(np.mod(results[:,i], 2*np.pi), 'b')

def energy(angles):
    
    th1, ph1, ph2, th2 = angles
    mu = 0.325
    denom = cos(th1)**2 + (cos(th2)*sin(th2))**2
    num1 = sin(2*th2)*sin(ph1)*(cos(th1)**3)
    num2 = sin(2*th1)*sin(ph2)*(cos(th2)**3)
    num3 = mu*(1-(cos(2*th1)*cos(2*th2)))/2
    
    return (num1 + num2 + num3)/denom

def apply_energy(results):
    return np.apply_along_axis(energy, 1, results)

def energy_of_run(n):
    with open(f"./2body_output/output/results_{n}.yml") as f:
        data = yaml.load(f, Loader = yaml.Loader)
        
    results = data['data']
    return apply_energy(results)

def plot_random_energies():
    fig, axes = plt.subplots(ncols = 5, nrows = 5, sharex = True, sharey = True)
    
    try_plot = lambda ax: ax.plot(energy_of_run(np.random.randint(1,100)))
        
    for ax in axes.flatten():
        ax.set_ylim(0, 100)
        success = False
        while success == False:
            try:
                try_plot(ax)
                success = True
            except:
                pass

def plot_energies_on_plane(folder_loc):
    """
    Plot Poincare 3D surface on a 2D plane with the colour of the plane defined
    by the energy of that point. 
    """
    cm = plt.cm.viridis
    plt.gca().set_prop_cycle(plt.cycler('color', cm(np.linspace(0, 1, len(range(470))))))
    t = [0.1*i for i in range(1200)]
    for i in range(1,550):
        try:
            with open(folder_loc + f"results_{i}.yml") as f:
                data = yaml.load(f, Loader = yaml.Loader)
        except:
            continue

        result = data['data']
        energies = apply_energy(result)
        mod_θ2 = centre_plane(result[:,0])
        x_points = find_crossing_points(mod_θ2)
        es = []
        ϕ1s = []
        ϕ2s = []
        θ2s = []
        for x in x_points:
            f1,f2,f3,f4 = interpolate_functions(x[0],t,result)
            t0 = find_zero(x[0],t,f1)
            ϕ1_0 = f2(t0)
            ϕ2_0 = f3(t0)
            θ2_0 = f4(t0)
            
            es.append(energies[x[0]])
            ϕ1s.append(ϕ1_0)
            ϕ2s.append(ϕ2_0)
            θ2s.append(θ2_0)

        plt.scatter(np.mod(np.array(ϕ1s),np.pi*2), 
                    np.mod(np.array(θ2s),np.pi*2),
                    c = es, s=0.5)
    plt.colorbar()
    plt.show()


def select_energy_starting_points(filename, max_iter, tol, loc = False, txt = False):
    # Select locs for debugging purposes 
    # to see where in the evolution the energy is conserved.
    nsp = np.empty((0,4)) # new starting points
    av_locs = [] # average locations of points with favourable energy tolerances
    missed_files = 0
    for i in range(1,max_iter):
        try:
            if txt:
                angles = np.loadtxt(filename + f"{i}.txt")
            else:
                with open(filename + f"{i}.yml") as f:
                    data = yaml.load(f, Loader = yaml.Loader)
                    angles = data['data']
        except:
            missed_files+=1
            if missed_files == max_iter-1:
                raise FileNotFoundError
            else:
                continue
            
        energy_of_angles = apply_energy(angles)
        within_tol = np.abs(energy_of_angles[-1]) < tol
        if within_tol:
            nsp = np.concatenate((nsp,angles[-1:,:]))

        if loc:
            where_within_tol = np.argwhere(within_tol)
            if len(where_within_tol) > 0:    
                av_loc = np.average(where_within_tol)
                av_locs.append(av_loc)
    
    if loc:
        return nsp, av_locs 
    else:
        return nsp

def second_order_sim(times):
    # times = [dt, timesteps]
    params = np.loadtxt("./1Dparams.txt")
    list_of_params = []
    ID = 1
    for row in params:
        list_of_params.append([row,ID])
        ID += 1
    
    list_of_angles = Parallel(n_jobs=-1)(delayed(simulate_scars_with_save)(i,times) for i in tq.tqdm(list_of_params))
    return list_of_angles

def post_select_energies(values, tol):
    new_conds = np.empty((0,4))
    for sol in values:
        es = apply_energy(sol.y.T)
        within_tol = np.abs(es) < tol
        where_within_tol = np.argwhere(within_tol).flatten()
        new_conds = np.concatenate((new_conds, sol.y.T[where_within_tol,:])) 
    
    print(new_conds.shape)
    return new_conds

def post_select_energies_eor(values, tol):
    new_conds = np.empty((0,4))
    for sol in values:
        es = apply_energy(sol.y.T)
        within_tol = np.abs(es[-1]) < tol
        if within_tol:
            new_conds = np.concatenate((new_conds, sol.y.T[-1:,:])) 
    
    return new_conds

def plot_energy_map(angles, show = True):
    cm = plt.cm.viridis
    plt.gca().set_prop_cycle(plt.cycler('color', cm(np.linspace(0, 1, len(angles)))))
    for solution in tq.tqdm(angles):
        t = solution.t
        results = solution.y.T
        mod_θ2 = centre_plane(results[:,0])
        x_points = find_crossing_points(mod_θ2)
        ϕ1s = []
        ϕ2s = []
        θ2s = []
        es = []
        for x in x_points:
            if x[0] <= 1: continue;
            f1,f2,f3,f4 = interpolate_functions(x[0],t,results)
            t0 = find_zero(x[0],t,f1)
            ϕ1_0 = f2(t0)
            ϕ2_0 = f3(t0)
            θ2_0 = f4(t0)

            ϕ1s.append(ϕ1_0)
            ϕ2s.append(ϕ2_0)
            θ2s.append(θ2_0)
            es.append(energy([0.9, ϕ1_0, ϕ2_0, θ2_0]))
        plt.scatter(np.mod(np.array(ϕ1s),np.pi*2), np.mod(np.array(θ2s),np.pi*2), c = es,s=0.5)
    if show:
        plt.colorbar()
        plt.show()

def save_high_order_params(filename, max_iter, tol, savename,txt):
    nsp = select_energy_starting_points(filename, max_iter, tol,txt=txt)
    param_string = ""
    for row in nsp:
        param_string += f"{row[0]} {row[1]} {row[2]} {row[3]}\n"
        
    with open(savename, 'w') as f:
        f.write(param_string)
        
 
def poincare_section(angles, t):
    phi1 = []
    phi2 = []
    theta2 = []
    
    # shift the angles theta1 so that the sign changes as it crosses the plane
    #   at theta1 = 0.9 
    mod_theta1 = centre_plane(angles[:,0])
    
    # find the points where the sign of mod_theta1 change
    x_points = find_crossing_points(mod_theta1)
    
    for x in x_points:
        # find the functions that approximate the angle evolution near the
        #   crossing point using function interpolation
        f1,f2,f3,f4 = interpolate_functions(x[0], t, angles)
        
        # find approximate crossing time using root finding algorithms on 
        #   approximate interpolated functions
        t0 = find_zero(x[0], t, f1)
        
        # put this crossing time in to the other interpolated functions to
        #   find their approximate values at the time of crossing
        phi1.append(f2(t0))
        phi2.append(f3(t0))
        theta2.append(f4(t0))
        
        # We know the crossing point of theta1 is approx 0.9. Return the 4 
        #   values in a 2x2 matrix.
    
    phi1 = np.mod(phi1, 2*np.pi)
    phi2 = np.mod(phi2, 2*np.pi)
    theta2 = np.mod(theta2, 2*np.pi)

    return np.column_stack(([0.9]*len(phi1), phi1, phi2, theta2))


       
if __name__ == "__main__":
    
    angles_1d = const_energy_simulation(0.1, 2000, 0.3, True)        
    plot_map_variable(angles_1d, show = True)
    
