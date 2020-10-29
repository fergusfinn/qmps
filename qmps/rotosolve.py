import cirq 
from xmps.spin import paulis, U4, swap
import numpy as np
import numpy.random as ra
from qmps.represent import split_2s, ShallowCNOTStateTensor, ShallowFullStateTensor, State, FullStateTensor, FullEnvironment
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from functools import reduce
from scipy.optimize import minimize_scalar
from xmps.tensor import partial_trace
from scipy.linalg import expm


def gate(v, symbol='U'):
    #return ShallowCNOTStateTensor(2, v[:-1])
    return ShallowFullStateTensor(2, v, symbol)
    #return FullStateTensor(U4(v))

def op_state(params, which='energy'):
    assert len(params)==30
    p2, p1 = np.split(params, 2)
    if which=='energy':
        qbs = cirq.LineQubit.range(4)
        C = cirq.Circuit.from_ops([gate(p1)(*qbs[2:]),
                                   gate(p2)(*qbs[1:3]),
                                   gate(p2)(*qbs[:2])])
        s = cirq.Simulator()
        return s.simulate(C).final_state
    elif which=='v_purity':
        qbs = [[cirq.GridQubit(y, x) for x in range(2)] 
                for y in range(2)]
        C = cirq.Circuit.from_ops([gate(p1)(*qbs[0][:2]),
                                   gate(p1)(*qbs[1][:2]), 
                                   cirq.SWAP(*qbs[0][:2])])
        s = cirq.Simulator()
        return s.simulate(C).final_state
    elif which=='u_purity':
        qbs = [[cirq.GridQubit(y, x) for x in range(3)]
                for y in range(2)]
        C = cirq.Circuit.from_ops([gate(p1)(*qbs[0][1:]), 
                                   gate(p2)(*qbs[0][:2]),
                                   gate(p1)(*qbs[1][1:]), 
                                   gate(p2)(*qbs[1][:2]), 
                                   cirq.SWAP(*qbs[0][:2]), 
                                   cirq.SWAP(*qbs[0][1:])])
        s = cirq.Simulator()
        return s.simulate(C).final_state
    elif which=='uv_purity':
        qbs = cirq.LineQubit.range(5)
        C = cirq.Circuit.from_ops([gate(p1)(*qbs[3:]), 
                                   gate(p2)(*qbs[2:4]), 
                                   gate(p1)(*qbs[:2]), 
                                   cirq.SWAP(*qbs[:2])])
        s = cirq.Simulator()
        return s.simulate(C).final_state
                                   
def op_H(H):
    #H = np.eye(4)
    return reduce(np.kron, [np.eye(2), H, np.eye(2)])

def evo_state(params, old_U, H, dt, which='eig'):
    assert len(params)==30
    p, env_p = np.split(params, 2)


    s = cirq.Simulator()
    L = Environment(np.eye(4)/np.sqrt(4), 'I')
    R = Environment(gate(env_p), 'R')

    if which=='eig':
        qbs = cirq.LineQubit.range(6)
        C = cirq.Circuit.from_ops([cirq.H(qbs[3]), cirq.CNOT(*qbs[3:5]),
                                   gate(old_U)(*qbs[2:4]),
                                   gate(old_U)(*qbs[1:3]),
                                   Environment(expm(-1j*H*dt), 'W')(*qbs[2:4]),
                                   L(*qbs[0:2]),
                                   R(*qbs[4:]),
                                   cirq.inverse(gate(p))(*qbs[1:3]),
                                   cirq.inverse(gate(p))(*qbs[2:4]),
                                   cirq.CNOT(*qbs[3:5]), cirq.H(qbs[3])])
        return s.simulate(C).final_state
        
    elif which=='norm':
        qbs = cirq.LineQubit.range(4)
        normC = cirq.Circuit.from_ops([cirq.H(qbs[1]),
                                       cirq.CNOT(*qbs[1:3]),
                                       L(*qbs[:2]),
                                       R(*qbs[-2:]),
                                       cirq.CNOT(*qbs[1:3]), 
                                       cirq.H(qbs[1])])
        return s.simulate(normC).final_state

def evo_Hs(D=2):
    return np.diag(np.eye(2**6)[0]), np.diag(np.eye(2**4)[0])

def swapper():
    return -np.kron(np.kron(np.eye(2), swap()), np.eye(8))

def sinusoids(H, state_function, parameters, args=()): 
    def ϵ(x):
        return np.real(state_function(x, *args).conj().T@H@state_function(x, *args))
    I = np.eye(len(parameters))
    xs = np.linspace(-π, π, 101)
    for i in range(15):
        es = []
        for x in xs:
            es.append(ϵ(parameters+x*I[i]))
        θ_ = (-π/2-np.arctan2(2*ϵ(parameters)-ϵ(parameters+I[i]*π/2)-ϵ(parameters-I[i]*π/2), ϵ(parameters+I[i]*π/2)-ϵ(parameters-I[i]*π/2)))
        θ = np.arctan2(np.sin(θ_), np.cos(θ_))
        plt.scatter([θ], [ϵ(parameters+θ*I[i])], marker='x')

        plt.plot(xs, es)
    plt.show()

def double_sinusoids(H, state_function, parameters, args=()): 
    def ϵ(x):
        u_state, v_state, e_state = state_function(x, 'u_purity'), state_function(x, 'v_purity'), state_function(x, 'energy')
        v_purity = np.real(v_state.conj().T@np.kron(np.eye(2), np.kron(swap(), np.eye(2)))@v_state)
        u_purity = np.real(u_state.conj().T@np.kron(np.eye(4), np.kron(swap(), np.eye(4)))@u_state)
        energy = np.real(e_state.conj().T@H@e_state) 
        return energy+u_purity+v_purity
    I = np.eye(len(parameters))
    xs = np.linspace(-π, π, 101)
    for i in range(15):
        es = []
        def M(x):
            return ϵ(parameters+I[i]*x)
        for x in xs:
            es.append(M(x))

        A = float((M(0)+M(np.pi)))
        B = float((M(0)-M(np.pi)))
        C = float((M(np.pi/2)+M(-np.pi/2)))
        D = float((M(np.pi/2)-M(-np.pi/2)))
        E = float((M(np.pi/4)-M(-np.pi/4)))

        a, b, c, d = 1/4*(2*E-np.sqrt(2)*D), 1/4*(A-C), 1/2*D, 1/2*B

        P = np.sqrt(a**2+b**2)
        u = np.arctan2(b, a)
        
        Q = np.sqrt(c**2+d**2)
        v = np.arctan2(d, c)

        def f(x): return (P*np.sin(2*x+u)+Q*np.sin(x+v))

        plt.scatter(xs, f(xs)-f(xs)[0]+es[0], s=10)
        θ = minimize_scalar(f, bounds = [-np.pi, np.pi]).x
        plt.scatter([θ], [M(θ)], marker='x')
        plt.plot(xs, es)
    plt.show()

def rotosolve(H, state_function, initial_parameters, args=(), N_iters=10):
    """rotosolve

    :param H: hamiltonian.
    :param state_function: function taking parameters and returning complex vector.
    :param initial_parameters: initial parameters.
    :param args: extra arguments to state_function.
    :param N_iters: maximum number of optimization iterations.
    """
    S = []
    es = []
    I = np.eye(len(initial_parameters))
    params = initial_parameters
    for _ in range(N_iters):
        #H = Hamiltonian({'ZZ': 1, 'X': 0.5}).to_matrix()

        def ϵ(x):
            return np.real(state_function(x, *args).conj().T@H@state_function(x, *args))


        for i, _ in enumerate(params):
            θ_ = (-np.pi/2-np.arctan2(2*ϵ(params)-ϵ(params+I[i]*π/2)-ϵ(params-I[i]*π/2), ϵ(params+I[i]*π/2)-ϵ(params-I[i]*π/2)))
            params[i] += np.arctan2(np.sin(θ_), np.cos(θ_))
            params[i] = np.arctan2(np.sin(params[i]), np.cos(params[i]))
        sinusoids(H, state_function, params)
        es.append(ϵ(params))
        S.append(params.copy())
    return es, S

def double_rotosolve(H, state_function, initial_parameters, args=(), N_iters=5):
    """rotosolve

    :param H: hamiltonian.
    :param state_function: function taking parameters and returning complex vector.
    :param initial_parameters: initial parameters.
    :param args: extra arguments to state_function.
    :param N_iters: maximum number of optimization iterations.
    """
    S = []
    ss = []
    es = []
    I = np.eye(len(initial_parameters))
    params = initial_parameters
    for _ in range(N_iters):
        #H = Hamiltonian({'ZZ': 1, 'X': 0.5}).to_matrix()

        def ϵ(x):
            ψ = state_function(x, *args)
            return np.real(ψ.conj().T@H@ψ)

        def ϵ_op(x):
            uv_state, u_state, v_state, e_state = state_function(x, 'uv_purity'), state_function(x, 'u_purity'), state_function(x, 'v_purity'), state_function(x, 'energy')
            v_purity = np.real(v_state.conj().T@np.kron(np.eye(2), np.kron(swap(), np.eye(2)))@v_state)
            u_purity = np.real(u_state.conj().T@np.kron(np.eye(4), np.kron(swap(), np.eye(4)))@u_state)
            uv_purity = np.real(uv_state.conj().T@np.kron(np.kron(np.eye(2), swap()), np.eye(4))@uv_state)
            energy = np.real(e_state.conj().T@H@e_state) 

            k = 10
            return energy,+k*u_purity,+k*v_purity,-2*k*uv_purity


        for i, _ in tqdm(enumerate(params)):
            def M(x):
                return np.sum(ϵ(params+I[i]*x))

            A = (M(0)+M(np.pi))
            B = (M(0)-M(np.pi))
            C = (M(np.pi/2)+M(-np.pi/2))
            D = (M(np.pi/2)-M(-np.pi/2))
            E = (M(np.pi/4)-M(-np.pi/4))

            a, b, c, d = 1/4*(2*E-np.sqrt(2)*D), 1/4*(A-C), 1/2*D, 1/2*B

            P = np.sqrt(a**2+b**2)
            u = np.arctan2(b, a)
            
            Q = np.sqrt(c**2+d**2)
            v = np.arctan2(d, c)

            def f(x): return (P*np.sin(2*x+u)+Q*np.sin(x+v))

            θ_ = minimize_scalar(f, bounds = [-np.pi, np.pi]).x
            #θ_ = (-np.pi/2-np.arctan2(2*ϵ(params)-ϵ(params+I[i]*π/2)-ϵ(params-I[i]*π/2), ϵ(params+I[i]*π/2)-ϵ(params-I[i]*π/2)))
            params[i] += np.arctan2(np.sin(θ_), np.cos(θ_))
            #params[i] = np.arctan2(np.sin(params[i]), np.cos(params[i]))
        #double_sinusoids(H, state_function, params)
        es.append(ϵ(params))
    return np.array(es), params
