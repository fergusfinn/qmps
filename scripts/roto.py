import numpy as np
from xmps.spin import paulis
from scipy.linalg import expm
from functools import reduce
import matplotlib.pyplot as plt
from tqdm import tqdm 
from scipy.optimize import minimize_scalar, minimize
plt.style.use('pub_fast')
π = np.pi
    
X, Y, Z = paulis(0.5)
I = np.eye(2)

L = 3
H = np.random.randn(2**L, 2**L)+1j*np.random.randn(2**L, 2**L)
H = H+H.conj().T

def state_function(x, L=L):
    H = []
    for i in range(L-2):
        H.append(reduce(np.kron, [I]*i+ [Z, Z]+[I]*(L-(i+2))))
        H.append(reduce(np.kron, [I]*i+ [X, X]+[I]*(L-(i+2))))
    for i in range(L-1):
        H.append(reduce(np.kron, [I]*i+ [X]+[I]*(L-(i+1))))
        H.append(reduce(np.kron, [I]*i+ [Z]+[I]*(L-(i+1))))

    return reduce(lambda x,y: x@y, [expm(-1j*x[i]/2*H[i]) for i in range(min([len(x), len(H)]))])@reduce(np.kron, [np.array([1, 0])]*L)

def ϵ(x):
    return np.real(state_function(x).conj().T@H@state_function(x))

def rotosolve(ϵ, initial_parameters, args=(), N_iters=10):
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
    es.append(ϵ(params))
    for _ in tqdm(range(N_iters)):
        for i, _ in tqdm(enumerate(params)):
            θ_ = (-np.pi/2-np.arctan2(2*ϵ(params)-ϵ(params+I[i]*π/2)-ϵ(params-I[i]*π/2), ϵ(params+I[i]*π/2)-ϵ(params-I[i]*π/2)))
            params[i] += np.arctan2(np.sin(θ_), np.cos(θ_))
            params[i] = np.arctan2(np.sin(params[i]), np.cos(params[i]))
        es.append(ϵ(params))
        S.append(params.copy())
    return RotosolveResult(es, es[-1], S[-1], '')

def double_rotosolve(ϵ, initial_parameters, N_iters=10, disp=False):
    S = []
    ss = []
    es = []
    params = initial_parameters
    I = np.eye(len(params))
    es.append(ϵ(params))
    for w in tqdm(range(N_iters)):
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
        es.append(ϵ(params))
        #double_sinusoids(H, state_function, params)
    return RotosolveResult(es, es[-1], params, '')

def general_rotosolve(ϵ, initial_parameters, N_iters=10, disp=False):
    ey = np.eye(len(initial_parameters))
    cos, sin = np.cos, np.sin
    def M_(a, b, i, j, params):
        return ϵ(params+a*ey[i]+b*ey[j])

    es = []
    params = initial_parameters
    es.append(ϵ(params))
    N = len(initial_parameters)
    for _ in tqdm(range(N_iters)):
        a=b=0
        for i in tqdm(range(N-1)):
            i_, j_ = i, N-i-1
            M = lambda θ, ϕ: M_(θ, ϕ, i_, j_, params)

            def obj(x):
                θ, ϕ = x
                M_ϕ = lambda θ: ((1+cos(ϕ))/2*M(θ, 0)+(1-cos(ϕ))/2*M(θ, π)+sin(ϕ)/2*(M(θ, π/2) - M(θ, -π/2)))

                return (1+cos(θ))/2*M_ϕ(0)+(1-cos(θ))/2*M_ϕ(π)+(sin(θ)/2)*(M_ϕ(π/2)-M_ϕ(-π/2))

            res = minimize(obj, np.array([a, b]))
            params += res.x[0]*ey[i_]+res.x[1]*ey[j_]
            params = np.arctan2(np.sin(params), np.cos(params))
        es.append(ϵ(params))

    return RotosolveResult(es, es[-1], params, '')

class RotosolveResult(object):
    def __init__(self, history, fun, x, message):
        self.history = history 
        self.fun = fun
        self.x = x
        self.message = message

if __name__=='__main__':
    A = double_rotosolve(ϵ, np.random.randn(8))
    plt.scatter(list(range(len(A.history))), A.history, label='double')

    B = general_rotosolve(ϵ, np.random.randn(8))
    plt.scatter(list(range(len(B.history))), B.history, marker='x', label='general')
    plt.legend()
    plt.show() 

