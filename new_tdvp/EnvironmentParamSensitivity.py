import numpy as np 
from xmps.iMPS import Map
from scipy.optimize import minimize
from scipy.stats import unitary_group
from scipy.linalg import expm
from numpy import exp, pi, cos,sin
def alternate_tensor(U2, U1):
    """
    Produce a tensor A[(σ1σ2),i,j]:
    
    A[(σ1σ2),i,j] = Σα U1[σ1,σ2,i,α]U2[α,j,0,0]
    """
    
    return np.tensordot(
        U1.reshape(2,2,2,2),
        U2[:,0].reshape(2,2),
        (3,0)).reshape(4,2,2)

def D3(theta):
    return np.array([
            [cos(theta), 0],
            [0, sin(theta)]
        ])

def X(theta):
    return np.array([
            [cos(pi * theta / 2), -1j * sin(pi * theta / 2)],
            [-1j * sin(pi * theta / 2), cos(pi * theta / 2)]
            ])
        
def Z(theta): 
    return np.array([        
            [1, 0],
            [0, exp(1j * pi * theta)]
            ])


def M(params):
    a,b,c,d,e,f = params 
    M = Z(b) @ X(c) @ Z(d) @ D3(a) @ X(e) @ Z(f)
    return M   

def cost_func(params, U2,U1,Ū2,Ū1):
    m = M(params)
    A1 = alternate_tensor(U2,U1)
    Ā2 = alternate_tensor(Ū2,Ū1)
    TransferMatrix = Map(A1,Ā2)
    EigenEqn = TransferMatrix @ m
    return np.linalg.norm(m - EigenEqn)
 
if __name__ == "__main__":
    A1 = np.random.rand(16).reshape(4,4)
    A2 = np.random.rand(16).reshape(4,4)
    
    H1 = 0.5*(A1 + A1.conj().T)
    H2 = 0.5*(A2 + A2.conj().T)
    
    U2 = expm(1j * H2 * 1)
    U1 = expm(1j * H1 * 1)

    Ū2 = expm(1j * H2 * 1.05)
    Ū1 = expm(1j * H1 * 1.05)

    params = [pi/4,0,0,0,0,0]
    
    res = minimize(
        cost_func,
        x0 = params,
        args = (U2,U1,Ū2,Ū1),
        
    )
    
