import numpy as np 
from xmps.iMPS import Map
from scipy.optimize import minimize
from scipy.stats import unitary_group
from scipy.linalg import expm, eig
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

def Map(A1,A2):
    return np.tensordot(A1,A2.conj(),[0,0]).transpose([0,2,1,3]).reshape(4,4)

def M(params):
    a,b,c,d,e,f = params 
    M = Z(b) @ X(c) @ Z(d) @ D3(a) @ X(e) @ Z(f)
    return M   

def cost_func(params, U2,U1,Ū2,Ū1):
    m = M(params)
    A1 = alternate_tensor(U2,U1)
    A2 = alternate_tensor(Ū2,Ū1)
    TransferMatrix = Map(A1,A2)
    EigenEqn = TransferMatrix @ m.reshape(4)
    return np.linalg.norm(m - EigenEqn.reshape(2,2))
 
if __name__ == "__main__":
    A1 = np.random.rand(16).reshape(4,4)
    A2 = np.random.rand(16).reshape(4,4)
    
    H1 = 0.5*(A1 + A1.conj().T)
    H2 = 0.5*(A2 + A2.conj().T)
    
    U2 = unitary_group.rvs(4)
    U1 = unitary_group.rvs(4)

    import matplotlib.pyplot as plt

    results = []

    timesteps = [0.01*(2**i) for i in range(10)]
    for dt in timesteps:
        Ū2 = expm(1j * H1 * dt) @ U2
        Ū1 = expm(1j * H1 * dt) @ U1

        params = [pi/4,0,0,0,0,0]

        res = minimize(
            cost_func,
            x0 = params,
            args = (U2,U1,Ū2,Ū1),
            method="Nelder-Mead",
            options = {"disp":True}
        )

        results.append(res.x - np.array(params))

    r = np.abs(np.array(results))
    x = np.arange(len(timesteps))

    fig, ax = plt.subplots()
    width = 0.1

    r1 = ax.bar(x - 3*width, r[:,0], width, label = 'p1')
    r2 = ax.bar(x - 2*width, r[:,1], width, label = 'p2')
    r3 = ax.bar(x - width, r[:,2], width, label = 'p3')
    r4 = ax.bar(x , r[:,3], width, label = 'p4')
    r5 = ax.bar(x + width, r[:,4], width, label = 'p5')
    r6 = ax.bar(x + 2*width, r[:,5], width, label = 'p6')

    ax.set_xticks(x)
    ax.set_xticklabels(list(map(str,timesteps)))
    ax.legend()
    ax.axhline(y=np.pi/24, xmin=0,xmax=10)
    ax.set_ylabel("Difference in Param Values from identity")
    ax.set_xlabel("dt")
    fig.tight_layout()
    plt.show()