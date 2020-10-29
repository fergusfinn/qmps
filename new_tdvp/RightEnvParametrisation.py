import numpy as np
from ClassicalTDVPStripped import Represent, RightEnvironment, CircuitSolver
import matplotlib.pyplot as plt 
from scipy.stats import unitary_group
from scipy.linalg import expm
from tqdm import tqdm

def r(m):
    return m.reshape(2,2,2,2)


def optimization_value_plots():
    """
    check how the parameters evolve with the optimization as a function of time.
    """
    U1, U2 = [unitary_group.rvs(4) for _ in range(2)]
    A1,A2 = [np.random.rand(16).reshape(4,4) + 1j*np.random.rand(16).reshape(4,4) for _ in range(2)]
    H1 = 0.5*(A1 + A1.conj().T)
    H2 = 0.5*(A2 + A2.conj().T)

    te1 = expm(1j * H1 * 0.01)
    te2 = expm(1j * H1 * 0.01)

    U1_ = (te1 @ U1).conj().T
    U2_ = (te2 @ U2).conj().T

    RE = Represent()

    t = 0.01
    answers = []
    for i in tqdm(range(1, 30)):
        t = t*i

        te1 = expm(1j * H1 * t)
        te2 = expm(1j * H2 * t)

        U1_ = (te1 @ U1).conj().T
        U2_ = (te2 @ U2).conj().T

        res = RE.optimize(r(U1),r(U2),r(U1_), r(U2_))
        answers.append(np.mod(res.x, 2*np.pi))

    return answers

def osciallating_param_costs():

    U1, U2, U1_, U2_ = [unitary_group.rvs(4).reshape(2,2,2,2) for _ in range(4)]
    RE = RightEnvironment()
    params = np.random.rand(6)
    C = CircuitSolver()
    for i in range(len(params)):
        Ms = []
        M_s = []
        for p in np.linspace(0,np.pi*2, 200):
            params[i] = p
            M = C.M(params)
            M_ = RE.circuit(U1, U2, U1_, U2_, M)

            Ms.append(M.reshape(4,))
            M_s.append(M_.reshape(4,))
            

        scores = np.linalg.norm(np.array(Ms) - np.array(M_s), axis = 1)
        plt.plot(scores, label = f"{i}")

        plt.legend()
        plt.show()

def poly_fit_eta():
    from numpy.polynomial.polynomial import Polynomial
    x = np.linspace(1,0,2000)
    for i in range(10):
        U1, U2 = [unitary_group.rvs(4) for _ in range(2)]
        A1,A2 = [np.random.rand(16).reshape(4,4) + 1j*np.random.rand(16).reshape(4,4) for _ in range(2)]
        H1 = 0.5*(A1 + A1.conj().T)
        H2 = 0.5*(A2 + A2.conj().T)

        te1 = expm(1j*H1*0.1)
        te2 = expm(1j*H2*0.1)
        
        U1_ = (te1 @ U1).conj().T
        U2_ = (te2 @ U2).conj().T

        RE = RightEnvironment()
        params = [np.pi/4,0,0,0,0,0] # np.random.rand(6)
        C = CircuitSolver()
        
        M = C.M(params)
        M_ = RE.circuit(r(U1), r(U2), r(U1_), r(U2_), M)

        
        scores = []
        for p in x:
            scores.append(np.linalg.norm(p * M - M_))

        coefs = Polynomial.fit(x[:10],scores[:10],deg = 2, domain = (1,0.9))
        new_vals = [coefs(a) for a in x]
        plt.plot(x, scores, label = "Exact")
        plt.plot(x, new_vals, label = "Poly Fit")
        plt.legend()
        plt.show()  

if __name__ == "__main__":
    ################
    # Set up a random environment calculation
    U1, U2, U1_, U2_ = [unitary_group.rvs(4).reshape(2,2,2,2) for _ in range(4)]
    RE = RightEnvironment()
    R = Represent()
    R.U1 = U1
    R.U2 = U2
    R.U1_ = U1_
    R.U2_ = U2_

    ###############
    # Demonstrate the sinusoidal cost Function
    params = np.random.rand(6)
    C = CircuitSolver()
    params = [1] + list(np.random.rand(6))
    CostFuncs = []
    x = np.linspace(0,2*np.pi, 200)

    for p in x:
        params[1]=p
        CF = R.cost_function(params)
        CostFuncs.append(CF)

    plt.plot(x, CostFuncs, label = "Exact")

    ###############
    # Attempt to fit an exact curve suing the RotoSolve Calcs:

    params[1] = 0
    M0 = R.cost_function(params)

    params[1] = np.pi / 2
    Mpi = R.cost_function(params)

    params[1] = np.pi / 4
    Mpi2 = R.cost_function(params)

    params[1] = -np.pi / 4
    Mpi2_ = R.cost_function(params)

    A = 0.5*np.sqrt( (M0 - Mpi)**2 + (Mpi2 - Mpi2_)**2)
    B = np.arctan2(M0 - Mpi, Mpi2 - Mpi2_)
    C= 0.5*(M0 + Mpi)

    sin_fit = lambda x:(A*np.sin(2*x + B) + C)
    fit_data = list(map(sin_fit, x))
    plt.plot(x, fit_data, label = "Fit")
    plt.legend()
    plt.show()








        

