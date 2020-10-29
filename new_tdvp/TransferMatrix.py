import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eig
from xmps.spin import U4
from BrickWallMPS import OO_unitary
from ClassicalTDVPStripped import CircuitSolver

if __name__ == "__main__":
    ######################
    # Full unitary parametrisation
    ######################

    p = np.random.rand(15)
    U = U4(p).reshape(2,2,2,2)
    C = CircuitSolver()

    def obj(p_, U, C):
        Mr = C.M(p_[:6])
        U_ = U4(p_[6:]).conj().T.reshape(2,2,2,2)
        score = np.tensordot(U[...,0,0], Mr, 1)
        score = np.tensordot(score, U_[0,0,...], 2)
        return -np.abs(score)

    init_params = [np.pi/4,0,0,0,0,0] + list(p)
    
    res = minimize(obj, init_params,
                    method = "Nelder-Mead",
                    options={"disp":True},
                    args = (U,C))

    U1 = np.round(U.reshape(4,4),3)
    U2 = np.round(U4(res.x[6:]),3)
    Mr = np.round(C.M(res.x[:6]),3)

    string = f"""
    U:{U1}\n
    U_:{U2}\n
    Mr:{Mr}\n
    Score:{res.fun}\n
    NotOptimal:{obj(init_params, U, C)}
    """

    print(string)


            
