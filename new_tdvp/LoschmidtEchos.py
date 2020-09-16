import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
from scipy.stats import unitary_group
from ClassicalTDVPStripped import tensor, CircuitSolver, ManifoldOverlap
from BrickWallMPS import optimize_2layer_bwmps, state_from_params
from qmps.ground_state import Hamiltonian
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import BrickWallMPS as BWMPS
from importlib import reload
reload(BWMPS)

def ground_state(H):

    def objg(p):
        psi1 = state_from_params(p, 3)
        return np.real(psi1.conj().T @ H @ psi1)

    init_p = np.random.rand(22)

    res = minimize(objg, init_p,
                   method="Nelder-Mead",
                   options={"maxiter": 10000},
                   tol=1e-8)

    return res.x


def r(m):
    return m.reshape(2, 2, 2, 2)


def time_evolve(init_p, H):
    C = CircuitSolver()
    Re = BWMPS.Represent()
    U = expm(-1j * H * 0.01)
    results = []

    for _ in tqdm(range(400)):
        results.append(init_p)
        res = minimize(obj3, init_p,
                       method="Nelder-Mead",
                       options={"maxiter": 10000, "disp": True},
                       tol=1e-3,
                       args=(init_p, U, Re, C))

        init_p = res.x

    return results


def plot_loschmidt(Us):
    U0 = Us[0]
    psi0 = state_from_params(U0, 3)

    overlaps = []

    for u in Us:
        psi = state_from_params(u, 3)
        overlaps.append(-np.log(np.abs(psi.conj().T @ psi0)**2))

    plt.plot(overlaps)
    plt.show()

    return overlaps


def H2(J, g):
    I = np.eye(2)
    H = Hamiltonian({"ZZ": J, "X": g}).to_matrix()
    return 1/3 * sum([tensor([H, I, I]), tensor([I, H, I]), tensor([I, I, H])])


def loschmidt():
    J, g0, g1 = -1, 1.5, 0.2
    I = np.eye(2)
    H2_ = H2(J, g0)
    H2_ = tensor([I, H2_, I])

    gs = ground_state(H2_)

    H2_ = H2(J, g1)

    te = time_evolve(gs, H2_)

    plot_loschmidt(te)

    with open("Loschmidt_results.pkl", "wb") as f:
        pickle.dump(te, f)

    return te


def obj(p, init_p, U, Re, Le, C):
    # calcultate the states
    psi1 = state_from_params(init_p, 3)
    psi2 = state_from_params(p, 3)
    
    # claculate the environmet Mr and Ml
    U1, U2 = C.paramU(init_p)
    U1_, U2_ = C.paramU(p)

    eta_l, Ml = Le.exact_environment(r(U1),
                            r(U2),
                            r(U1_.conj().T),
                            r(U2_.conj().T))
    
    eta_r, Mr = Re.exact_environment(r(U1),
                              r(U2),
                              r(U1_.conj().T),
                              r(U2_.conj().T))

    U_ev = tensor([eta_l*Ml, U, eta_r*Mr])

    return -np.sqrt(2*np.abs(psi2.conj().T @ U_ev @ psi1))


def obj2(p, init_p, U, Re, Mo, C):
    U1, U2 = C.paramU(init_p)
    U1_, U2_ = C.paramU(p)

    Mr = Re.exact_environment(r(U1),
                              r(U2),
                              r(U1_.conj().T),
                              r(U2_.conj().T))[1]

    O = Mo.circuit(r(U1),
                    r(U2),
                    r(U1_.conj().T),
                    r(U2_.conj().T),
                    Mr,
                    Mr.conj().T,
                    U.reshape(2,2,2,2,2,2,2,2))

    return -np.sqrt(2*np.abs(O))


def obj3(p, init_p, U, Rep, C):
    # calcultate the states
    psi1 = state_from_params(init_p, 3)
    psi2 = state_from_params(p, 3)
    
    # claculate the environmet Mr and Ml
    U1, U2 = C.paramU(init_p)
    U1_, U2_ = C.paramU(p)
    
    Mres = Rep.optimize(r(U1),
                        r(U2),
                        r(U1_.conj().T),
                        r(U2_.conj().T))

    Mr = Rep.M(Mres.x[1:])

    U_ev = tensor([Mr.conj().T, U, Mr])

    return -np.sqrt(2*np.abs(psi2.conj().T @ U_ev @ psi1)**2)


def load_and_plot(file):
    with open(file, "rb") as f:
        data = pickle.load(f)

    plot_loschmidt(data)



if __name__ == "__main__":
    ########################
    # Plot Loschmidt Echos From File
    #######################
    # file = "./Loschmidt_results.pkl"
    # overlaps = load_and_plot(file)

    ################################

    ################################
    # Run Loschmidt Echos Code
    ################################
    loschmidt()

    ################################

    ################################
    # Test Single Echo Update
    ################################
    # J, g0, g1 = -1, 1.5, 0.2

    # I = np.eye(2)
    # H2_ = H2(J, g0)
    # H2_ = tensor([I, H2_, I])
    # gs = np.random.rand(22) # ground_state(H2_)

    # H2_ = H2(J, g1)
    # U = expm(-1j * H2_ * 0.01)
    # C = CircuitSolver()
    # Re = BWMPS.RightEnvironment()
    # Le = BWMPS.LeftEnvironment()
    # Rep = BWMPS.Represent()
    # Mo=ManifoldOverlap()
    # for _ in tqdm(range(10)):
    #     gs = np.random.rand(22)

    #     res = minimize(obj, gs,o-
    #                     method="Nelder-Mead",
    #                     options={"maxiter": 10000, "disp": True},
    #                     tol=1e-3,
    #                     args=(gs, U, Re, Le, C))

    #     psi1 = state_from_params(gs, 3)
    #     psi2 = state_from_params(res.x, 3)

    #     # claculate the environmet Mr
    #     U1, U2 = C.paramU(gs)
    #     U1_, U2_ = C.paramU(res.x)
    #     eta, Mr = Re.exact_environment(r(U1),
    #                                 r(U2),
    #                                 r(U1_.conj().T),
    #                                 r(U2_.conj().T))

    #     etal, Ml = Le.exact_environment(r(U1),
    #                                 r(U2),
    #                                 r(U1_.conj().T),
    #                                 r(U2_.conj().T))

    #     U_ev = tensor([etal*Ml, U, eta*Mr])
    #     score = -np.sqrt(2*np.abs(psi2.conj().T @ U_ev @ psi1)**2)
    #     print("Cost Function Score: ", round(score, 3))
    #     print("Right Environment: \n", np.round(Mr, 3) )
    #     print("Transfer Matrix Largest Eigenvalue: ", np.round(eta, 3))
###############################
    
    ###############################
    # Test environment calculating code
    ###############################

    # Re = RightEnvironment()

    # U1, U2 = [unitary_group.rvs(4) for _ in range(2)]
    # U1_ = U1.conj().T
    # U2_ = U2.conj().T
    

    # eta, M = Re.exact_environment(r(U1), r(U2), r(U1_), r(U2_))
    # print(np.round(M,3))
    # print(eta)

    # X = Re.circuit(r(U1), r(U2), r(U1_), r(U2_), np.eye(2))

    # print(eta*np.eye(2))
    # print(X)
