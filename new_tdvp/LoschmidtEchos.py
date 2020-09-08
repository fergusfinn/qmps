import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
from ClassicalTDVPStripped import tensor, CircuitSolver
from BrickWallMPS import optimize_2layer_bwmps, state_from_params, LeftEnvironment, RightEnvironment
from qmps.ground_state import Hamiltonian
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle


def ground_state(H):

    def obj(p):
        psi1 = state_from_params(p, 3)
        return np.real(psi1.conj().T @ H @ psi1)

    init_p = np.random.rand(22)

    res = minimize(obj, init_p,
                   method="Nelder-Mead",
                   options={"maxiter": 10000},
                   tol=1e-8)

    return res.x


def r(m):
    return m.reshape(2, 2, 2, 2)


def time_evolve(init_p, H):
    C = CircuitSolver()
    Re = RightEnvironment()
    U = expm(-1j * H * 0.01)
    results = []

    for _ in tqdm(range(500)):
        results.append(init_p)
        res = minimize(obj, init_p,
                       method="Nelder-Mead",
                       options={"maxiter": 10000, "disp": True},
                       tol=1e-4,
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


def obj(p, init_p, U, Re, C):
    # calcultate the states
    psi1 = state_from_params(init_p, 3)
    psi2 = state_from_params(p, 3)

    # claculate the environmet Mr
    # U1, U2 = C.paramU(init_p)
    # U1_, U2_ = C.paramU(p)
    # Mr = Re.exact_environment(r(U1),
    #                           r(U2),
    #                           r(U1_.conj().T),
    #                           r(U2_.conj().T))[1]

    U_ev = tensor([np.eye(2), U, np.eye(2)])

    return -np.abs(psi2.conj().T @ U_ev @ psi1)


if __name__ == "__main__":
    loschmidt()
    # J, g0, g1 = -1, 1.5, 0.2
    # I = np.eye(2)
    # H2_ = H2(J, g0)
    # H2_ = tensor([I, H2_, I])

    # gs = ground_state(H2_)

    # H2_ = H2(J, g1)
    # U = expm(-1j * H2_ * 0.001)
    # C = CircuitSolver()
    # Re = RightEnvironment()
    # Le = LeftEnvironment()
    # res = minimize(obj, gs,
    #                method="Nelder-Mead",
    #                options={"maxiter": 10000, "disp": True},
    #                tol=1e-3,
    #                args=(gs, U, Re, C))

    # psi1 = state_from_params(gs, 3)
    # psi2 = state_from_params(res.x, 3)

    # # claculate the environmet Mr
    # U1, U2 = C.paramU(gs)
    # U1_, U2_ = C.paramU(res.x)
    # Mr = Re.exact_environment(r(U1),
    #                           r(U2),
    #                           r(U1_.conj().T),
    #                           r(U2_.conj().T))[1]

    # Ml = Le.exact_environment(r(U1),
    #                           r(U2),
    #                           r(U1_.conj().T),
    #                           r(U2_.conj().T))[1]

    # U_ev = tensor([np.eye(2)/np.sqrt(2), U, np.eye(2) / np.sqrt(2)])

    # -2 * np.abs(psi2.conj().T @ U_ev @ psi1)
