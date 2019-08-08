import unittest
import numpy as np
from xmps.spin import spins
from xmps.iMPS import iMPS
from scipy.linalg import expm
from qmps.represent import FullStateTensor, FullEnvironment, get_env_exact, State
from qmps.tools import environment_to_unitary, tensor_to_unitary, unitary_to_tensor, environment_from_unitary, RepresentMPS
from qmps.time_evolve import MPSTimeEvolve
import matplotlib.pyplot as plt
from xmps.spin import U4
from qmps.ground_state import NonSparseFullEnergyOptimizer
import cirq
from scipy.stats import unitary_group

Sx, Sy, Sz = spins(0.5)
Sx, Sy, Sz = 2*Sx, 2*Sy, 2*Sz


As = [iMPS().random(2, 2).mixed() for _ in range(3)]


def dat_to_plot_data(file):
    f = open(file, 'r')
    lines = f.readlines()
    f.close()
    xs = []
    ys = []
    for line in list(lines):
        no_n = line.replace('\n', '')
        x, y = no_n.split('\t')
        xs.append(float(x))
        ys.append(float(y))
    return xs, ys


def simulate_state(u, v):
    state = State(FullStateTensor(u), FullEnvironment(v),1)
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops([state.on(*qubits)])
    sim = cirq.Simulator()
    results = sim.simulate(circuit)
    return results, qubits

def test_time_evolve():
    np.random.seed(0)
    initial_params = np.random.rand(15)

    unitary = U4(initial_params)
    A = iMPS([unitary_to_tensor(unitary)]).left_canonicalise()
    environment = get_env_exact(unitary)

    J, g = -1, 0.5
    H = np.array([[J, g / 2, g / 2, 0],
                  [g / 2, -J, 0, g / 2],
                  [g / 2, 0, -J, g / 2],
                  [0, g / 2, g / 2, J]])

    T = np.linspace(0, 1, 200)
    dt = T[1] - T[0]
    evs = []
    es = []

    ################ CIRQ BIT ######################

    counter = 0
    bloch_sphere_results = []

    U = FullStateTensor(unitary)
    V = FullEnvironment(environment)
    hamiltonian = FullStateTensor(expm(-1j * H * dt * 2))
    # hamiltonian = FullStateTensor(np.identity(4))
    evolver = MPSTimeEvolve(u_initial=U, hamiltonian=hamiltonian, v_initial=V, initial_params= initial_params,
                            settings={'method': 'Nelder-Mead',
                                      'maxiter': 1000,
                                      'verbose': True,
                                      'tol': 1e-5
                                       })
    # initial classical value
    evs.append(iMPS([A.data[0]]).Es([Sx, Sy, Sz]))

    # initial unitary value
    results, qubits = evolver.simulate_state()
    bloch_sphere = results.bloch_vector_of(qubits[1])
    bloch_sphere_results.append(bloch_sphere)

    for _ in T:
        print(counter)
        dA = A.dA_dt([H]) * dt
        es.append(A.e)
        A = (A + dA).left_canonicalise()
        evs.append(A.Es([Sx, Sy, Sz]))
        '''
        Below is the cirq time evolution. Gate is made using scipy.expm(H) * dt which is put into a Tensor, and 
        and this is the hamiltonian.
        '''
        evolver.evolve_single_step()
        results, qubits = evolver.simulate_state()
        bloch_sphere = results.bloch_vector_of(qubits[1])
        bloch_sphere_results.append(bloch_sphere)
        counter += 1

    plt.figure()
    plt.plot(np.array(bloch_sphere_results), 'b')
    plt.plot(np.array(evs), 'r')

    plt.legend()
    plt.show()
    return np.array(evs), np.array(bloch_sphere_results)


def losmicht_echo():
    # Get ground state of original ham
    J, g = -1, 1.5
    H = np.array([[J, g / 2, g / 2, 0],
                  [g / 2, -J, 0, g / 2],
                  [g / 2, 0, -J, g / 2],
                  [0, g / 2, g / 2, J]])

    opt = NonSparseFullEnergyOptimizer(H)
    opt.optimize()
    optimized_result = opt.u
    # Analytic initial state
    A = iMPS([unitary_to_tensor(cirq.unitary(optimized_result))]).left_canonicalise()
    a_unitary = tensor_to_unitary(A.data[0])
    a_env = get_env_exact(a_unitary)
    results, qubits = simulate_state(a_unitary, a_env)
    a_original_state = results.final_simulator_state.state_vector

    # Quantum Initial State
    environment = get_env_exact(cirq.unitary(optimized_result))
    q_results, qubits = simulate_state(cirq.unitary(optimized_result), environment)
    original_state = q_results.final_simulator_state.state_vector

###########################################################
    g = -16
    H = np.array([[J, g / 2, g / 2, 0],
                  [g / 2, -J, 0, g / 2],
                  [g / 2, 0, -J, g / 2],
                  [0, g / 2, g / 2, J]])

    T = np.linspace(0, 2, 500)
    dt = T[1] - T[0]
    hamiltonian = FullStateTensor(expm(-1j * H * dt*2))
###########################################################
    V = FullEnvironment(environment)
    evolver = MPSTimeEvolve(u_initial=optimized_result, hamiltonian=hamiltonian, v_initial=V,
                            settings={'method': 'Nelder-Mead',
                                      'maxiter': 10000,
                                      'verbose': True,
                                      'tol': 1e-7
                                        })

    counter = 0

    q_results = []
    a_results = []
    for _ in T:
        print(counter)
        # analytix
        a_unitary = tensor_to_unitary(A.data[0])
        a_environment = get_env_exact(a_unitary)
        a_new_results, qubits = simulate_state(a_unitary, a_environment)
        a_new_state = a_new_results.final_simulator_state.state_vector
        a_results.append(-np.log(np.dot(a_original_state.conj(), a_new_state)))
        dA = A.dA_dt([H]) * dt
        A = (A + dA).left_canonicalise()

        # qntm
        q_result, qubits = evolver.simulate_state()
        q_new_state = q_result.final_simulator_state.state_vector
        q_results.append(-np.log(np.dot(original_state.conj(), q_new_state)))
        evolver.evolve_single_step()
        counter += 1

    return q_results, a_results, T


def losmicht_echo_analytic():

    J, g = -1/2, 0.5
    # Get Ground state of g = 1.5
    H = np.array([[J, g / 2, g / 2, 0],
                  [g / 2, -J, 0, g / 2],
                  [g / 2, 0, -J, g / 2],
                  [0, g / 2, g / 2, J]])
    opt = NonSparseFullEnergyOptimizer(H)
    opt.optimize()
    optimized_result = cirq.unitary(opt.u)

    A = iMPS([unitary_to_tensor(optimized_result)]).left_canonicalise()
    unitary = tensor_to_unitary(A.data[0])
    environment = get_env_exact(unitary)

    results, qubits = simulate_state(unitary, environment)
    original_state = results.final_simulator_state.state_vector

    g = 2
    H = np.array([[J, g / 2, g / 2, 0],
                  [g / 2, -J, 0, g / 2],
                  [g / 2, 0, -J, g / 2],
                  [0, g / 2, g / 2, J]])

    T = np.linspace(0, 4, 8000)
    dt = T[1] - T[0]
    overlaps = []
    counter= 0
    for _ in T:
        print(counter)

        dA = A.dA_dt([H]) * dt
        A = (A + dA).left_canonicalise()

        unitary = tensor_to_unitary(A.data[0])
        environment = get_env_exact(unitary)

        new_results, qubits = simulate_state(unitary, environment)
        new_state = new_results.final_simulator_state.state_vector
        overlaps.append(-np.log(np.abs(np.dot(new_state.conj(), original_state))))

        counter += 1
    return overlaps, T
'''

pi/2J * sqrt((g_init + g_fin) / ((g_init - g_fin)(1-gfin^2)))

'''


def test_represent():
    np.random.seed = 0
    U = unitary_group.rvs(4, random_state=0)
    get_env = RepresentMPS(FullStateTensor(U), ansatz='Full')
    get_env.optimize()
    get_env.change_settings({'method': 'Nelder-Mead', 'maxiter': 5000})

    qubits = cirq.LineQubit.range(10)
    circuit = cirq.Circuit.from_ops([State(get_env.u, get_env.v, 1).on(*qubits[:3]),
                                     get_env.v.on(*qubits[3:5])])

    sim = cirq.Simulator()
    res = sim.simulate(circuit)
    den1 = res.bloch_vector_of(*qubits[0:1])
    den2 = res.bloch_vector_of(*qubits[3:4])
    print(np.linalg.norm(den1 - den2))
    return 0


if __name__ == '__main__':
    a_results, t = losmicht_echo_analytic()
