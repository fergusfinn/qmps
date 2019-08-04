import unittest
import numpy as np
from xmps.spin import spins
from xmps.iMPS import iMPS
from scipy.linalg import expm
from qmps.represent import FullStateTensor, FullEnvironment, get_env_exact, State
from qmps.tools import environment_to_unitary, tensor_to_unitary, unitary_to_tensor, environment_from_unitary
from qmps.time_evolve import MPSTimeEvolve
import matplotlib.pyplot as plt
from xmps.spin import U4
from qmps.ground_state import NonSparseFullEnergyOptimizer
import cirq

Sx, Sy, Sz = spins(0.5)
Sx, Sy, Sz = 2*Sx, 2*Sy, 2*Sz


As = [iMPS().random(2, 2).mixed() for _ in range(3)]

def test_time_evolve():
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

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    ax[0].plot(np.array(evs), c='r')
    ax[0].plot(np.array(bloch_sphere_results), c='b')
    ax[1].plot(es)
    plt.show()
    fig.savefig('xy.png')
    return np.array(evs), np.array(bloch_sphere_results)

def losmicht_echo():
    J, g = -1, 1.5
    H = np.array([[J, g / 2, g / 2, 0],
                  [g / 2, -J, 0, g / 2],
                  [g / 2, 0, -J, g / 2],
                  [0, g / 2, g / 2, J]])

    opt = NonSparseFullEnergyOptimizer(H)
    opt.optimize()
    optimized_result = opt.u

    g = 0.5
    H = np.array([[J, g / 2, g / 2, 0],
                  [g / 2, -J, 0, g / 2],
                   [g / 2, 0, -J, g / 2],
                  [0, g / 2, g / 2, J]])

    T = np.linspace(0, 32, 8000)
    dt = T[1] - T[0]
    hamiltonian = FullStateTensor(expm(-1j * H * dt))

    U = optimized_result
    environment = get_env_exact(cirq.unitary(optimized_result))
    V = FullEnvironment(environment)

    evolver = MPSTimeEvolve(u_initial=U, hamiltonian=hamiltonian, v_initial=V,
                            settings={'method': 'Nelder-Mead',
                                      'maxiter': 1000,
                                      'verbose': True,
                                      'tol': 1e-5
                                       })

    qubits = cirq.LineQubit.range(3)
    original_circuit = cirq.Circuit.from_ops([State(U, V, 1).on(*qubits)])
    sim = cirq.Simulator()
    original_state = sim.simulate(original_circuit).final_simulator_state.state_vector
    counter = 0
    results = []
    for _ in T:
        print(counter)
        evolver.evolve_single_step()
        result, _ = evolver.simulate_state()
        overlap = -np.log(np.abs(original_state.conj().T@result.final_simulator_state.state_vector)**2)
        results.append(overlap)
        counter += 1

    plt.plot(range(len(results)), results)
    plt.savefig('stupid_long_loschmidt_echos')
    plt.show()
    return results

'''

pi/2J * sqrt((g_init + g_fin) / ((g_init - g_fin)(1-gfin^2)))

'''
results = losmicht_echo()