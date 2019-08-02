import cirq
from numpy.random import randn
from xmps.iMPS import iMPS, Map
from qmps.tools import tensor_to_unitary, unitary_to_tensor, eye_like, environment_to_unitary, RepresentMPS
from qmps.represent import FullStateTensor, FullEnvironment
from qmps.States import State
from qmps.represent import *
from xmps.spin import spins
from qmps.tools import RepresentMPS
Sx, Sy, Sz = spins(0.5)
Sx, Sy, Sz = 2*Sx, 2*Sy, 2*Sz


def test_density_matrix_cost_funtion():
    N = 3
    As = [iMPS().random(2, 2).mixed() for _ in range(N)]
    for AL, AR, C in As:
        U = FullStateTensor(tensor_to_unitary(AL.data[0]))
        v_analytic = FullEnvironment(environment_to_unitary(C))

        represent = RepresentMPS(U)
        represent.change_settings({'method': 'Powell',
                                   'maxiter':100,
                                   'tol': 1e-8,
                                   'verbose': True,
                                   'store_values': True})
        represent.optimize()
        v_optimized = represent.v

        # State with analytic environment
        analytic_state = State(U, v_analytic, 1)

        # State with 'optimized' environment
        optimized_state = State(U, v_optimized, 1)

        qubits = cirq.LineQubit.range(5)

        # circuit with the environment found by RepresentMPS
        optimized_circuit = cirq.Circuit.from_ops([optimized_state(*[qubits[0], qubits[1], qubits[2]]),
                                                   v_optimized(*[qubits[3], qubits[4]])])

        # circuit with analytic environment
        analytic_circuit = cirq.Circuit.from_ops([analytic_state(*[qubits[0], qubits[1], qubits[2]]),
                                                  v_analytic(*[qubits[3], qubits[4]])])

        # circuit using analytic environment that has CNOTS and H gates to get cost function
        analytic_swap_circuit = analytic_circuit.copy()
        analytic_swap_circuit.append([cirq.CNOT(qubits[0], qubits[3]), cirq.H(qubits[0])])

        # circuit with CNOT and H using optimized environment
        optimized_swap_circuit = optimized_circuit.copy()
        optimized_swap_circuit.append([cirq.CNOT(qubits[0], qubits[3]), cirq.H(qubits[0])])

        sim = cirq.Simulator()

        # results of optimized circuit and density matrix circuits
        optimized_results = sim.simulate(optimized_circuit)
        optimized_density_matrix_results = sim.simulate(optimized_swap_circuit)

        # results of analytic circuit and density matrix circuits
        analytic_results = sim.simulate(analytic_circuit)
        analytic_density_matrix_results = sim.simulate(analytic_swap_circuit)

        # bloch vectors
        optimized_bloch0 = optimized_results.bloch_vector_of(qubits[0])
        optimized_bloch3 = optimized_results.bloch_vector_of(qubits[3])

        analytic_bloch0 = analytic_results.bloch_vector_of(qubits[0])
        analytic_bloch3 = analytic_results.bloch_vector_of(qubits[3])

        # density matrices
        analytic_density_matrix = analytic_density_matrix_results.density_matrix_of([qubits[0], qubits[3]])
        analytic_prob_all_ones = analytic_density_matrix[-1, -1]
        analytic_score = np.abs(analytic_prob_all_ones)

        optimized_density_matrix = optimized_density_matrix_results.density_matrix_of([qubits[0], qubits[3]])
        optimized_prob_all_ones = optimized_density_matrix[-1, -1]
        optimized_score = np.abs(optimized_prob_all_ones)

        print('Swap Test Results')
        print(f'Bloch vector 3 {optimized_bloch3}')
        print(f'Bloch Vetor 0 {optimized_bloch0}')
        print(f'OVerlap of bloch vetors: {np.dot(optimized_bloch0, optimized_bloch3)}')
        # print(optimized_score)
        zero_purity = np.trace(optimized_results.density_matrix_of([qubits[0]])@optimized_results.density_matrix_of([qubits[0]]))
        three_purity = np.trace(optimized_results.density_matrix_of([qubits[3]]) @ optimized_results.density_matrix_of([qubits[3]]))
        print(f'Purity of zero: {zero_purity}')
        print(f'Purity of 3: {three_purity}')

        print('Analytic Result')
        print(f'Bloch Vector 0 {analytic_bloch0}')
        print(f'Bloch Vector 3 {analytic_bloch3}')
        print(f'Overlap of vectors: {np.dot(analytic_bloch0, analytic_bloch3)}')
        #print(analytic_score)
        print(f'purity {np.trace(analytic_results.density_matrix_of([qubits[3]])@analytic_results.density_matrix_of([qubits[3]]))}')

test_density_matrix_cost_funtion()