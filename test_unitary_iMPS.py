import unittest

from xmps.iMPS import iMPS, Map, TransferMatrix
from xmps.tensor import embed, deembed
from unitary_iMPS import *
from random_unitaries import random_unitary
import cirq

from numpy import allclose
from numpy.random import randn
from numpy.linalg import qr
import matplotlib.pyplot as plt
from tqdm import tqdm

class TestcMPS(unittest.TestCase):
    """TestcMPS"""

    def setUp(self):
        N = 3  
        self.xs = [randn(8, 8)+1j*randn(8, 8) for _ in range(N)]
        self.As = [iMPS().random(2, 2).mixed() for _ in range(N)]

    def test_mat_demat(self):
        for x in self.xs:
            self.assertTrue(allclose(mat(demat(x)), x))

    def test_to_unitaries_l(self):
        for AL, AR, C in self.As:
            Us = to_unitaries_l(AL)
            As = from_unitaries_l(Us)
            self.assertTrue(allclose(As[0], AL[0]))

    def test_full_environment_objective_function(self):
        for AL, AR, C in self.As:
            AL, AR = AL.data[0], AR.data[0]

            # consistency checks
            r = C@C.conj().T
            l = C.conj().T@C
            I = eye_like(l)

            self.assertTrue(Map(AL, AL).is_right_eigenvector(r))
            self.assertTrue(Map(AL, AL).is_left_eigenvector(I))

            self.assertTrue(Map(AR, AR).is_right_eigenvector(I))
            self.assertTrue(Map(AR, AR).is_left_eigenvector(l))

            # make unitaries
            U = to_unitaries_l([AL])[0]
            V = embed(C)

            self.assertTrue(full_env_obj_fun(U, V)<1e-6)

    def test_sampled_environment_objective_function(self):
        for AL, AR, C in self.As:
            AL, AR = AL.data[0], AR.data[0]
            # make unitaries
            U = to_unitaries_l([AL])[0]
            V = embed(C)
            self.assertTrue(sampled_env_obj_fun(U, V, reps=10000)<1e-1)

    def test_sampled_bloch_vector_of(self):
        circuit = random_unitary(5)
        sim = cirq.Simulator()
        v1 = sim.simulate(circuit).bloch_vector_of(list(circuit.all_qubits())[0])
        v2 = sampled_bloch_vector_of(list(circuit.all_qubits())[0], circuit, 100000)
        self.assertTrue(sum(np.abs(v1-v2))<0.1)

    @unittest.skip('slow')
    def test_get_env(self):
        for AL, AR, C in self.As:
            AL, AR = AL.data[0], AR.data[0]
            # make unitaries
            U = to_unitaries_l([AL])[0]
            V = get_env(U)
            C_ = deembed(V)
            self.assertTrue(allclose(svals(C),svals(C_), atol=1e-5))
            self.assertTrue(full_env_obj_fun(U, embed(C_))<1e-5)
            self.assertTrue(full_env_obj_fun(U, embed(C)) <1e-5)

    @unittest.skip('slow')
    def test_optimize_ising(self):
        from scipy import integrate
        for AL, AR, C in self.As:
            g = 0.5
            U, V = optimize_ising_D_2(1, g)

            f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
            E0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
            print("E_exact =", E0_exact)

    @unittest.skip('useless')
    def test_state_class(self):
        n = 2
        D = 2**n
        U = eye(2*D)
        V = eye(D**2)
        qbs = cirq.LineQubit.range(2*n+1)
        C = cirq.Circuit()
        C.append([State(ShallowStateTensor(D, 0.1, 0.1), 
                        ShallowEnvironment(D, 0.5, 0.1))(*qbs)])
        print('\n', C.to_text_diagram(transpose=True), '\n')
        sim = cirq.Simulator()
        print(sim.simulate(C).final_state)

    def test_shallow_env_obj_fun(self):
        N = 200
        n_qubits = 5 # remember req. qubits is 2*n_qubits+1
        X, Y = np.linspace(0, 2, N), np.linspace(0, 2, N)
        bg = (np.abs(randn(2)),)
        for u in np.linspace(0, 1, 10):
            Z = np.array([[shallow_env_obj_fun(bg, ((x, y), (u, 0)), n_qubits) for x in X]
                           for y in tqdm(Y)])
            print(np.min(Z))
            plt.contourf(Z)
            plt.colorbar()
            plt.xlabel('β')
            plt.ylabel('γ')
            plt.show()
        
if __name__=='__main__':
    unittest.main(verbosity=1)
