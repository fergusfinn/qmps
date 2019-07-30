import unittest

from numpy.random import randn
 from scipy.stats import unitary_group
from xmps.iMPS import iMPS, Map
from qmps.tools import tensor_to_unitary, unitary_to_tensor, eye_like, environment_to_unitary, RepresentMPS
from qmps.represent import FullStateTensor, FullEnvironment
from qmps.represent import *
from xmps.spin import spins
Sx, Sy, Sz = spins(0.5)
Sx, Sy, Sz = 2*Sx, 2*Sy, 2*Sz


class TestRepresent(unittest.TestCase):
    def setUp(self):
        N = 3  
        self.xs = [randn(8, 8)+1j*randn(8, 8) for _ in range(N)]
        self.As = [iMPS().random(2, 2).mixed() for _ in range(N)]
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

    def test_expectation_values(self):
        for AL, AR, C in self.As:
            AL, AR = AL.data[0], AR.data[0]

            S = iMPS([AL]).Es([Sx, Sy, Sz])

            # make unitaries
            U, passed = tensor_to_unitary(AL, True)
            U = FullStateTensor(U)
            V = FullEnvironment(environment_to_unitary(C))

            qbs = cirq.LineQubit.range(3)
            sim = cirq.Simulator()
            C =  cirq.Circuit().from_ops(cirq.decompose_once(State(U, V)(*qbs)))
            S_ = sim.simulate(C).bloch_vector_of(qbs[1])
            self.assertTrue(allclose(S, S_))

    def test_full_environment_objective_function(self):
        for AL, AR, C in self.As:
            AL, AR = AL.data[0], AR.data[0]

            # make unitaries
            U = FullStateTensor(tensor_to_unitary(AL))
            V = FullEnvironment(environment_to_unitary(C))

            self.assertTrue(full_tomography_env_objective_function(U, V)<1e-6)
            self.assertTrue(sampled_tomography_env_objective_function(U, V, 10000)<1e-1)

    def test_full_parameterization_convergence(self):
        for u in [unitary_group.rvs(4) for i in range(3)]:
            state = FullStateTensor(U)
            get_env = RepresentMPS(state)


if __name__ == '__main__':
    unittest.main(verbosity=1)

