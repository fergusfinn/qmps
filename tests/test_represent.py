import unittest

from numpy.random import randn
 
from xmps.iMPS import iMPS, Map

from qmps.tools import tensor_to_unitary, unitary_to_tensor, eye_like, environment_to_unitary
from qmps.represent import FullStateTensor, FullEnvironment
from qmps.represent import *
from xmps import Map


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

    def test_full_environment_objective_function(self):
        for AL, AR, C in self.As:
            AL, AR = AL.data[0], AR.data[0]

            # make unitaries
            U = FullStateTensor(tensor_to_unitary(AL))
            V = FullEnvironment(environment_to_unitary(C))

            self.assertTrue(full_tomography_env_objective_function(U, V)<1e-6)
            self.assertTrue(sampled_tomography_env_objective_function(U, V, 10000)<1e-1)


if __name__ == '__main__':
    unittest.main(verbosity=1)
