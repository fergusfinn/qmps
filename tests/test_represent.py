from qmps.represent import *
from xmps import Map


class TestRepresent(unittest.TestCase):
    def setUp(self):
        N = 3  
        self.xs = [randn(8, 8)+1j*randn(8, 8) for _ in range(N)]
        self.As = [iMPS().random(2, 2).mixed() for _ in range(N)]

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
            U = unitary_from_tensor(AL[0])
            V = environment_to_unitary(C)

            self.assertTrue(full_env_obj_fun(U, V)<1e-6)
