import unittest

from numpy.random import randn
from numpy import allclose
from xmps.iMPS import iMPS

from qmps.tools import *

class TestTools(unittest.TestCase):
    def setUp(self):
        N = 3  
        self.xs = [randn(8, 8)+1j*randn(8, 8) for _ in range(N)]
        self.As = [iMPS().random(2, 2).mixed() for _ in range(N)]

    def test_tensor_to_unitary(self):
        for AL, AR, C in self.As:
            U, passed = tensor_to_unitary(AL[0], testing=True)
            self.assertTrue(passed)
            AL_new = unitary_to_tensor(U)
            self.assertTrue(allclose(AL_new, AL[0]))

if __name__=='__main__':
    unittest.main(verbosity=1)
