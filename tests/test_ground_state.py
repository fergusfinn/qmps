import unittest 

from qmps.ground_state import *
from scipy import integrate
import numpy as np
from numpy.random import randn
from xmps.iMPS import iMPS

class TestGroundState(unittest.TestCase):
    def setUp(self):
        N = 3  
        self.xs = [randn(8, 8)+1j*randn(8, 8) for _ in range(N)]
        self.As = [iMPS().random(2, 2).mixed() for _ in range(N)]

    def test_optimize_ising(self):
        for AL, AR, C in self.As:
            g = 0.5

            f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
            E0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
            print("E_exact =", E0_exact)

            U, V = optimize_ising_D_2(1, g)

            f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
            E0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
            print("E_exact =", E0_exact)


if __name__=='__main__':
    unittest.main(verbosity=1)
