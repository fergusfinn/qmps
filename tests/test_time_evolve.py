import unittest
import numpy as np
from xmps.spin import spins
from xmps.iMPS import iMPS
from scipy.linalg import expm
from ..qmps.represent import Tensor
from ..qmps.time_evolve import
import matplotlib.pyplot as plt

Sx, Sy, Sz = spins(0.5)


class TestTimeEvolve(unittest.TestCase):
    def setUp(self):
        N = 3  
        self.As = [iMPS().random(2, 2).mixed() for _ in range(N)]

    def test_time_evolve(self):
        for AL, AR, C in [self.As[0]]:
            J, g = -1, 0.5
            H = np.array([[J, g/2, g/2, 0],
                          [g/2, -J, 0, g/2],
                          [g/2, 0, -J, g/2],
                          [0, g/2, g/2, J]])


            T = np.linspace(0, 5, 2000)
            dt = T[1]-T[0]
            evs = []
            es = []
            A = AL
            for _ in T:
                dA = A.dA_dt([H])*dt
                es.append(A.e)
                A = (A+dA).left_canonicalise()
                evs.append(A.Es([Sx, Sy, Sz]))

            fig, ax = plt.subplots(2, 1, sharex=True)
            ax[0].plot(np.array(evs))
            ax[1].plot(es)
            plt.show()
            '''
            Below is the cirq time evolution. Gate is made using scipy.expm(H) * dt which is put into a Tensor, and 
            and this is the hamiltonian.
            '''

if __name__ == '__main__':
    unittest.main(verbosity=1)
