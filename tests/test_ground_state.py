import unittest 

from qmps.ground_state import *
from scipy import integrate
import numpy as np
from numpy.random import randn
from xmps.iMPS import iMPS

import matplotlib.pyplot as plt

class TestGroundState(unittest.TestCase):
    def setUp(self):
        N = 3  
        self.xs = [randn(8, 8)+1j*randn(8, 8) for _ in range(N)]
        self.As = [iMPS().random(2, 2).mixed() for _ in range(N)]

    def test_optimize_ising(self):
        for AL, AR, C in [self.As[0]]:
            gs = np.linspace(0, 2, 10)
            exact_es = []
            mps_es = []
            for g in gs:
                f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
                E0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
                print(E0_exact)

                opt = NonSparseFullEnergyOptimizer(-1, g)
                sets = opt._settings_
                sets['store_values'] = True
                sets['method'] = 'Nelder-Mead'
                sets['verbose'] = True
                sets['maxiter'] = 4000
                sets['tol'] = 1e-4
                opt.settings(sets)
                opt.get_env()
                print(opt.obj_fun_values[-1], E0_exact)
                mps_es.append(opt.obj_fun_values[-1])
                exact_es.append(E0_exact)
            np.save('exact', np.array(exact_es))
            np.save('calc', np.array(mps_es))
            plt.plot(gs, exact_es)
            plt.plot(gs, mps_es)
            plt.show()

if __name__=='__main__':
    unittest.main(verbosity=1)
