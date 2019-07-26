import unittest 

from qmps.ground_state import *
from qmps.tools import unitary_to_tensor
from scipy import integrate
import numpy as np
from numpy.random import randn
from xmps.iMPS import iMPS
from xmps.iOptimize import find_ground_state
from xmps.spin import N_body_spins
from scipy.linalg import norm
Sx1, Sy1, Sz1 = N_body_spins(1/2, 1, 2)
Sx2, Sy2, Sz2 = N_body_spins(1/2, 2, 2)

import matplotlib.pyplot as plt

class TestGroundState(unittest.TestCase):
    def setUp(self):
        N = 3  
        self.xs = [randn(8, 8)+1j*randn(8, 8) for _ in range(N)]
        self.As = [iMPS().random(2, 2).mixed() for _ in range(N)]

    def test_optimize_ising(self):
        for AL, AR, C in [self.As[0]]:
            gs = [2]
            exact_es = []
            qmps_es = []
            xmps_es = []
            for g in gs:
                J, g = -1, g
                f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
                E0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
                exact_es.append(E0_exact)
                H =  np.array([[J,g/2,g/2,0], 
                               [g/2,-J,0,g/2], 
                               [g/2,0,-J,g/2], 
                               [0,g/2,g/2,J]] )


                ψ, e = find_ground_state(H, 2)
                xmps_es.append(e[-1])

                opt = NonSparseFullEnergyOptimizer(H)
                sets = opt._settings_
                sets['store_values'] = True
                sets['method'] = 'Powell'
                sets['verbose'] = True
                sets['maxiter'] = 5000
                sets['tol'] = 1e-5
                opt.settings(sets)
                opt.get_env()
                tm = iMPS([unitary_to_tensor(opt.U)]).transfer_matrix().asmatrix()

                print(opt.obj_fun_values[-1], e[-1], E0_exact)
                qmps_es.append(opt.obj_fun_values[-1])
            #np.save('exact', np.array(exact_es))
            #np.save('calc', np.array(mps_es))
            plt.plot(gs, exact_es)
            plt.plot(gs, xmps_es)
            plt.plot(gs, qmps_es)
            plt.show()

if __name__=='__main__':
    unittest.main(verbosity=1)
