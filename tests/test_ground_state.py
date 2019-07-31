import unittest 

from qmps.ground_state import *
from qmps.tools import unitary_to_tensor, random_circuit
from scipy import integrate
import numpy as np
from numpy.random import randn
from xmps.iMPS import iMPS
from xmps.iOptimize import find_ground_state
from xmps.spin import N_body_spins
from scipy.linalg import norm
from tqdm import tqdm
Sx1, Sy1, Sz1 = N_body_spins(1/2, 1, 2)
Sx2, Sy2, Sz2 = N_body_spins(1/2, 2, 2)

import matplotlib.pyplot as plt

class TestGroundState(unittest.TestCase):
    def setUp(self):
        N = 3  
        self.xs = [randn(8, 8)+1j*randn(8, 8) for _ in range(N)]
        self.As = [iMPS().random(2, 2).mixed() for _ in range(N)]
        self.verbose=True

    @unittest.skip('slow')
    def test_NonSparseFullEnergyOptimizer(self):
        for AL, AR, C in [self.As[0]]:
            gs = np.linspace(0, 5, 10)
            exact_es = []
            qmps_es = []
            xmps_es = []
            for g in tqdm(gs):
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
                sets['method'] = 'Nelder-Mead'
                sets['verbose'] = self.verbose
                sets['maxiter'] = 5000
                sets['tol'] = 1e-5
                opt.settings(sets)
                opt.get_env()
                tm = iMPS([unitary_to_tensor(opt.U)]).transfer_matrix().asmatrix()

                qmps_es.append(opt.obj_fun_values[-1])
                self.assertTrue(opt.obj_fun_values[-1] > E0_exact-1e-3)
            qmps_norm = norm(np.array(exact_es)-np.array(qmps_es))
            xmps_norm = norm(np.array(exact_es)-np.array(xmps_es))
            print('xmps norm', xmps_norm)
            print('qmps norm', qmps_norm)

            self.assertTrue(qmps_norm < 1e-1 or qmps_norm < xmps_norm)
            plt.plot(gs, exact_es)
            plt.plot(gs, xmps_es)
            plt.plot(gs, qmps_es)
            plt.show()

    @unittest.skip('slow')
    def test_SparseFullEnergyOptimizer(self):
        for AL, AR, C in [self.As[0]]:
            gs = np.linspace(0.1, 5, 10)
            exact_es = []
            qmps_es = []
            xmps_es = []
            for g in tqdm(gs):
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

                opt = SparseFullEnergyOptimizer(H, 2)
                sets = opt._settings_
                sets['store_values'] = True
                sets['method'] = 'Nelder-Mead'
                sets['verbose'] = self.verbose
                sets['maxiter'] = 5000
                sets['tol'] = 1e-5
                opt.settings(sets)
                opt.get_env()

                qmps_es.append(opt.obj_fun_values[-1])
                self.assertTrue(opt.obj_fun_values[-1] > E0_exact-1e-3)
            qmps_norm = norm(np.array(exact_es)-np.array(qmps_es))
            xmps_norm = norm(np.array(exact_es)-np.array(xmps_es))
            print('xmps norm', xmps_norm)
            print('qmps norm', qmps_norm)

            self.assertTrue(qmps_norm < 1e-1 or qmps_norm < xmps_norm)
            plt.plot(gs, exact_es)
            plt.plot(gs, xmps_es)
            plt.plot(gs, qmps_es)
            plt.show()

    def test_Hamiltonian_to_matrix(self):
        J = -1
        g = 1
        H =  np.array([[J,g/2,g/2,0], 
                       [g/2,-J,0,g/2], 
                       [g/2,0,-J,g/2], 
                       [0,g/2,g/2,J]] )

        H_ = Hamiltonian({'ZZ': -1, 'X': 1}).to_matrix()
        H__ = Hamiltonian({'ZZ': -1, 'IX': 1/2, 'XI': 1/2}).to_matrix()

        self.assertTrue(np.allclose(H, H_))
        self.assertTrue(np.allclose(H, H__))
    
    def test_Hamiltonian_measure(self):
        circuit = random_circuit(2, 2)
        qubits = list(circuit.all_qubits())

        N = 2
        for _ in range(N):
            gs = randn(2**4)
            gs/=norm(gs)
            H = Hamiltonian({a+b: gs[i] for i, (a, b) in enumerate(product(*[['X', 'Y', 'Z', 'I']]*2))})
            e = H.measure_energy(circuit, qubits, 10000)
            e_ = H.calculate_energy(circuit)
            print(norm(e-e_))



if __name__=='__main__':
    unittest.main(verbosity=1)
