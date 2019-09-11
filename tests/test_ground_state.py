import unittest 

from qmps.ground_state import *
from qmps.tools import unitary_to_tensor, random_circuit, random_full_rank_circuit
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

    @unittest.skip('x')
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

    @unittest.skip('x')
    def test_PauliMeasure(self):
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit().from_ops([cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1])])
        sim = cirq.Simulator()

        strings = list(map(lambda x: x[0]+x[1], list(product(*[['I', 'X', 'Y', 'Z']]*2))))
        strings.remove('II')
        for string in strings:
            ψ = sim.simulate(circuit).final_state
            O = Hamiltonian({string: 1}).to_matrix()
            ev = np.around(np.real(ψ.conj().T@O@ψ), 3)

            c = circuit.copy()
            c.append([PauliMeasure(string)(*qubits), cirq.measure(qubits[0], key=string)])
            meas = sim.run(c, repetitions=10000).measurements[string]
            ev_ = np.around(array(list(map(lambda x: 1-2*int(x), meas))).mean(), 3)
            diff = norm(ev-ev_)
            self.assertTrue(norm(ev-ev_)<5e-2)

        circuit = random_circuit(2, 4)
        sim = cirq.Simulator()

        strings = list(map(lambda x: x[0]+x[1], list(product(*[['I', 'X', 'Y', 'Z']]*2))))
        strings.remove('II')
        for string in strings:
            ψ = sim.simulate(circuit).final_state
            O = Hamiltonian({string: 1}).to_matrix()
            ev = np.around(np.real(ψ.conj().T@O@ψ), 3)
            c = circuit.copy()
            c.append([PauliMeasure(string)(*qubits), cirq.measure(qubits[0], key=string)])
            meas = sim.run(c, repetitions=10000).measurements[string]
            ev_ = np.around(array(list(map(lambda x: 1-2*int(x), meas))).mean(), 3)
            diff = norm(ev-ev_)
            self.assertTrue(norm(ev-ev_)<5e-2)
    
    @unittest.skip('x')
    def test_Hamiltonian_measure(self):
        qubits = cirq.LineQubit.range(2)
        circuit = random_circuit(2, 4)
        sim = cirq.Simulator()

        N = 2
        for _ in range(N):
            gs = randn(2**4)
            strings = list(map(lambda x: x[0]+x[1], list(product(*[['I', 'X', 'Y', 'Z']]*2))))
            strings.remove('II')
            H = Hamiltonian({a+b: gs[i] for i, (a, b) in enumerate(strings)})
            e = H.measure_energy(circuit, qubits, reps=300000)
            e_ = H.calculate_energy(circuit)
            self.assertTrue(norm(e-e_)<5e-2)

    @unittest.skip('x')
    def test_NonSparseFullEnergyOptimizer(self):
        for AL, AR, C in [self.As[0]]:
            gs = np.linspace(0, 2, 20)
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


                #ψ, e = find_ground_state(H, 2)
                #xmps_es.append(e[-1])

                opt = NonSparseFullEnergyOptimizer(H, 4)
                sets = opt.settings
                sets['store_values'] = True
                sets['method'] = 'Nelder-Mead'
                sets['verbose'] = self.verbose
                sets['maxiter'] = 5000
                sets['tol'] = 1e-5

                opt.change_settings(sets)
                opt.optimize()
                tm = iMPS([unitary_to_tensor(opt.U)]).transfer_matrix().asmatrix()
                qmps_es.append(opt.obj_fun_values[-1])
            plt.plot(gs, exact_es, label='exact')
            #plt.plot(gs, xmps_es)
            plt.plot(gs, qmps_es, label='optimized')
            plt.xlabel('$\\lambda$')
            plt.ylabel('$E_0$')
            plt.legend()
            plt.savefig('/Users/fergusbarratt/Desktop/gs_opt.pdf', bbox_inches='tight')
            plt.show()

    @unittest.skip('x')
    def test_NoisyNonSparseFullEnergyOptimizer_no_noise(self):
        for AL, AR, C in [self.As[0]]:
            J, g = -1, 1 
            H =  np.array([[J,g/2,g/2,0], 
                           [g/2,-J,0,g/2], 
                           [g/2,0,-J,g/2], 
                           [0,g/2,g/2,J]] )

            opt_noisy = NoisyNonSparseFullEnergyOptimizer(H, 0.)
            opt_clean = NonSparseFullEnergyOptimizer(H)
            N = 10 
            for _ in range(N):
                x = randn(15)
                self.assertTrue(np.allclose(opt_noisy.objective_function(x), opt_clean.objective_function(x)))

    def test_NoisyNonSparseFullEnergyOptimizer(self):
        for AL, AR, C in [self.As[0]]:
            gs = np.linspace(0, 2, 10)
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

    #            ψ, e = find_ground_state(H, 2)
    #            xmps_es.append(e[-1])

                opt = NoisyNonSparseFullEnergyOptimizer(H, 1e-2)
                sets = opt.settings
                sets['store_values'] = True
                sets['method'] = 'Nelder-Mead'
                sets['verbose'] = True
                #sets['maxiter'] = 5000
                #sets['tol'] = 1e-5

                opt.change_settings(sets)
                opt.optimize()

                qmps_es.append(opt.obj_fun_values[-1])
                #self.assertTrue(opt.obj_fun_values[-1] > E0_exact-1e-3)
            qmps_norm = norm(np.array(exact_es)-np.array(qmps_es))
    #        xmps_norm = norm(np.array(exact_es)-np.array(xmps_es))
    #        print('xmps norm', xmps_norm)
            print('qmps norm', qmps_norm)

            #self.assertTrue(qmps_norm < 1e-1 or qmps_norm < xmps_norm)
            plt.plot(gs, exact_es)
    #        plt.plot(gs, xmps_es)
            plt.plot(gs, qmps_es)
            plt.show()

    @unittest.skip('x')
    def test_SparseFullEnergyOptimizer(self):
        for AL, AR, C in [self.As[0]]:
            gs = np.linspace(0.2, 2, 10)
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


                #ψ, e = find_ground_state(H, 2)
                #xmps_es.append(e[-1])

                opt = SparseFullEnergyOptimizer(H, 4, 4)
                sets = opt.settings
                sets['store_values'] = True
                sets['method'] = 'Nelder-Mead'
                sets['verbose'] = self.verbose
                sets['maxiter'] = 5000
                sets['tol'] = 1e-5
                opt.change_settings(sets)
                opt.optimize()

                qmps_es.append(opt.obj_fun_values[-1])
                self.assertTrue(opt.obj_fun_values[-1] > E0_exact-1e-3)
            qmps_norm = norm(np.array(exact_es)-np.array(qmps_es))
            #xmps_norm = norm(np.array(exact_es)-np.array(xmps_es))
            #print('xmps norm', xmps_norm)
            print('qmps norm', qmps_norm)

            #self.assertTrue(qmps_norm < 1e-1 or qmps_norm < xmps_norm)
            plt.title('D=4, qaoa depth 3 ansatz')
            plt.plot(gs, exact_es)
            plt.xlabel('$\\lambda$')
            plt.ylabel('$E_0$')
            #plt.plot(gs, xmps_es)
            plt.plot(gs, qmps_es)
            plt.show()

if __name__=='__main__':
    unittest.main(verbosity=2)
