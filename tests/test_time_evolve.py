import unittest
from scipy.linalg import expm
from qmps.represent import FullStateTensor, FullEnvironment, get_env_exact
from qmps.tools import environment_to_unitary, tensor_to_unitary
from qmps.time_evolve import MPSTimeEvolve
from xmps.spin import spins
from xmps.iMPS import iMPS
import numpy as np
import cirq
import matplotlib.pyplot as plt

Sx, Sy, Sz = spins(0.5)
Sx, Sy, Sz = 2*Sx, 2*Sy, 2*Sz


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

            T = np.linspace(0, 1, 100)
            dt = T[1]-T[0]
            evs = []
            es = []

            ################ CIRQ BIT ######################

            counter = 0
            bloch_sphere_results = []
            U = FullStateTensor(tensor_to_unitary(AL.left_canonicalise().data[0]))
            V = FullEnvironment(get_env_exact(cirq.unitary(U)))
            hamiltonian = FullStateTensor(expm(-1j * H * dt))
            evolver = MPSTimeEvolve(u_initial=U, hamiltonian=hamiltonian, v_initial=V, settings={
                'method': 'Nelder-Mead',
                'maxiter': 500
            })

            A = AL
            for _ in T:
                print(counter)
                evs.append(iMPS([A.data[0]]).Es([Sx, Sy, Sz]))
                dA = A.dA_dt([H])*dt
                es.append(A.e)
                A = (A+dA).left_canonicalise()
                '''
                Below is the cirq time evolution. Gate is made using scipy.expm(H) * dt which is put into a Tensor, and 
                and this is the hamiltonian.
                '''
                results, qubits = evolver.simulate_state()
                bloch_sphere = results.bloch_vector_of(qubits[1])
                bloch_sphere_results.append(bloch_sphere)
                evolver.evolve_single_step()
                counter += 1

            fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
            ax[0].plot(np.array(evs), c='r')
            ax[0].plot(np.array(bloch_sphere_results), c='b')
            ax[1].plot(es)
            plt.show()


if __name__ == '__main__':
    unittest.main(verbosity=1)
