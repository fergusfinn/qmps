import unittest
import numpy as np
from xmps.spin import spins
from xmps.iMPS import iMPS
from scipy.linalg import expm
from qmps.represent import FullStateTensor
from qmps.time_evolve import MPSTimeEvolve
import matplotlib.pyplot as plt
from xmps.tensor import embed

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

            T = np.linspace(0, 1, 10)
            dt = T[1]-T[0]
            evs = []
            es = []
            A = AL

            ################ CIRQ BIT ######################

            counter = 0
            bloch_sphere_results = []
            U = FullStateTensor(embed(A.data[0]))
            V = FullStateTensor(embed(C))  # don't know if this is correct, will just find optimum V variationally
            hamiltonian = FullStateTensor(expm(1j * H * dt))
            evolver = MPSTimeEvolve(u_initial=U, hamiltonian=hamiltonian, v_initial=V, settings={
                'method': 'Nelder-Mead',
                'maxiter': 500
            })

            for _ in T:
                print(counter)
                dA = A.dA_dt([H])*dt
                es.append(A.e)
                A = (A+dA).left_canonicalise()
                evs.append(A.Es([Sx, Sy, Sz]))
                '''
                Below is the cirq time evolution. Gate is made using scipy.expm(H) * dt which is put into a Tensor, and 
                and this is the hamiltonian.
                '''
                evolver.evolve_single_step()
                results, qubits = evolver.simulate_state()
                bloch_sphere = results.bloch_vector_of(qubits[1])
                bloch_sphere_results.append(bloch_sphere)
                counter += 1

            fig, ax = plt.subplots(3, 1, sharex=True)
            ax[0].plot(np.array(evs))
            ax[1].plot(np.array(bloch_sphere_results))
            ax[2].plot(es)
            plt.show()


if __name__ == '__main__':
    unittest.main(verbosity=1)
