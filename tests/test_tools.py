import unittest

from numpy.random import randn
from numpy import allclose, abs, sum
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

    def test_unitary_to_tensor(self):
        N = 10
        As = []
        for i in range(N):
            U = random_unitary(4, 4)._unitary_()
            if U.shape[0] == 16:
                As.append(unitary_to_tensor(U))
        
        for A in As:
            self.assertTrue(allclose(A.shape, (2, 8, 8)))

    def test_sampled_bloch_vector_of(self):
        circuit = random_circuit(5)
        sim = cirq.Simulator()
        v1 = sim.simulate(circuit).bloch_vector_of(list(circuit.all_qubits())[0])
        v2 = sampled_bloch_vector_of(list(circuit.all_qubits())[0], circuit, 100000)
        self.assertTrue(sum(abs(v1-v2))<0.1)

if __name__=='__main__':
    unittest.main(verbosity=1)
