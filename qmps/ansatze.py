import cirq as cirq
import unittest
import numpy as np
from numpy.random import randn
from scipy.optimize import minimize
from qmps.tools import get_env_exact
from qmps.represent import ShallowFullStateTensor

class Vr(cirq.Gate):
    def __init__(self, params):
        self.params = params
        #assert len(self.params) == 3
        self.n_qubits = 2

    def num_qubits(self):
        return self.n_qubits

    def _decompose_(self, qubits):
        γ, ψ, ϕ, ω, δ, ξ = self.params
        return [[(cirq.YY**γ)(*qubits), 
                 (cirq.X**ψ)(qubits[0]), 
                 (cirq.Z**ϕ)(qubits[0]), 
                 (cirq.Z**(-ξ))(qubits[1]), #arbitrary unitary on the second qubit is possible
                 (cirq.X**(-ω))(qubits[1]), 
                 (cirq.Z**(-δ))(qubits[1])]]

class TestAnsatze(unittest.TestCase):
    def setUp(self):
        N = 10
        self.envs = [get_env_exact(np.linalg.qr(np.random.randn(4, 4)+1j*np.random.randn(4, 4))[0]) for _ in range(N)]

    def test_RightEnvironmentGate2(self):
        qbs = cirq.LineQubit.range(2)
        for exact_Vr in self.envs:
            def circ(v):
                #Vr = lambda v: ShallowFullStateTensor(2, v)
                c = cirq.Circuit.from_ops([cirq.TwoQubitMatrixGate(exact_Vr)(*qbs), 
                                           cirq.inverse(Vr(v)(*qbs))])
                sim = cirq.Simulator()
                ψ = sim.simulate(c).final_state
                #print(1-np.abs(ψ[0])**2)
                return 1-np.abs(ψ[0])**2
            res = minimize(fun=circ, x0=np.random.rand(6), method='Nelder-Mead', tol=1e-10)
            print(res.fun)

if __name__ == '__main__':
    unittest.main()
            

