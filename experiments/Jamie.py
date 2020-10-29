import cirq
from numpy import pi
import numpy as np
from sympy import Symbol, Matrix

sqrtiSWAP = cirq.ISwapPowGate(exponent = 0.5)    
sqrtiSWAPinv = cirq.ISwapPowGate(exponent = -0.5)    

class K(cirq.Gate):
    def __init__(self, theta):
        self.theta = theta

    def num_qubits(self):
        return 2

    def _decompose_(self, qubits):
        return [
            cirq.rz(-pi/4)(qubits[0]),
            cirq.rz(pi/4)(qubits[1]),
            sqrtiSWAP.on(*qubits),
            cirq.rz(self.theta)(qubits[0]),
            cirq.rz(-self.theta)(qubits[1]),
            sqrtiSWAPinv.on(*qubits),
            cirq.rz(pi/4)(qubits[0]),
            cirq.rz(-pi/4)(qubits[1])
        ]

class expYY(cirq.Gate):
    def __init__(self, gamma):
        self.gamma = gamma

    def num_qubits(self:
        return 2

    def _decompose_(self, qubits):
        return [
            K()
        ]


if __name__ == "__main__":
    c = cirq.Circuit()
    q = cirq.LineQubit.range(2)
    c.append([
        K(pi.4).on(*q)
    ])

    print(np.round(cirq.unitary(c),3))
