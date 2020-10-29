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

    def num_qubits(self):
        return 2

    def _decompose_(self, qubits):
        return [
            K(self.gamma).on(*qubits),
            cirq.X.on(qubits[1]),
            K(-self.gamma).on(*qubits),
            cirq.X.on(qubits[1])
        ]

class V(cirq.Gate):
    def __init__(self, params):
        self.gamma = params[0]
        self.psi = params[1]
        self.phi = params[2]
    
    def num_qubits(self):
        return 2

    def _decompose_(self, qubits):
        return [
            expYY(self.gamma).on(*qubits),
            cirq.rx(self.psi).on(qubits[0]),
            cirq.rz(self.phi).on(qubits[0])
        ]

class CPHASE(cirq.Gate):
    def __init__(self, phi, alpha, xi_one, xi_two):
        self.phi = phi
        self.alpha = alpha
        self.xi_one = xi_one
        self.xi_two = xi_two

    def num_qubits(self):
        return 2

    def _decompose_(self, qubits):
        return [
            cirq.rz(-self.phi/2)(qubits[0]),
            cirq.rz(-self.phi/2)(qubits[1]),
            cirq.rx(self.xi_one)(qubits[0]),
            cirq.rx(self.xi_two)(qubits[1]),
            sqrtiSWAPinv(*qubits),
            cirq.rx(-2*self.alpha)(qubits[0]),
            sqrtiSWAPinv(*qubits),
            cirq.rx(self.xi_one)(qubits[0]),
            cirq.rx(-self.xi_two)(qubits[1])
        ]

class CPHASEPure(CPHASE):
    def __init__(self, phi):
        super().__init__(phi, 0,0,0)


class TFIM(cirq.Gate):
    def __init__(self, J, g, dt):
        self.J = J
        self.g = g
        self.dt = dt

    def num_qubits(self):
        return 2

    def _decompose_(self, qubits):
        return [
            cirq.Y(qubits[0]),
            cirq.Y(qubits[1]),
            K(J*dt)(*qubits),
            cirq.X(qubits[1]),
            K(J*dt)(*qubits),
            cirq.X(qubits[0]),
            CPHASEPure(g*dt)(*qubits),
            cirq.X(qubits[0]),
            cirq.X(qubits[1]),
            CPHASEPure(g*dt)(*qubits),
            cirq.Y(qubits[0]),
            cirq.Y(qubits[1])
        ]


if __name__ == "__main__":
    c = cirq.Circuit()
    q = cirq.LineQubit.range(2)
    c.append([
        K(pi/4).on(*q)
    ])

    print(np.round(cirq.unitary(c),3))
