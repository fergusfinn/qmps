import cirq
from numpy import pi
import numpy as np
from sympy import Symbol, Matrix
from cirq.contrib.svg import SVGCircuit

sqrtiSWAP = cirq.ISwapPowGate(exponent = 0.5)    
sqrtiSWAPinv = cirq.ISwapPowGate(exponent = -0.5)    

def round(matrix):
    return np.round(matrix,3)

class ParametrisedCircuit(cirq.Gate):
    """
    This circuit parametrises an MPS state using repeated layers of Ry 
    gates and sqrt(iSWAP) gate as are available on Google's device
    """
    def __init__(self, depth, params):
        """
        params = [[Θ1, Θ2],...], 2 params for each layer
        """
        assert len(params) == depth
        self.d = depth
        self.p = params

    def num_qubits(self):
        return 2

    def _decompose_(self, qubits):
        gates = []
        for p in self.p:
            gates += [cirq.ry(p[0]).on(qubits[0]),
                      cirq.ry(p[1]).on(qubits[1]),
                      sqrtiSWAP.on(*qubits)]
        return gates


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

class CPHASEExact(CPHASE):
    def __init__(self, phi):
        self.phi = phi

    def num_qubits(self):
        return 2

    def _decompose_(self, qubits):
        return [cirq.CZPowGate(exponent=self.phi / pi)(*qubits)]

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
            K(self.J*self.dt)(*qubits),
            cirq.X(qubits[1]),
            K(self.J*self.dt)(*qubits),
            cirq.X(qubits[0]),
            CPHASEExact(self.g*self.dt / pi)(*qubits),
            cirq.X(qubits[0]),
            cirq.X(qubits[1]),
            CPHASEExact(self.g*self.dt / pi)(*qubits),
            cirq.Y(qubits[0]),
            cirq.Y(qubits[1])
        ]

def tests():

    def testK():
        exactK = lambda theta: np.array([
        [1,0,0,0],
        [0, np.cos(theta), -1j*np.sin(theta),0],
        [0, -1j*np.sin(theta), np.cos(theta),0],
        [0,0,0,1] 
        ])

        params = np.random.rand(5)
        for p in params:
            q =  cirq.LineQubit.range(2)
            c = cirq.Circuit()
            c.append(K(p).on(*q))

            assert np.allclose(cirq.unitary(c) - exactK(p), 0)

        print("K Gate is correct")

    def testExpYY():
        expYYexact = lambda theta: np.array([
            [np.cos(theta),0,0,1j*np.sin(theta)],
            [0,np.cos(theta),-1j*np.sin(theta),0],
            [0,-1j*np.sin(theta), np.cos(theta),0],
            [1j*np.sin(theta),0,0,np.cos(theta)]
        ]) 

        for p in np.random.rand(5):
            q = cirq.LineQubit.range(2)
            c = cirq.Circuit()
            c.append([expYY(p).on(*q)])
        
            assert np.allclose(cirq.unitary(c) - expYYexact(p), 0)

        print("expYY Gate is correct")

    testK()
    testExpYY()

if __name__ == "__main__":
    # Representing Circuit:
    q = cirq.LineQubit.range(5)
    c = cirq.Circuit()
    state_symbols = [[Symbol("Θ1"), Symbol("Θ2")], [Symbol("Θ3"), Symbol("Θ4")]]
    env_symbols = [Symbol("γ"), Symbol("ϕ1"), Symbol("ϕ2")]
    c.append([
        V(env_symbols)(*q[2:4]),
        ParametrisedCircuit(2, state_symbols)(*q[3:5]),
        V(env_symbols)(*q[0:2])
    ])
