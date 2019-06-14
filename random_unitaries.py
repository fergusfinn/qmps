from cirq.devices import GridQubit
from cirq import Rx, Ry, Rz, Circuit, CNOT
from numpy.random import rand, randint
from numpy import pi as π, arcsin, sqrt
import numpy as np

from xmps.fMPS import fMPS
from xmps.spin import spins

Sx, Sy, Sz = spins(0.5)

L = 5
D = 2
d = 2


def random_unitary(length, depth=10, p=0.5):
    '''10.1103/PhysRevA.75.062314'''
    qubits = [GridQubit(i, 0) for i in range(length)]
    circuit = Circuit()

    def U(i):
        """U: Random SU(2) element"""
        ψ = 2*π*rand()
        χ = 2*π*rand()
        φ = arcsin(sqrt(rand()))
        for g in [Rz(χ+ψ), Ry(2*φ), Rz(χ-ψ)]:
            yield g(GridQubit(i, 0))
    for i in range(depth):
        if rand()>p:
            # one qubit gate
            circuit.append(U(randint(0, length)))
        else:
            # two qubit gate
            i = randint(0, length-1)
            if rand()>0.5:
                circuit.append(CNOT(qubits[i], qubits[i+1]))
            else:
                circuit.append(CNOT(qubits[i+1], qubits[i]))
    return circuit

def random_mps(L, d=2, D=2, lr = 'r'):
    '''create a random mps TODO: check right/left canonical are right'''
    gates = [random_unitary(j) for j in range(L)]
    if lr=='l':
        return reduce(lambda x, y: x+y, gates)
    else:
        return reduce(lambda x, y: x+y, reversed(gates))
