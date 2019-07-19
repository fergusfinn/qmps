from .tools import Optimizer, cirq_qubits
from .represent import State, ShallowEnvironment, ShallowStateTensor, VerticalSwapOptimizer
import cirq
import numpy as np
from typing import Dict


class TimeEvolveOptimizer(Optimizer):
    def __init__(self, u_original: cirq.Gate, v_original: cirq.Gate, hamiltonian: cirq.Gate, **kwargs):
        super().__init__(u_original, v_original, **kwargs)
        self.hamiltonian = hamiltonian

    def objective_function(self, params):
        ham_qubits = self.hamiltonian.num_qubits()
        aux_qubits = int(self.v.num_qubits()/2)

        u_target = ShallowStateTensor(self.bond_dim, params)
        target_state = State(u_target, self.v, ham_qubits)

        original_state = State(self.u, self.v, ham_qubits)
        qubits = cirq_qubits(original_state.num_qubits())

        swap_state = cirq.Circuit.from_ops([original_state(*qubits),
                                           self.hamiltonian(*qubits[aux_qubits:aux_qubits+ham_qubits]),
                                           cirq.inverse(target_state)])

        self.circuit = swap_state
        simulator = cirq.Simulator()
        results = simulator.simulate(self.circuit)

        final_state = results.final_simulator_state.state_vector[0]
        score = np.abs(final_state)**2
        return 1 - score

    def update_final_circuits(self):
        u_params = self.optimized_result.x
        self.u = ShallowStateTensor(self.bond_dim, u_params)


class MPSTimeEvolve:
    def __init__(self, u_initial: cirq.Gate, hamiltonian: cirq.Gate, v_initial: cirq.Gate = None, qaoa_depth: int=1,
                 settings: Dict = None):
        self.u = u_initial
        self.hamiltonian = hamiltonian
        self.qaoa_depth = qaoa_depth
        self.v = v_initial if v_initial else self.get_v_params().v
        self.TimeEvoOptimizer = None
        self.EnvOptimizer = None
        self.settings = settings

    def get_v_params(self):
        self.EnvOptimizer = VerticalSwapOptimizer(self.u, self.v, qaoa_depth=self.qaoa_depth, settings=self.settings)
        return self.EnvOptimizer

    def get_u_params(self):
        self.TimeEvoOptimizer = TimeEvolveOptimizer(self.u, self.v, self.hamiltonian, qaoa_depth=self.qaoa_depth,
                                                    settings=self.settings)
        self.TimeEvoOptimizer.get_env()
        return self.TimeEvoOptimizer

    def evolve_single_step(self):
        self.u = self.get_u_params().u
        self.v = self.get_v_params().v

    def evolve_multiple_steps(self, steps):
        for _ in range(steps):
            self.evolve_single_step()
