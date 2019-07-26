from qmps.tools import Optimizer, cirq_qubits
from qmps.represent import State, ShallowEnvironment, ShallowStateTensor, VerticalSwapOptimizer, FullStateTensor, Tensor
import cirq
from xmps.spin import U4
import numpy as np
from typing import Dict


class TimeEvolveOptimizer(Optimizer):
    def __init__(self, u_original: cirq.Gate, v_original: cirq.Gate, hamiltonian: cirq.Gate, depth: int = 0, **kwargs):
        super().__init__(u_original, v_original, depth=depth, **kwargs)
        self.hamiltonian = hamiltonian

    def objective_function(self, params):
        ham_qubits = self.hamiltonian.num_qubits()
        aux_qubits = int(self.v.num_qubits()/2)
        if self.full_param:
            u_target = FullStateTensor(U4(params))
        else:
            u_target = ShallowStateTensor(self.bond_dim, params)
        target_state = State(u_target, self.v, ham_qubits)

        original_state = State(self.u, self.v, ham_qubits)
        qubits = cirq_qubits(original_state.num_qubits())

        swap_state = cirq.Circuit.from_ops([original_state(*qubits),
                                           self.hamiltonian(*qubits[aux_qubits:aux_qubits+ham_qubits]),
                                           cirq.inverse(target_state).on(*qubits)])

        self.circuit = swap_state
        simulator = cirq.Simulator()
        results = simulator.simulate(self.circuit)

        final_state = results.final_simulator_state.state_vector[0]
        score = np.abs(final_state)**2
        return 1 - score

    def update_final_circuits(self):
        u_params = self.optimized_result.x
        if self.full_param:
            self.u = FullStateTensor(U4(u_params))
        else:
            self.u = ShallowStateTensor(self.bond_dim, u_params)


class MPSTimeEvolve:
    def __init__(self, u_initial: cirq.Gate, hamiltonian: cirq.Gate, v_initial: cirq.Gate = None, depth: int=0,
                 settings: Dict = None):
        self.u = u_initial
        self.hamiltonian = hamiltonian
        self.depth = depth
        self.TimeEvoOptimizer = None
        self.EnvOptimizer = None
        self.settings = settings
        self.bond_dim = 2 ** (u_initial.num_qubits() - 1)
        # has to be initiated last or it won't know settings exist: annoying error
        self.v = v_initial
        if not v_initial:
            self.v = self.get_v_params().v

    def get_v_params(self):
        v = self.v if self.v else ShallowEnvironment(self.bond_dim, np.random.rand(2 * self.depth))
        self.EnvOptimizer = VerticalSwapOptimizer(u_original=self.u, v_original=v, depth=self.depth,
                                                  initial_guess=np.random.rand(15))  # find a way to remove this

        if self.settings:
            self.EnvOptimizer.settings(self.settings)

        self.EnvOptimizer.get_env()
        return self.EnvOptimizer

    def get_u_params(self):
        self.TimeEvoOptimizer = TimeEvolveOptimizer(u_original=self.u, v_original=self.v, hamiltonian=self.hamiltonian,
                                                    depth=self.depth,
                                                    initial_guess=np.random.rand(15))  # and this
        if self.settings:
            self.TimeEvoOptimizer.settings(self.settings)

        self.TimeEvoOptimizer.get_env()
        return self.TimeEvoOptimizer

    def evolve_single_step(self):
        self.u = self.get_u_params().u
        self.v = self.get_v_params().v

    def evolve_multiple_steps(self, steps):
        for _ in range(steps):
            self.evolve_single_step()

    def evolve_bloch_sphere(self, evo_steps):
        current_step = 0
        n_qubits = self.hamiltonian.num_qubits()
        qubit_1 = []

        while current_step < evo_steps:
            # evolve a single step
            self.evolve_single_step()

            # prepare new optimized state
            state = State(self.u, self.v, n=n_qubits)

            # simulate state
            qubits = cirq_qubits(state.num_qubits())
            circuit = cirq.Circuit.from_ops([state.on(*qubits)])
            simulator = cirq.Simulator()
            results = simulator.simulate(circuit)

            # get bloch sphere of physical qubit
            qubits = cirq_qubits(state.num_qubits())
            qb1 = results.bloch_vector_of(qubits[1])

            # record results
            qubit_1.append(qb1)
            current_step += 1

        x_evo = [step[0] for step in qubit_1]
        y_evo = [step[1] for step in qubit_1]
        z_evo = [step[2] for step in qubit_1]
        return x_evo, y_evo, z_evo
