from .tools import Optimizer, cirq_qubits
from .States import State, ShallowEnvironment, ShallowStateTensor, FullStateTensor, Tensor
import cirq
from .tools import TimeEvolveOptimizer, RepresentMPS, GuessInitialFullParameterOptimizer
from xmps.spin import U4
import numpy as np
from scipy.linalg import expm
from typing import Dict


class MPSTimeEvolve:
    def __init__(self, u_initial: cirq.Gate, hamiltonian: cirq.Gate, v_initial: cirq.Gate = None, depth: int=0,
                 settings=None, optimizer_settings=None,
                 reps=0):
        self.u = u_initial
        self.hamiltonian = hamiltonian

        self.kwargs = {}
        if reps:
            self.kwargs.update({'reps': reps})
        if depth:
            self.kwargs.update({'depth': depth})

        self.optimizer_settings = optimizer_settings if optimizer_settings else \
            {'vertical': 'Vertical', 'ansatz': 'Full', 'simulate': 'Simulate'}
        self.evo_optimizer_settings = self.optimizer_settings.copy()
        self.evo_optimizer_settings.update({'vertical': 'Horizontal'})

        self.TimeEvoOptimizer = None
        self.EnvOptimizer = None

        self.settings = settings
        self.initial_guess_u = self.get_initial_params(self.u)
        self.initial_guess_v = None

        self.v = v_initial
        if not v_initial:
            self.v = self.get_v_params().v

    @staticmethod
    def get_initial_params(u):
        initial_guess_optimizer = GuessInitialFullParameterOptimizer(u)
        initial_guess_optimizer.change_settings({'verbose': True})
        initial_guess_optimizer.optimize()
        return initial_guess_optimizer.optimized_result.x

    def get_v_params(self):
        self.EnvOptimizer = RepresentMPS(self.u, initial_guess=self.initial_guess_v, **self.kwargs)

        if self.settings:
            self.EnvOptimizer.change_settings(self.settings)
        self.EnvOptimizer.optimize()
        self.initial_guess_v = self.EnvOptimizer.optimized_result.x
        return self.EnvOptimizer

    def get_u_params(self):
        self.TimeEvoOptimizer = TimeEvolveOptimizer(self.u, self.v, hamiltonian=self.hamiltonian,
                                                    initial_guess=self.initial_guess_u, **self.kwargs)
        if self.settings:
            self.TimeEvoOptimizer.change_settings(self.settings)

        self.TimeEvoOptimizer.optimize()
        self.initial_guess_u = self.TimeEvoOptimizer.optimized_result.x
        return self.TimeEvoOptimizer

    def evolve_single_step(self):
        self.u = self.get_u_params().u
        self.v = self.get_v_params().v

    def evolve_multiple_steps(self, steps):
        for _ in range(steps):
            self.evolve_single_step()

    def simulate_state(self):
        state = State(self.u, self.v, 1)
        qubits = cirq.LineQubit.range(state.num_qubits())
        circuit = cirq.Circuit.from_ops([state.on(*qubits)])
        simulator = cirq.Simulator()
        return simulator.simulate(circuit), qubits

    def evolve_bloch_sphere(self, evo_steps):
        current_step = 0
        n_qubits = self.hamiltonian.num_qubits()
        qubit_1 = []

        results, qubits = self.simulate_state()
        qb1 = results.bloch_vector_of(qubits[1])
        qubit_1.append(qb1)

        while current_step < evo_steps:
            # evolve a single step
            self.evolve_single_step()

            # simulate the new state
            results, qubits = self.simulate_state()

            # get bloch sphere of physical qubit
            qb1 = results.bloch_vector_of(qubits[1])

            # record results
            qubit_1.append(qb1)

            current_step += 1

        x_evo = [step[0] for step in qubit_1]
        y_evo = [step[1] for step in qubit_1]
        z_evo = [step[2] for step in qubit_1]
        return x_evo, y_evo, z_evo

    def loschmidt_echo(self, steps):
        '''
        Search for loshmidt echos in Ising Hamiltonian
        :param steps: time steps

        Value that is being evaluated is the square of the Loschmidt amplitude:
        https://royalsocietypublishing.org/doi/pdf/10.1098/rsta.2015.0160

        need to get ground state of Ising hamiltonian with lambda<0.5, then evolve with lambda > 0.5
        '''
        original_state, original_qubits = self.simulate_state()

        original_wavefunction = original_state.final_simulator_state.state_vector

        state_overlap = []
        current_step = 0
        while current_step < steps:
            self.evolve_single_step()
            new_state, _ = self.simulate_state()
            new_wavefunction = new_state.final_simulator_state.state_vector

            overlap = np.abs(np.dot(original_wavefunction, new_wavefunction.conj()))**2
            state_overlap.append(overlap)
            current_step += 1
            print(current_step)
        return state_overlap
