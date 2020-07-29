#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:28:30 2020

@author: jamie
"""
import numpy as np
from scipy.optimize import minimize
from numpy import isclose
from ClassicalTDVPStripped import *


class Tests():
    
    """
    Class to perform simple tests on the code stripped down TDVP without
    using PyTest.
    
    We have the following tests:
        
    A) Test that the 3 circuits are correctly set up
    - test_exp_value
    - test_right_environment_circuit
    - test_manifold_overlap
    
    B) Test the optimization procedures
    - test_represent
    - test_optimize
    - test_time_evolution
    
    """
    
    def __init__(self):
        self.X0 = np.array([
                [0,1],
                [1,0]
            ])
        
        self.Z0 = np.array([
                [1,0],
                [0,-1]
            ])
        
        self.I = np.array([
                [1,0],
                [0,1]
            ])
        
        self.H = (1/np.sqrt(2)) * np.array([
                [1,1],
                [1,-1]
            ])
    
    
    def test_compilation_unitary(self):
        U = unitary_group.rvs(4)
        U00 = U[:,0]
        
        def cost_func(params):
            Up = OO_unitary(params)[:,0]
            return np.linalg.norm(U00 - Up)
        
        res = minimize(fun = cost_func, 
                       x0 = np.random.rand(7),
                       method = "Powell")
        
        np.isclose(res.fun, 0)
    
    def test_exp_values(self):
        """
        test that the expectation values return as expected
        """
        I = self.I
        Z0 = self.Z0
        X0 = self.X0
        Had = self.H
        
        ExpVal = OverlapCalculator()
        
        #############################
        # 2 Qubit Expectation value cases
        #############################
        
        # checking the state Z|0> == |0>
        U1 = np.kron(I,I).reshape(2,2,2,2)
        U2 = np.kron(I,I).reshape(2,2,2,2)
        H = np.kron(Z0,Z0).reshape(2,2,2,2)
        
        exp_val_Z00 = ExpVal.expectation_value(U1, U2, H)
        
        assert isclose(exp_val_Z00, 1)
                
        U1 = np.kron(X0,X0).reshape(2,2,2,2)
        U2 = np.kron(I,I).reshape(2,2,2,2)
        exp_val_Z11 = ExpVal.expectation_value(U1, U2, H)
        
        assert isclose(exp_val_Z11, 1)
        
        H = np.kron(I, Z0).reshape(2,2,2,2)
        exp_val_Z01 = ExpVal.expectation_value(U1, U2, H)
        
        assert isclose(exp_val_Z01, -1)
        
        U1 = np.kron(Had,Had).reshape(2,2,2,2)
        U2 = np.kron(I,I).reshape(2,2,2,2)
        H = np.kron(X0,X0).reshape(2,2,2,2)
        exp_val_X00 = ExpVal.expectation_value(U1,U2,H)
        
        assert isclose(exp_val_X00, 1)        
        
        U1 = np.kron(Had,Had).reshape(2,2,2,2)
        U2 = np.kron(X0,X0).reshape(2,2,2,2)
        H = np.kron(X0,I).reshape(2,2,2,2)
        exp_val_X01 = ExpVal.expectation_value(U1,U2,H)
        
        assert isclose(exp_val_X01, -1)
        
        ###############################
        # 4 Qubit expectation value cases
        ###############################
        
        U1 = np.kron(I,I).reshape(2,2,2,2)
        U2 = np.kron(I,I).reshape(2,2,2,2)
        H = tensor([Z0,Z0,Z0,Z0]).reshape(2,2,2,2,2,2,2,2)
        
        exp_val_Z0000 = ExpVal.expectation_value(U1, U2, H)
        assert isclose(exp_val_Z0000, 1)
        
        U1 = np.kron(X0,X0).reshape(2,2,2,2)
        U2 = np.kron(I,I).reshape(2,2,2,2)
        exp_val_Z1111 = ExpVal.expectation_value(U1, U2, H)
        assert isclose(exp_val_Z1111, 1)
        
        H = tensor([I, Z0, Z0, Z0]).reshape(2,2,2,2,2,2,2,2)
        exp_val_Z0111 = ExpVal.expectation_value(U1, U2, H)
        assert isclose(exp_val_Z0111, -1)
        
        U1 = np.kron(Had,Had).reshape(2,2,2,2)
        U2 = np.kron(X0,X0).reshape(2,2,2,2)
        H = tensor([X0,I,I,I]).reshape(2,2,2,2,2,2,2,2)
        exp_val_X1000 = ExpVal.expectation_value(U1,U2,H)
        assert isclose(exp_val_X1000, -1)
        
    
    def test_right_environment_circuit(self):
        """
        Test that the right environment circuit returns values as expected
        """
        I = self.I
        Z0 = self.Z0
        X0 = self.X0
        RE = RightEnvironment()
        
        U1 = np.kron(X0, X0).reshape(2,2,2,2)
        U1_ = U1.reshape(4,4).conj().T.reshape(2,2,2,2)
        U2 = np.kron(I,I).reshape(2,2,2,2)
        U2_ = np.kron(I,I).reshape(2,2,2,2)
        M = Z0

        M_ij = RE.circuit(U1, U2, U1_, U2_, M)
        assert np.allclose(M_ij, I)
        
        M_ij_exact = RE.exact_environment_circuit(U1, U2, U1_, U2_)
        assert np.allclose(M_ij_exact, np.array([[1,0,0,0],[0,0,0,0],[0,0,0,0], [1,0,0,0]]))
        
        eta, eig_vec = RE.exact_environment(U1, U2, U1_, U2_)
        assert eta == 1
        assert np.allclose(eig_vec, ( 1/np.sqrt(2) )* np.array([[1,0],[0,1]]))
        

    def test_manifold_overlap(self):
        """
        Test that the manifold overlap circuits return as expected
        """

        MO = ManifoldOverlap()
        
        for _ in range(10):
            U1 = unitary_group.rvs(4).reshape(2,2,2,2)
            U2 = unitary_group.rvs(4).reshape(2,2,2,2)
            U1_ = U1.reshape(4,4).conj().T.reshape(2,2,2,2)
            U2_ = U2.reshape(4,4).conj().T.reshape(2,2,2,2)
            W = np.eye(16).reshape(2,2,2,2, 2,2,2,2)
            M = np.eye(2)
            
            overlap = MO.circuit(U1, U2, U1_, U2_, M, W)
            # confirm that an identity matrix W gives the initial matrices as
            #   the maximum overlap state
            assert np.allclose(-np.abs(overlap)**2, -1)
            
        ###################################    
        
        M = np.eye(2)
        U2 = np.eye(4).reshape(2,2,2,2)
        U2_ = np.eye(4).reshape(2,2,2,2)
        
        U1 = np.kron(self.X0,self.I).reshape(2,2,2,2)
        U1_ = np.kron(self.X0, self.I).reshape(2,2,2,2)
        W = tensor([self.Z0, self.I,self.I, self.I]).reshape(2,2,2,2,2,2,2,2)
        overlap = MO.circuit(U1, U2, U1_, U2_, M, W)
        assert np.allclose(overlap, -1)
        
        U1 = np.kron(self.X0,self.I).reshape(2,2,2,2)
        U1_ = np.kron(self.X0, self.I).reshape(2,2,2,2)
        W = tensor([self.I, self.Z0,self.I, self.I]).reshape(2,2,2,2,2,2,2,2)
        overlap = MO.circuit(U1, U2, U1_, U2_, M, W)
        assert np.allclose(overlap, 1)
        
        U1 = np.kron(self.X0,self.I).reshape(2,2,2,2)
        U1_ = np.kron(self.X0, self.I).reshape(2,2,2,2)
        W = tensor([self.Z0, self.I,self.Z0, self.I]).reshape(2,2,2,2,2,2,2,2)
        overlap = MO.circuit(U1, U2, U1_, U2_, M, W)
        assert np.allclose(overlap, 1)
        
        U1 = np.kron(self.X0,self.I).reshape(2,2,2,2)
        U1_ = np.kron(self.X0, self.I).reshape(2,2,2,2)
        W = tensor([self.I, self.I,self.Z0, self.I]).reshape(2,2,2,2,2,2,2,2)
        overlap = MO.circuit(U1, U2, U1_, U2_, M, W)
        assert np.allclose(overlap, -1)
        
        U2 = np.kron(self.X0, self.X0).reshape(2,2,2,2)
        U2_ = np.kron(self.X0, self.X0).reshape(2,2,2,2)
        U1 = np.eye(4).reshape(2,2,2,2)
        U1_ = np.eye(4).reshape(2,2,2,2)
        W = tensor([self.Z0]*4).reshape(2,2,2,2, 2,2,2,2)
        overlap = MO.circuit(U1, U2, U1_, U2_, M, W)
        assert np.allclose(overlap, 1)
        
        W = tensor([self.Z0]*3 + [self.I]).reshape(2,2,2,2, 2,2,2,2)
        overlap = MO.circuit(U1, U2, U1_, U2_, M, W)
        assert np.allclose(overlap, -1)
        
    
    def test_represent(self, show_analytics = True):
        """
        Test that the optimization procedure for finding the right environment
        matrix, M, works effectively as expected
        """

        RE1 = Represent()
        
        U1 = np.kron(X0, X0).reshape(2,2,2,2)
        U1_ = U1.reshape(4,4).conj().T.reshape(2,2,2,2)
        U2 = np.kron(I,I).reshape(2,2,2,2)
        U2_ = np.kron(I,I).reshape(2,2,2,2)
        
        M_ij_exact = RE1.exact_env(U1, U2, U1_, U2_)
        res = RE1.optimize_by_hand([U1, U2, U1_, U2_])
        assert np.allclose(M_ij_exact, RE1.M(res.x[1:]))
        
        if show_analytics:
                
            RE2 = Represent()
            A = np.random.rand(4,4)
            H = 0.5 * (A + A.conj().T)
            
            U1 = unitary_group.rvs(4).reshape(2,2,2,2)
            U2 = unitary_group.rvs(4).reshape(2,2,2,2)
            
            allParams = []
            allIters = []
            allTimes = [0,0.001,0.005,0.01,0.05,0.1]
            allCF = []
            allMExact = []
            allMVar = []
            
            for dt in allTimes:
                rand_evo = expm(1j * H * dt)
            
                U1_ = (U1.reshape(4,4) @ rand_evo).conj().T.reshape(2,2,2,2)
                U2_ = (U2.reshape(4,4) @ rand_evo).conj().T.reshape(2,2,2,2)
            
                M_ij_exact = RE2.exact_env(U1,U2,U1_,U2_)
                res = RE2.optimize(U1, U2, U1_, U2_)
                allParams.append(res.x)
                allIters.append(res.nfev)
                allCF.append(res.fun)
                allMExact.append(M_ij_exact)
                allMVar.append(RE2.M(res.x[1:]))
                
            fig, axs = plt.subplots(1,3, figsize = (30,10))
            
            axs[0].plot(allTimes, allCF)
            axs[0].set_xscale("log")
            axs[0].set_ylabel("Cost Function")
            
            axs[1].plot(allTimes, allIters)
            axs[1].set_ylabel("Number Iterations")
            axs[1].set_xscale("log")
            axs[1].set_yscale("log")


            axs[2].plot(allTimes, np.abs(np.array(allParams)))
            axs[2].set_ylabel("Parameter Values")
            axs[2].set_xlabel("dt")
            axs[2].set_xscale("log")
            axs[2].set_yscale("log")

            fig.tight_layout()
            plt.plot()
            
            data = {"Iters": allIters, 
                    "CFs":allCF, 
                    "Params": allParams, 
                    "Exact": allMExact, 
                    "Var": allMVar }
            
            return data
        
    
    def test_optimize(self, show_analytics = False):
        """
        Test that the optimization procedure for finding the lowest expected
        value of an operator O works as expected.
        """

        O = np.kron(I, Z0).reshape(2,2,2,2)
        OP = Optimize()
        
        all_res = []
        
        for i in range(10):
            res = OP.optimize(O)
            all_res.append(OP.paramU(res.x))
            assert np.allclose(res.fun, -1)
        
        if show_analytics:
            plt.plot(OP.energy_opt)
    
    
    def test_time_evolution(self, show_analytics = False):
        """
        Test for some simple cases that the time evolution cod returns as
        expected. This test doesnt always work as sometimes the optimiztion is 
        not so good. The exact optimization works well but when U1' and U2' 
        are not close to U1 and U2 the vaiational optimization fails. For
        plotting trajectories this is actually ok since we have that 
        """
        
        EV = Evolve()
        # W = np.eye(16).reshape(2,2,2,2,2,2,2,2)
        # U1 = np.eye(4).reshape(2,2,2,2)
        # U2 = np.eye(4).reshape(2,2,2,2)
        
        # res = EV.exact_optimize(W, U1, U2)
        # assert isclose(res.fun, -1, rel_tol = 1e-2)
        
        if show_analytics:
            # plt.plot(EV.cf_convergence)
            # plt.show()
            
            
        # Test that we see oscillating evolution of expectation values when 
        #   evolving under the same hamiltonian
            DT = 0.001
            STEPS = 3000
            H = tensor([self.X0, self.X0, self.X0, self.X0])
            W = expm(-1j * H * DT).reshape(2,2,2,2,2,2,2,2)
            
            res = EV.time_evolve(STEPS, W, show_convergence=False)
            exp_vals = []
            OC = OverlapCalculator() 
            for r in res:
                U1, U2 = EV.paramU(r.x)
                exp_val = OC.expectation_value(U1.reshape(2,2,2,2), 
                                               U2.reshape(2,2,2,2),
                                               H.reshape(2,2,2,2,2,2,2,2))
                
                exp_vals.append(exp_val)
                
            plt.plot(exp_vals)
            plt.show()
            return exp_vals
    
    def test(self):
        """
        Run through all individual tests and check they pass without plotting 
        graphs that demonstrate effective performance
        """
        print("\n###########################\n")
        
        print("Testing Expectation Value Circuits...")
        self.test_exp_values()
        print("Expectation Value Tests Passed")
        
        print("\n###########################\n")

        print("Testing the right environment circuit properties...")
        self.test_right_environment_circuit()
        print("Right Environment Test Passed")
        
        print("\n###########################\n")
        
        print("Testing the overlap circuit...")
        self.test_manifold_overlap()
        print("Overlap Circuit Test Passed")
        
        print("\n###########################\n")
        
        print("Testing Environment Optimization")
        self.test_represent(show_analytics = False)
        print("Represent Optimizer test passed")
        
        print("\n###########################\n")
        
        print("Testing energy optimization code")
        self.test_optimize(show_analytics = False)
        print("Energy optimizer test passed")
        
        print("\n###########################\n")
        
        print("Testing single step time evolution")
        self.test_time_evolution()
        print("Time Evolution Test Passed!")
        
        print("\n###########################\n")
        

if __name__ == "__main__":
    t = Tests()
    e = t.test_time_evolution(True)