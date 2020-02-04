import numpy as np
from numpy.random import randn
from xmps.iMPS import iMPS, Map
from xmps.spin import U4
from qmps.ground_state import Hamiltonian
from itertools import permutations
from scipy.optimize import minimize

dt = 1e-1
H = Hamiltonian({'ZZ': 1, 'X':1}).to_matrix()

A = iMPS().random(2, 2).left_canonicalise()
B = (A + dt*A.dA_dt([H])).left_canonicalise()
EAB = Map(A[0], B[0])

e, r = EAB.right_fixed_point()

def vec(p):
    return U4(p)[0]

def objective(p, EAB=EAB, η=1):
    EAB = EAB.asmatrix()
    er, eim, p1, p2 = p[0], p[1], p[2:17], p[17:]
    e = er+1j*eim
    #v = EAB@vec(p1)-e*vec(p1)
    v = vec(p1)
    return np.real(v.conj().T@EAB.conj().T@EAB@v) + np.abs(e)**2 - 2*np.real(e*v.conj().T@EAB@v)+ η*np.abs((1-e))**2
    #return np.real(v.conj().T@v) + η*np.abs((1-e))**2

def abs_objective(p, EAB=EAB, η=1):
    EAB = EAB.asmatrix()
    er, eim, p1, p2 = p[0], p[1], p[2:17], p[17:]
    e = er+1j*eim
    #v = EAB@vec(p1)-e*vec(p1)
    v = vec(p1)
    return np.real(v.conj().T@EAB.conj().T@EAB@v) + np.abs(e)**2 - 2*np.abs(e)*np.abs(v.conj().T@EAB@v)+ η*np.abs((1-e))**2
    #return np.real(v.conj().T@v) + η*np.abs((1-e))**2

x = randn(17)
x[0] = 1
x[1] = 0
res = minimize(objective, randn(17), options={'disp': True})
λ, result_vector = res.x[0]+1j*res.x[1], vec(res.x[2:17])

abs_res = minimize(abs_objective, randn(17), options={'disp': True})
abs_λ, abs_result_vector = abs_res.x[0]+1j*abs_res.x[1], vec(abs_res.x[2:17])

def dephase(v):
    return v/np.exp(1j*np.angle(v[0]))

print('eigenvectors')
print('actual: ', r.reshape(-1).real)
print('variational: ', dephase(result_vector).real)
print('quantum variational: ', dephase(abs_result_vector).real)
print('\n')
print(np.abs(e), np.abs(λ), λ*result_vector, EAB.asmatrix()@result_vector, sep='\n')
print('\n')
print(np.abs(e), np.abs(abs_λ), abs_λ*abs_result_vector, EAB.asmatrix()@abs_result_vector, sep='\n')
print('\n')
print(objective(res.x), objective(abs_res.x))
