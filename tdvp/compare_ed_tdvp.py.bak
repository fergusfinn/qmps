from tdvp_fast import *
from ed import *
from misc import *

from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

import numpy as np
import pylab as pl

from scipy import sparse
from scipy.sparse.linalg import expm_multiply

import time

if __name__ == "__main__":	
	#################### Define parameters here
	L = 12
	chi = 64
	dt = 0.05
	N_steps = 100
	
	hx = 0.9045
	hz = 0.7090
	Jx = 0.4
	Jz = 0.
	
	np.random.seed(0)
	Sx_list,Sy_list,Sz_list,Sp_list = gen_spin_operators(L,0.5)
	psi_init = Sp_list[L/2]*(0.5-np.random.rand(2**L) + 1j*(0.5-np.random.rand(2**L)))
	psi_init = psi_init/np.linalg.norm(psi_init)

	############ ED ###########################
	print "ED:"
	psi = psi_init.copy()
	
	H_xx, H_x, H_z = gen_hamiltonian(Sx_list,Sy_list,Sz_list,L,0.)
	H = Jx*H_xx + hx*H_x + hz*H_z
	
	H_mid = hx*Sx_list[L/2] + hz*Sz_list[L/2]
	H_mid = H_mid + Jx*Sx_list[L/2-1]*Sx_list[L/2]/2. + Jx*Sx_list[L/2]*Sx_list[L/2+1]/2.
	
	print "E0 = ",np.real(np.dot(np.conj(psi),H_mid*psi))
	
	E_mid = []
	t_list = []
	for i in range(1,N_steps):
		psi = expm_multiply(-1j*dt*H, psi)
		E_mid.append(np.real(np.dot(np.conj(psi),H_mid*psi)))
		t_list.append(i*dt)
		print t_list[-1],E_mid[-1]
	
	pl.loglog(t_list,np.abs(E_mid),'-s')

	############ TDVP #########################
	print "TDVP:"
	Psi = compress_state(psi_init,L,2,chi)
		
	W = MPO_TFI(Jx,Jz,hx,hz)
	W_site = middle_site_hamiltonian(Jx,Jz,hx,hz,L)
		
	print "E0 = ",np.real(mpo_expectation_value(list(Psi),W_site))
	
	E_mid = [];t_list = []
	t0 = time.time()
	for i in range(1,N_steps):
		Psi,Rp_list,spectrum = tdvp(Psi, L*[W], 1j*dt/2., Rp_list=None)
		E_mid.append(np.real(mpo_expectation_value(list(Psi),W_site)))
		t_list.append(i*np.abs(dt))
		print t_list[-1],E_mid[-1]
	print time.time()-t0
	pl.loglog(t_list,np.abs(E_mid),'-^')
	
	pl.show()