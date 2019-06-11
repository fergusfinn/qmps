from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

import numpy as np

from scipy import sparse
from scipy.sparse.linalg import expm_multiply

def site_spin_operators(S):
	d = int(np.rint(2*S + 1))
	dz = np.zeros(d)
	mp = np.zeros(d-1)

	for n in range(d-1):
		dz[n] = S - n
		mp[n] = np.sqrt((2*S - n)*(n + 1))

	dz[d - 1] = -S
	Sp = np.diag(mp,1)
	Sm = np.diag(mp,-1)
	Sx = 0.5*(Sp + Sm)
	Sy = -0.5j*(Sp - Sm)
	Sz = np.diag(dz)
	return Sp,Sm,2*Sx,2*Sy,2*Sz

def gen_spin_operators(L,S): 
	Sp,Sm,Sx,Sy,Sz = site_spin_operators(S)
	d = Sz.shape[0]

	Sx_list = []
	Sy_list = []
	Sz_list = []
	Sp_list = []

	for i_site in range(L): 
		if i_site==0: 
			X = sparse.csr_matrix(Sx)
			Y = sparse.csr_matrix(Sy)
			Z = sparse.csr_matrix(Sz) 
			P = sparse.csr_matrix(Sp) 
		else: 
			X = sparse.csr_matrix(np.eye(d)) 
			Y = sparse.csr_matrix(np.eye(d)) 
			Z = sparse.csr_matrix(np.eye(d))
			P = sparse.csr_matrix(np.eye(d))
		for j_site in range(1,L): 
			if j_site==i_site: 
				X=sparse.kron(X,Sx, 'csr')
				Y=sparse.kron(Y,Sy, 'csr')
				Z=sparse.kron(Z,Sz, 'csr')
				P=sparse.kron(P,Sp, 'csr') 
			else: 
				X=sparse.kron(X,np.eye(d),'csr') 
				Y=sparse.kron(Y,np.eye(d),'csr') 
				Z=sparse.kron(Z,np.eye(d),'csr') 
				P=sparse.kron(P,np.eye(d),'csr') 
				
		Sx_list.append(X)
		Sy_list.append(Y)
		Sz_list.append(Z) 
		Sp_list.append(P)

	return Sx_list,Sy_list,Sz_list,Sp_list

def gen_hamiltonian(Sx_list,Sy_list,Sz_list,L,W): 
	H_xx = sparse.csr_matrix((2**L,2**L))
	H_x = sparse.csr_matrix((2**L,2**L))
	H_z = sparse.csr_matrix((2**L,2**L))

	for i in range(L-1):
		H_xx = H_xx + (1+W*(0.5-np.random.rand()))*Sx_list[i]*Sx_list[np.mod(i+1,L)]

	for i in range(L):
		H_z = H_z + (1+W*(0.5-np.random.rand()))*Sz_list[i]
		H_x = H_x + (1+W*(0.5-np.random.rand()))*Sx_list[i]

	return H_xx, H_x, H_z 
	
def compress_state(psi,L,d,chi_max):
	A,s,V = np.linalg.svd(np.reshape(psi,[d,d**(L-1)]),full_matrices=0)

	A_list = []
	chi = np.min([np.sum(s>10.**(-12)), chi_max])
	A_list.append(np.reshape(A[:,:chi],(d,1,chi)))
	for i in range(1,L-1):
		psi = np.tensordot(np.diag(s),V,axes=(1,0))[:chi,:]

		A,s,V = np.linalg.svd(np.reshape(psi,[chi*d,d**(L-i-1)]),full_matrices=0)
		
		A = np.reshape(A,[chi,d,-1])
		chi = np.min([np.sum(s>10.**(-12)), chi_max])
		A_list.append(np.transpose(A[:,:,:chi],(1,0,2)))

	A_list.append(np.reshape(np.transpose(np.tensordot(np.diag(s),V,axes=(1,0))[:chi,:],(1,0)),(d,chi,1)))

	return A_list
