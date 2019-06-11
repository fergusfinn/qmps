from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
import numpy as np
from numpy import sign, diag, real as re
import pylab as pl
from .misc import *
from scipy.sparse.linalg import onenormest

def tdvp(Psi, W, dt, Rp_list=None, k=5, a=1, b=1):
        """
                Applies TDVP on an MPS.
                a, b scale reg/adj timesteps
        """
        L = len(Psi)
        spectrum = []
        def sweep(Psi, W, dt, Lp_list, Rp_list):
                s = np.ones([1,1])
                for j in range(L):
                        # Get theta
                        theta = np.tensordot(s,Psi[j], axes = [1,1]) # a,i,b
                        theta = theta.transpose(1,0,2) # i,a,b
                        
                        # Apply expm (-dt H) for 1-site
                        d,chia,chib = theta.shape
                        H = H1_mixed(Lp_list[-1],Rp_list[L-j-1],W[j])

                        theta = evolve_lanczos(H, theta.reshape(d*chia*chib), -dt, np.min([d*chia*chib-1,k]))
                        #theta = expm_multiply(H, theta.reshape(d*chia*chib), -dt, np.min([d*chia*chib-1,k]))
                        
                        theta = theta.reshape(d*chia,chib)/np.linalg.norm(theta)

                        # SVD and update environment
                        U,s,V = np.linalg.svd(theta,full_matrices=0)

                        spectrum.append(s/np.linalg.norm(s))
                        U = U.reshape(d,chia,chib)
                        s = np.diag(s)
                        V = V.reshape(chib,chib)
                        
                        Psi[j] = U
                
                        Lp = np.tensordot(Lp_list[-1],U,axes=(0,1)) # ap,m,i,b 
                        Lp = np.tensordot(Lp,W[j],axes=([1,2],[0,2])) # ap,b,n,ip
                        Lp = np.tensordot(Lp,np.conj(U), axes=([0,3],[1,0])) # ap,n,b 
                        Lp = np.transpose(Lp,(0,2,1)) # ap,b,n 
                        Lp_list.append(Lp)
                        
                        if j < L-1:
                                # Apply expm (-dt H) for 0-site
                                
                                Psi[j+1] = np.tensordot(V,Psi[j+1],axes=(1,1)).transpose(1,0,2)
                                
                                Rp = np.tensordot(np.conj(V),Rp_list[L-j-1],axes=(1,1))
                                Rp = np.tensordot(V,Rp,axes=(1,1))
                                
                                H = H0_mixed(Lp_list[-1],Rp)
                                
                                s = evolve_lanczos(H, s.reshape(chib*chib), dt, np.min([chib*chib-1,k]))
                                #s = expm_multiply(H, s.reshape(chib*chib), dt, np.min([chib*chib-1,k]))
                                
                                s = s.reshape(chib,chib)/np.linalg.norm(s)
                                
                return Psi, Lp_list, spectrum
        
        D = W[0].shape[0]
        
        if Rp_list==None:
                Rp_list = [np.zeros([1,1,D])]; Rp_list[0][0,0,D-1] = 1
                for i in np.arange(L-1,-1,-1):
                        Rp = np.tensordot(Psi[i], Rp_list[-1], axes=(2,0)) # i a b m
                        Rp = np.tensordot(W[i], Rp, axes=([1,2],[3,0])) # m ip a b
                        Rp = np.tensordot(np.conj(Psi[i]), Rp, axes=([0,2],[1,3])) # b m a  
                        Rp = np.transpose(Rp,(2,0,1))
                        Rp_list.append(Rp)
        
        Lp_list = [np.zeros([1,1,D])]; Lp_list[0][0,0,0] = 1
        Psi, Rp_list, spectrum = sweep(Psi, W, a*dt, Lp_list,Rp_list)
        
        Psi = mps_invert(Psi)
        W = mpo_invert(W)
        
        Lp_list = [np.zeros([1,1,D])]; Lp_list[0][0,0,D-1] = 1
        Psi, Rp_list, spectrum = sweep(Psi, W, b*dt, Lp_list,Rp_list)

        Psi = mps_invert(Psi)   
        W = mpo_invert(W)
                
        return Psi,Rp_list,spectrum

class H0_mixed(object):
        def __init__(self,Lp,Rp,dtype=float):
                self.Lp = Lp # a,ap,m
                self.Rp = Rp # b,bp,n
                self.chi1 = Lp.shape[0]         
                self.chi2 = Rp.shape[0]
                self.shape = np.array([self.chi1*self.chi2,self.chi1*self.chi2])
                self.dtype = dtype
                                
        def matvec(self,x):
                x = np.reshape(x,(self.chi1,self.chi2)) # a,b
                x = np.tensordot(self.Lp,x,axes=(0,0)) # ap,m,b
                x = np.tensordot(x,self.Rp,axes=([1,2],[2,0])) # ap,bp
                x = np.reshape(x,self.chi1*self.chi2)
                return(x)

class H1_mixed(object):
        def __init__(self,Lp,Rp,M,dtype=float):
                self.Lp = Lp # a,ap,m
                self.Rp = Rp # b,bp,n
                self.M = M # m,n,i,ip
                self.d = M.shape[3]
                self.chi1 = Lp.shape[0]         
                self.chi2 = Rp.shape[0]
                self.shape = np.array([self.d*self.chi1*self.chi2,self.d*self.chi1*self.chi2])
                self.dtype = dtype
                                
        def matvec(self,x):
                x=np.reshape(x,(self.d,self.chi1,self.chi2)) # i,a,b
                x = np.tensordot(self.Lp,x,axes=(0,1)) # ap,m,i,b
                x = np.tensordot(x,self.M,axes=([1,2],[0,2])) # ap,b,n,ip               
                x = np.tensordot(x,self.Rp,axes=([1,2],[0,2])) # ap,ip,bp
                x = np.transpose(x,(1,0,2))
                x = np.reshape(x,self.d*self.chi1*self.chi2)
                return(x)
                
def evolve_lanczos(H, psiI, dt, krylovDim):
        Dim = psiI.shape[0]
        if Dim == 2:
            Z = np.zeros((2, 2), dtype=np.complex128)
            Z[0] = H.matvec(np.array([1., 0.]))
            Z[1] = H.matvec(np.array([0., 1.]))
            return expm(Z*dt)@psiI
        if Dim == 1:
            Z = np.zeros((1, 1), dtype=np.complex128)
            Z = H.matvec(np.array([1.]))
            return np.exp(Z*dt)@psiI
        
        Vmatrix = np.zeros((Dim,krylovDim),dtype=np.complex128)

        psiI = psiI/np.linalg.norm(psiI)
        Vmatrix[:,0] = psiI

        alpha = np.zeros(krylovDim,dtype=np.complex128)
        beta = np.zeros(krylovDim,dtype=np.complex128)

        w = H.matvec(psiI)

        alpha[0] = np.inner(np.conjugate(w),psiI)  
        w = w -  alpha[0] * Vmatrix[:,0]
        beta[1] = np.linalg.norm(w)
        Vmatrix[:,1] = w/beta[1]

        for jj in range(1,krylovDim-1):
                w =   H.matvec(Vmatrix[:,jj]) - beta[jj]* Vmatrix[:,jj-1]
                alpha[jj] = np.real(np.inner(np.conjugate(w),Vmatrix[:,jj]))
                w = w -  alpha[jj] * Vmatrix[:,jj]
                beta[jj+1] = np.linalg.norm(w) 
                Vmatrix[:,jj+1] = w/beta[jj+1]

        w = H.matvec(Vmatrix[:,krylovDim-1]) - beta[krylovDim-1]*Vmatrix[:,krylovDim-2]
        alpha[krylovDim-1] = np.real(np.inner(np.conjugate(w),Vmatrix[:,krylovDim-1]))    

        Tmatrix = np.diag(alpha,0) + np.diag(beta[1:krylovDim],1) + np.diag(beta[1:krylovDim],-1) 
        
        unitVector=np.zeros(krylovDim,dtype=complex)
        unitVector[0]=1.

        subspaceFinal = np.dot(expm(dt*Tmatrix),unitVector)

        psiF = np.dot(Vmatrix,subspaceFinal)
        return psiF     

def expm_multiply(A, v, time, m):
        iflag = np.array([1])
        tol = 0.0
        n = A.shape[0]
        anorm = 1
        wsp = np.zeros(7+n*(m+2)+5*(m+2)*(m+2),dtype=complex)
        iwsp = np.zeros(m+2,dtype=int)
        
        output_vec,tol0,iflag0 = zgexpv(m,time,v,tol,anorm,wsp,iwsp,A.matvec,0)
        return output_vec

def MPO_TFI(Jx,Jz,hx,hz, Sx=None, Sz=None):
        if Sx is None or Sz is None:
            Id = np.eye(2, dtype = float)
            Sx = np.array( [[0., 1.], [1., 0.]])
            Sz = np.array( [[1., 0.], [0., -1.]])
            d = 2
        else:
            assert Sx is not None and Sz is not None
            print(Sx.shape, Sz.shape)
            d = len(Sx)
            Id = np.eye(d, dtype = float)

        chi = 4
        W = np.zeros((chi, chi, d, d))
        W[0,0] += Id    
        W[0,1] += Sz
        W[0,2] += Sx 
        W[0,3] += hz*Sz + hx*Sx
                
        W[1,3] += Jz*Sz 
        W[2,3] += Jx*Sx 
        W[3,3] += Id
        
        return W

def MPO_XXZ(Jp,Jz,hx=0,hy=0,hz=0, Sx=None, Sy=None, Sz=None):
    if Sx is None or Sy is None or Sz is None:
        d = 2
        S0 = np.eye(d)
        Sp = np.array([[0.,1.],[0.,0.]])
        Sm = np.array([[0.,0.],[1.,0.]])
        Sz = np.array([[0.5,0.],[0.,-0.5]])
        w_list = []
    else:
        assert Sx is not None and Sy is not None and Sz is not None
        d = len(Sx)
        S0 = np.eye(d)
        Sp = Sx+1j*Sy
        Sm = Sx-1j*Sy

    w = np.zeros((5,5,d,d),dtype=np.complex128)
    w[0,:4] = [S0,Sp,Sm,Sz]
    w[0:,4] = [hx*Sx+hy*Sy+hz*Sz, Jp/2.*Sm, Jp/2.*Sp, Jz*Sz, S0]
    return w

def middle_bond_hamiltonian(Jx,Jz,hx,hz,L): 
        """" Returns the spin operators sigma_x and sigma_z for L sites """
        sx = np.array([[0.,1.],[1.,0.]])
        sz = np.array([[1.,0.],[0.,-1.]])
        
        H_bond = Jx*np.kron(sx,sx) + Jz*np.kron(sz,sz) 
        H_bond = H_bond + hx/2*np.kron(sx,np.eye(2)) +  hx/2*np.kron(np.eye(2),sx)
        H_bond = H_bond + hz/2*np.kron(sz,np.eye(2)) +  hz/2*np.kron(np.eye(2),sz)
        H_bond = H_bond.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4) #i1 i2 i1' i2' -->
        U,s,V  = np.linalg.svd(H_bond)
        
        M1 = np.dot(U,np.diag(s)).reshape(2,2,1,4).transpose(2,3,0,1)
        M2 = V.reshape(4,1,2,2)
        M0 = np.tensordot(np.tensordot([1],[1],axes=0),np.eye(2),axes=0)
        W = []

        for i in range(L):
                if i == L/2-1:
                        W.append(M1)
                elif i == L/2:
                        W.append(M2)
                else:
                        W.append(M0)
        return W
        
def middle_site_hamiltonian(Jx,Jz,hx,hz,L):
        M0 = np.tensordot(np.tensordot([1],[1],axes=0),np.eye(2),axes=0)
        M1 = MPO_TFI(0,0,0,0)[0:1,:,:,:]
        M2 = MPO_TFI(Jx/2.,Jz/2.,hx,hz)
        M3 = MPO_TFI(Jx/2.,Jz/2.,0,0)[:,3:4,:,:]
        
        W = []
        for i in range(L):
                if i == L/2-1:
                        W.append(M1)
                elif i == L/2:
                        W.append(M2)
                elif i == L/2+1:
                        W.append(M3)
                else:
                        W.append(M0)
        return(W)
