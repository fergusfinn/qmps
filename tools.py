from numpy import eye, allclose, concatenate, array, real, imag, sqrt
from numpy.random import randn
from scipy.linalg import norm, svd, null_space
from pymps.spin import Sx1, Sy1, Sz1, Sx2, Sy2, Sz2

def svals(A):
    q = svd(A)[1]
    return q/norm(q)

def is_unitary(u):
    """is_unitary: check if matrix is unitary
    """
    return allclose(eye(u.shape[0]), u@u.conj().T)

def embed(v):
    '''put a matrix into a unitary'''
    v = v.reshape(1, -1)/norm(v)
    vs = null_space(v).conj().T
    return concatenate([v, vs], 0).T

def deembed(u):
    '''take a matrix out of a unitary'''
    return (u@array([1, 0, 0, 0])).reshape(2, 2)

def mat(v):
    """mat: deserialize 2x2 matrix from a real vector
    """
    re, im = v[:4], v[4:]
    C = (re+im*1j).reshape(int(sqrt(len(v))), -1)
    return C

def demat(A):
    """demat: serialize a real vector from a matrix
    """
    re, im = real(A).reshape(-1), imag(A).reshape(-1)  
    return concatenate([re, im], axis=0)

Q = randn(2, 2)+1j*randn(2, 2)
assert is_unitary(embed(Q))
assert allclose(Q, deembed(embed(Q))*norm(Q))
assert allclose(norm(mat(demat(Q))-Q), 0)

SWAP = ((Sx1@Sx2+Sy1@Sy2+Sz1@Sz2)+eye(4))/2
