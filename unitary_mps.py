from numpy import tensordot as td, array, eye, copy, trace, sum, real
from scipy.linalg import norm, polar
from numpy.random import randn
from numpy.linalg import qr
from tdvp.tdvp_fast import MPO_TFI, MPO_XXZ
from pymps.spin import spins
from pymps.ncon import ncon
from pymps.tensor import H as cT, T, C as c
from pymps.iMPS import iMPS
W = MPO_TFI(0, 1, 0., 0.)
Sx, Sy, Sz = spins(0.5)
#W = MPO_XXZ(0.5, 0.2)
#print(td(U, U.conj().transpose(2, 3, 0, 1), [[-2, -1], [0, 1]]).reshape(4, 4)) 

def to_canonical_mps(U):
    """to_mps: take a unitary U and return [AL, AR, C]
    """
    U = U.reshape(2, 2, 2, 2)
    z = array([1, 0])

    # take a unitary, return [AL, AR, C]
    AL = td(U, z, [3, 0]).transpose([0, 2, 1]) # -> (s, j, i) -> (s, i, j)
    AR = td(U, z, [2, 0]).transpose([1, 0, 2]) # -> (i, s, j) -> (s, i, j)

    assert allclose(norm(AL-T(AR)), 0)

    def obj(v):
        C = mat(v)
        return norm(AL@C-C@AR)

    res = minimize(obj, randn(8), method='Powell')
    P = mat(res.x)
    P /= norm(P)

    A = iMPS([copy(AR)])
    _, l, r = A.transfer_matrix().eigs()
    L, R = cholesky(l), cholesky(r)
    Q = L@R
    assert allclose(svals(Q)/norm(svals(Q)), svals(P/norm(svals(P))))
    #print(res.fun, svals(Q)/norm(svals(Q)), svals(P)/norm(svals(P)), sep='\n')
    # P comes from minimization procedure, Q from eigenvalues. 
    # P is correct, (in that it pulls through) but only unitarily the same as Q
    # Q doesn't pull through?

    assert res.fun<1e-10

    assert allclose(sum(AL@cT(AL), 0), eye(2))
    assert not allclose(sum(AR@cT(AR), 0), eye(2))

    assert allclose(sum(cT(AR)@AR, 0), eye(2))
    assert not allclose(sum(cT(AL)@AL, 0), eye(2))
    return [P, AL, AR]

def left(L, A, W):
    return ncon([L, A, cT(A), W], [[1, 2, 3], [4, 1, -1], [5, 3, -3], [2, -2, 4, 5]])

def right(R, A, W):
    return ncon([A, cT(A), W, R], [[4, -1, 1], [5, -3, 3], [-2, 2, 4, 5], [1, 2, 3]])

def to_mps_tensor(U, lr='l'):
    z = array([1, 0])
    if lr=='l':
        A = td(U, z, [3, 0]).transpose([0, 2, 1]) # -> (s, j, i) -> (s, i, j)
    elif lr=='r':
        A = td(U, z, [2, 0]).transpose([1, 0, 2]) # -> (i, s, j) -> (s, i, j)
    return A

def optimize(W):
    """optimize: optimize ground state energy unitarily using mpo H
    """
    U = qr(randn(4, 4)+1j*randn(4, 4))[0].reshape(2, 2, 2, 2)
    L = ncon([array([1, 0, 0, 0]), eye(2)], [[-2], [-1, -3]])
    R = ncon([array([0, 0, 0, 1]), eye(2)], [[-2], [-1, -3]])

    def update_left(L, U, W):
        """update_left: update the left MPO env (L) (up, mid, down)
        """
        return left(L, to_mps_tensor(U, 'l'), W)

    def update_right(R, U, W):
        """update_right: update the right MPO env (R) (up, mid, down)
        """
        return right(R, to_mps_tensor(U, 'l'), W)

    def energy(L, R):
        return real(ncon([L, R], [[1, 2, 3], [1, 2, 3]]))
    
    def update_U(U, L, R):
        A = to_mps_tensor(U, 'l')
        V = ncon([L, A, W, cT(A), R], [[-1, 1, 2], [-4, -3, 3], [1, 4, 5, -2], [5, 2, 6], [3, 4, 6]]).reshape(4, 4)
        return (polar(V)[0]@U.reshape(4, 4)).reshape(2, 2, 2, 2)

    block = 100
    for _ in range(block):
        L0, R0 = copy(L), copy(R)
        L, R = update_left(L, U, W), update_right(R, U, W)
        print(norm(L-L0), norm(R-R0))

    n_iters = 50
    for _ in range(n_iters):
        U = update_U(U, L, R)
        L, R = update_left(L, U, W), update_right(R, U, W)
    return U

if __name__=='__main__':
    A = iMPS([to_mps_tensor(optimize(W))], canonical='l')
    print('EVs')
    print(A.E(Sx), A.E(Sy), A.E(Sz))
