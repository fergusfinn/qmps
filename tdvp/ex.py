import numpy as np
from numpy.random import randn
from numpy import array
from scipy.linalg import expm
from scipy.linalg import block_diag, sqrtm, polar, schur

def sympmat(n):
    r""" Returns the symplectic matrix of order n

    Args:
        n (int): order
        hbar (float): the value of hbar used in the definition
            of the quadrature operators
    Returns:
        array: symplectic matrix
    """
    idm = np.identity(n)
    omega = np.concatenate((np.concatenate((0*idm, idm), axis=1),
                            np.concatenate((-idm, 0*idm), axis=1)), axis=0)
    return omega

def changebasis(n):
    r"""Change of basis matrix between the two Gaussian representation orderings.

    This is the matrix necessary to transform covariances matrices written
    in the (x_1,...,x_n,p_1,...,p_n) to the (x_1,p_1,...,x_n,p_n) ordering

    Args:
        n (int): number of modes
    Returns:
        array: :math:`2n\times 2n` matrix
    """
    m = np.zeros((2*n, 2*n))
    for i in range(n):
        m[2*i, i] = 1
        m[2*i+1, i+n] = 1
    return m

def williamson(V, tol=1e-11):
    r"""Performs the Williamson decomposition of positive definite (real) symmetric matrix.

    Note that it is assumed that the symplectic form is

    ..math:: \Omega = \begin{bmatrix}0&I\\-I&0\end{bmatrix}

    where :math:`I` is the identity matrix and :math:`0` is the zero matrix.

    See https://math.stackexchange.com/questions/1171842/finding-the-symplectic-matrix-in-williamsons-theorem/2682630#2682630

    Args:
        V (array): A positive definite symmetric (real) matrix V
        tol (float): the tolerance used when checking if the matrix is symmetric: :math:`|V-V^T| \leq tol`

    Returns:
        tuple(array,array): Returns a tuple ``(Db, S)`` where ``Db`` is a diagonal matrix
            and ``S`` is a symplectic matrix such that :math:`V = S^T Db S`
    """
    (n, m) = V.shape
    diffn = np.linalg.norm(V-np.transpose(V))

    if diffn >= tol:
        raise ValueError("The input matrix is not symmetric")
    if n != m:
        raise ValueError("The input matrix is not square")
    if n % 2 != 0:
        raise ValueError(
            "The input matrix must have an even number of rows/columns")

    n = n//2
    omega = sympmat(n)
    rotmat = changebasis(n)
    vals = np.linalg.eigvalsh(V)

    for val in vals:
        if val <= 0:
            raise ValueError("Input matrix is not positive definite")

    Mm12 = sqrtm(np.linalg.inv(V)).real
    r1 = Mm12 @ omega @ Mm12
    s1, K = schur(r1)
    X = np.array([[0, 1], [1, 0]])
    I = np.identity(2)
    seq = []

    # In what follows I construct a permutation matrix p  so that the Schur matrix has
    # only positive elements above the diagonal
    # Also the Schur matrix uses the x_1,p_1, ..., x_n,p_n  ordering thus I use rotmat to
    # go to the ordering x_1, ..., x_n, p_1, ... , p_n

    for i in range(n):
        if s1[2*i, 2*i+1] > 0:
            seq.append(I)
        else:
            seq.append(X)

    p = block_diag(*seq)
    Kt = K @ p
    s1t = p @ s1 @ p
    dd = np.transpose(rotmat) @ s1t @rotmat
    Ktt = Kt @ rotmat
    Db = np.diag([1/dd[i, i+n] for i in range(n)] + [1/dd[i, i+n]
                                                     for i in range(n)])
    S = Mm12 @ Ktt @ sqrtm(Db)
    return Db, np.linalg.inv(S).T

#n = 4 
#A = randn(2*n, 2*n)
#A = A.T@A
#S = williamson(A)[1]
#P = S@S.T

def tr_symp(P, i):
    """drc: delete matching row and column for P
    """
    n = P.shape[0]//2
    X = np.delete(P, i, axis=0)
    X = np.delete(X, i, axis=1)
    X = np.delete(X, n+i-1, axis=0)
    X = np.delete(X, n+i-1, axis=1)
    return X

def trace_out(P, js):
    js = reversed(sorted(js))
    for j in js:
        P = tr_symp(P, j)
    return P

def tr_symm(A, i):
    X = np.delete(A, i, axis=0)
    X = np.delete(X, i, axis=1)
    return X

def ses(P):
    n = P.shape[0]//2
    return np.sort(np.diag(williamson(P)[0])[:n])

def apply(f, x, n, args):
    xs = [x]
    for _ in range(n):
        xs.append(f(xs[-1], *args))
    return xs

#print("\n".join([str(ses(M)) for M in apply(tr_symp, P, n-1, (0,))]))
#print("\n".join([str(np.sort(np.linalg.eigvals(A))) for A in apply(tr_symp, P, n-1, (0,))]))

