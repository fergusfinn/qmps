from numpy import array, allclose, sqrt, zeros, reshape
from numpy import tensordot, kron, identity, diag, arange
from itertools import product
from functools import reduce
from math import log as logd, sqrt


def levi_civita(dim):
    """levi_civita symbol rank dim
    https://bitbucket.org/snippets/lunaticjudd/doqp7/python-implementation-of-levi-civita

    :param dim:
    """
    def perm_parity(a,b):
        """Modified from
        http://code.activestate.com/recipes/578236-alternatve-generation-of-the-parity-or-sign-of-a-p/"""
        
        a = list(a)
        b = list(b)

        if sorted(a) != sorted(b): return 0
        inversions = 0
        while a:
            first = a.pop(0)
            inversions += b.index(first)
            b.remove(first)
        return -1 if inversions % 2 else 1

    def loop_recursive(dim,n,q,s,paritycheck):
        if n < dim:
            for x in range(dim):
                q[n] = x
                loop_recursive(dim,n+1,q,s,paritycheck)
        else:
            s.append(perm_parity(q,paritycheck))
    qinit = zeros(dim)
    paritycheck = range(dim)
    flattened_tensor = []
    loop_recursive(dim,0,qinit,flattened_tensor,paritycheck)

    return reshape(flattened_tensor,[dim]*dim)

def tensor(ops):
    return reduce(kron, ops)

def n_body(op, i, n, d=None):
    """n_body: n_body versions of local operator

    :param op: operator to tensor into chain of identities
    :param i: site for operator. 1-indexed
    :param n: length of chain
    :param d: local dimension of identities to tensor. If None, use size of op
    """
    i = i-1
    if d is None:
        d = op.shape[0]
        l = [identity(d)*(1-m) + op*m for m in map(lambda j: int(not i-j), range(n))]
    else:
        l = [identity(d) for _ in range(n+1-int(logd(op.shape[0], d)))]
        l.insert(i+1, op)
        #l = [op if j==i else identity(d) for j in range(n-int(logd(op.shape[0], d)))]
    if not i < n:
        raise Exception("i must be less than n")
    return tensor(l)

def spins(S):
    """spins. returns [Sx, Sy, Sz] for spin S

    :param S: spin - must be in [0.5, 1, 1.5]
    """
    def spin(S, i):
        """i=0: Sx
           i=1: Sy
           i=2: Sz
           """
        if S == 1/2:
            if i == 0:
                return 1/2*array([[0, 1], 
                                  [1, 0]])
            if i == 1: 
                return 1/2j*array([[0  , 1]   ,
                                   [-1 , 0]])
            if i == 2:
                return 1/2*array([[1 , 0 ] ,
                                  [0 , -1]] )
        if S == 1:
            if i == 0:
                return 1/sqrt(2)*array([[0, 1, 0],
                                        [1, 0, 1], 
                                        [0, 1, 0]])
            if i == 1:
                return -1j/sqrt(2)*array([[0 , 1 , 0],
                                          [-1, 0 , 1],
                                          [0 , -1, 0]])
            if i == 2:
                return array([[1, 0, 0 ], 
                              [0, 0, 0 ], 
                              [0, 0, -1]])
        if S == 3/2:
            if i == 0:
                return 1/2*array([[0       , sqrt(3) , 0       , 0      ],
                                  [sqrt(3) , 0       , 2       , 0      ],
                                  [0       , 2       , 0       , sqrt(3)] ,
                                  [0       , 0       , sqrt(3) , 0      ]])
            if i == 1:
                return 1/2j*array([[0        , sqrt(3) , 0        , 0       ],
                                   [-sqrt(3) , 0       , 2        , 0       ],
                                   [0        , -2      , 0        , sqrt(3) ],
                                   [0        , 0       , -sqrt(3) , 0       ]])
            if i == 2:
                return array([[3/2 , 0   , 0    , 0   ],
                              [0   , 1/2 , 0    , 0   ],
                              [0   , 0   , -1/2 , 0   ],
                              [0   , 0   , 0    , -3/2]])

    def arc(x):
        return array(list(x))

    def Cp(j, m): return sqrt((j-m)*(j+m+1))
    def Cm(j, m): return sqrt((j+m)*(j-m+1))
    def Sp(j): return diag(arc(Cp(j, m) for m in arange(j-1, -j-1, -1)), 1)
    def Sm(j): return diag(arc(Cm(j, m) for m in arange(j, -j, -1)), -1)

    def Sx(j): return (Sp(j)+Sm(j))/2
    def Sy(j): return (Sp(j)-Sm(j))/2j
    def Sz(j): return diag(arc(arange(j, -j-1, -1)))

    return (Sx(S), Sy(S), Sz(S))

def ladders(S):
    """ladders

    :param S: spin
    returns: list: [S_-, S_+]
    """
    def ladder(S, pm):
        """ladder: return S_+ and S_- for given S

        :param S: spin
        """
        if S == 1/2:
            if pm == 1:
                return array([[0, 1], [0, 0]])
            if pm == -1:
                return array([[0, 0], [1, 0]])
        if S == 1:
            if pm == 1:
                return sqrt(2)*array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
            if pm == -1:
                return sqrt(2)*array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        if S == 3/2:
            if pm == 1:
                return array([[0 , sqrt(3) , 0 , 0      ],
                              [0 , 0       , 2 , 0      ],
                              [0 , 0       , 0 , sqrt(3)],
                              [0 , 0       , 0 , 0      ]])
            if pm == -1:
                return array([[0       , 0 , 0       , 0],
                              [sqrt(3) , 0 , 0       , 0],
                              [0       , 2 , 0       , 0],
                              [0       , 0 , sqrt(3) , 0]])
    return [ladder(S, pm) for pm in [-1, 1]]

def N_body_spins(S, i, N):
    """N_body_spiNs: S_x^i etc. -> local spiN operators with ideNtities 
       teNsored iN oN either side

    :param S: spiN
    :param i: site for local spiN operator: 1-iNdexed
    :param N: leNgth of chaiN 
    """
    return [n_body(s, i, N) for s in spins(S)]

def N_body_ladders(S, i, N):
    """N_body_ladders: S_+^i etc. -> local spiN ladder operators 
       with ideNtities teNsored iN oN either side

    :param S: spiN
    :param i: site for local spiN operator: 1-iNdexed
    :param N: leNgth of chaiN 
    """
    return [n_body(s, i, N) for s in ladders(S)]

def comm(A, B):
    return A@B - B@A

def acomm(A, B):
    return A@B + B@A

def CR(Sx, Sy, Sz):
    """CR: Determine if a set of spin operators satisfy spin commutation relations
    """
    S = [Sx, Sy, Sz]
    satisfied = True 
    eps = levi_civita(3)
    for j, k in product(range(3), range(3)):
        satisfied = satisfied and allclose(comm(S[j], S[k]), 
                                           tensordot(eps[j, k]*1j, S, [0, 0]))
    return satisfied   

class spinHamiltonians(object):
    """1d spin Hamiltonians"""
    def __init__(self, S, finite=True):
        """__init__"""
        self.Sx = lambda i:   N_body_spins(S, i, 2)[0] 
        self.Sy = lambda i:   N_body_spins(S, i, 2)[1]
        self.Sz = lambda i:   N_body_spins(S, i, 2)[2] 
        self.finite = finite

    def nn_general(self, Jx, Jy, Jz, hx, hy, hz):
        """nn_general: nn spin model with all nn couplings (Jx, Jy, Jz)
           and fields (hx, hy, hz). All couplings positive by default
        """
        Sx, Sy, Sz = self.Sx, self.Sy, self.Sz
        h_bulk = Jx * Sx(1) @ Sx(2) + \
                 Jy * Sy(1) @ Sy(2) + \
                 Jz * Sz(1) @ Sz(2) + \
                 hx * Sx(1) + \
                 hy * Sy(1) + \
                 hz * Sz(1) + \
                 hx * Sx(2) + \
                 hy * Sy(2) + \
                 hz * Sz(2)

        if self.finite:
            N = self.N
            h_end = h_bulk + \
                     hx * Sx(2) + \
                     hy * Sy(2) + \
                     hz * Sz(2)

            return [h_bulk]*(N-2) + [h_end]
        else:
            return h_bulk

    def heisenberg_ferromagnet(self, J):
        """heisenberg_ferromagnet: -J \sum S_{i} S_{i+1}
        """
        return self.nn_general(-J, -J, -J, 0, 0, 0)

    def heisenberg_antiferromagnet(self, J):
        """heisenberg_antiferromagnet: J \sum S_{i} S_{i+1} 
        """
        return self.nn_general(J, J, J, 0, 0, 0)
    
    def XY(self, gamma):
        """XY: H = -\sum_{i} [(1+gamma) S_{i}^x S_{i+1}^x + (1-gamma) S_{i}^y S_{i+1}^y]
        """
        return self.nn_general(1+gamma, 1-gamma, 0, 0, 0, 0)

    def XXZ(self, J, delta):
        """XXZ: H = \sum_{i} J (S^x_{i} S^x_{i+1} + S^y_{i} S^y_{i+1}) + \Delta S^z_{i} S^z_{i+1}
        """
        return self.nn_general(J, J, delta, 0, 0, 0)

    def TFIM(self, l):
        """TFIM: H = -\sum_{i} [S_i^x + l S_{i}^z S_{i+1}^z]
        """
        return self.nn_general(0, 0, -l, -1, 0, 0)

    def AKLT(self):
        SS = self.heisenberg_antiferromagnet(1)
        if not self.finite:
            return 1/2 * SS + 1/6 * SS@SS  + 1/3
        else:
            N = self.N
            return [1/2*SS + 1/6 * SS@SS + 1/3]*(N-1)

assert all([CR(*spins(S)) for S in [0.5, 1, 1.5, 2., 2.5, 3]])
