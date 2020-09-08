import numpy as np
import time
from scipy.stats import unitary_group

def pr(m):
    print(np.round(m,3))

def i(m):
    return m.reshape(4,4).conj().T.reshape(2,2,2,2)

U1, U2, U3, U4 = [unitary_group.rvs(4).reshape(2,2,2,2) for _ in range(4)]


start0 = time.time()
for _ in range(1000):
    M_ij = np.einsum(
                U1, [6,12,9,10],
                U2, [7,8,12,11],
                U3, [4,5,7,8],
                U4, [0,1,3,4],
                [0,1,9,10,6,3,11,5]
            )[0,0,0,0,:,:,:,:].reshape(4,4)
end0 = time.time()

print(f"Elapsed Time is {(end0-start0)/1000}")
A1e = np.einsum(U2, [0,3,5,6],
                U1, [1,2,3,4],
                [5,6,0,1,2,4])[0,0,...].reshape(2,4,2)

pr(M_ij) 
print("\n")

start1 = time.time()
for _ in range(1000):
    A1 = np.tensordot(U1[:,:,0,0], U2, (1,2)).reshape(2,4,2)
    A2 = np.tensordot(U3, U4[0,0,:,:], (0,1)).reshape(2,4,2)
    A12 = np.transpose(np.tensordot(A1, A2, (1,1)), [0,3,1,2]).reshape(4,4)

end1 = time.time()

print(f"Elapsed Time is {(end1-start1)/1000}")
pr(A12)

def to_matrix(U2, U1):
    