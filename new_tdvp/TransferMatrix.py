import numpy as np

class TransferMatrix():
    def __init__(self, Us, Uts):
        """
        Us = [U1, U2, U1_, U2_]
        Uts = [Utupper, Utlower]

        This class implements as a circuit the Mixed Transfer Matrix E^{A1}_{A2}:

        0   0   0   0   j0 ----|
        |U1 |   |U1 |   |      | = A1
        i0  |U2 |   |U2 |  ____|
            |Ut1|   |Ut1|  ----|
        i1  |   |   |   j1     | = H
        |Ut2|   |Ut2|      ____|
        i2  |   |   |   j2 ----|
            |   |   |   |
        i3  |U2 |   |U2 |      |
        |U1 |   |U1 |   |      |= A2
        |   |   |   |   j3     |
        0   0   0   0      ----|


        A1 =  i0 ---A1--- j0  = A1[i0,j0,a,k]
                   ||| | 
                    k  a    <-- keep a separate so the same code can be used to produce A2

        """
        self.Us = np.vectorize(np.reshape)(Us, (2,2,2,2))
        self.Uts = np.vectorize(np.reshape)(Uts, (2,2,2,2))
        
    @property
    def A1(self):
        U1, U2 = self.Us[0:2]
        U1 = U1[:,:,0,0]
        return np.einsum(U2, [1,2,5,6],
                         U2, [3,4,7,8],
                         U1, [0,5],
                         U1, [6,7],
                         [0,8,4,1,2,3]).reshape(2,2,2,8)

    @property
    def A2(self):
        U1_, U2_ = self.Us[2:4]
        U1_ = U1_.reshape(4,4).conj().T.reshape(2,2,2,2)
        U2_ = U2_.reshape(4,4).conj().T.reshape(2,2,2,2)
        pass

    