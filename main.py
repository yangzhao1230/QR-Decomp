import numpy as np
from mgs import mgs
from householder import householder

np.set_printoptions(precision=4, suppress=True)

if __name__ == '__main__':


    A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 7],
              [4, 2, 3],
              [4, 2, 2]], dtype=np.float64)
    
    Q1, R1 = mgs(A)
    print('Q of mgs:')
    print(Q1)
    print('R of mgs')
    print(R1)

    Q2, R2 = householder(A)
    print('Q of householder:')
    print(Q2)
    print('R of householder')
    print(R2)

