import numpy as np

def mgs(A):
    (num_rows, num_cols) = np.shape(A)

    Q = np.empty([num_rows, num_rows])
    cnt = 0

    for a in A.T:
        u = np.copy(a)
        for i in range(0, cnt):
            proj = np.dot(np.dot(Q[:, i].T, a), Q[:, i])
            u -= proj

        e = u / np.linalg.norm(u)
        Q[:, cnt] = e

        cnt += 1 

    R = np.dot(Q.T, A)

    # return Q, R
    return Q[:,0:num_cols], R[0:num_cols,:]