import numpy as np
from typing import Union

def householder(A):

    def householder_vectorized(a):
        v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
        v[0] = 1
        tau = 2 / (v.T @ v)
        
        return v,tau

    m,n = A.shape
    R = A.copy()
    Q = np.identity(m)
    
    for j in range(0, n):
        v, tau = householder_vectorized(R[j:, j, np.newaxis])
        
        H = np.identity(m)
        H[j:, j:] -= tau * (v @ v.T)
        R = H @ R
        Q = H @ Q
        
    return Q.T, np.triu(R[:n])
    #return Q.T, np.triu(R)