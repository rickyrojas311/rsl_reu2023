"""
Practice using the sigpy library to create a downsampling operator and its transpose 
as well as verfiy that it is actually A^T
"""

import numpy as np
import sigpy as sp

def is_tranpose(A, A_transpose):
    """
    Checks properties of the transpose of A to verify A.H is the transpose\
    of A
    """
    B = np.random.rand(A[0], A[1])
    k = np.random.rand(A[0])
    if A_transpose.H == A and (A + B).H == (A_transpose + B.H)\
    and (k @ A).H == k @ A_transpose and (A @ B).H == (A_transpose @ B.H):
        return True
    return False

if __name__ == "__main__":
    np.random.seed(1)
    print(high_res := np.random.rand(6, 1))
    A = sp.linop.Downsample((6, 1), (2,1))
    print(low_res := A * high_res)
    print(upscale_res := A.H * low_res)
    # print(is_tranpose(A, A.H))
