"""
Practice using the sigpy library to create a downsampling operator and its transpose 
as well as verfiy that it is actually A^T
"""

import numpy as np
import sigpy as sp
import math
import downsampling_subclass as spl

def is_tranpose(ishape, factor):
    """
    Checks properties of the transpose of A to verify A.H is the transpose
    of A
    """
    vectorA_size = np.prod(ishape)
    A = spl.AverageDownsampling((vectorA_size,), factor)
    A_transpose = A.H
    x = np.random.rand(*ishape).flatten()
    A_x = A * x
    y = np.random.rand(A_x.size)
    left = np.dot(A_x, y)
    right = np.dot(x, A_transpose * y)
    return left, right, math.isclose(left, right)


if __name__ == "__main__":
    np.random.seed(100)
    # ishape = (3, 2)
    # factor = (2, 2)
    # print(high_res := np.random.rand(*ishape))
    # print(A := sp.linop.Downsample(ishape, factor))
    # print(low_res := A.apply(high_res))
    # print(upscale_res := A.H * low_res)
    # print(is_tranpose(ishape, factor))

    ishape = (5,)
    print("Starting Matrix: \n", x := np.random.rand(*ishape))
    # x = np.array([[1,2],[3,4],[5,6]])
    print(factor := (2,))
    A = spl.AverageDownsampling(ishape, factor)
    print("Downsample: \n", y := A(x))
    print("Upsample: \n", x_prime := A.H * y)
    print("Is tranpose:", is_tranpose(ishape, factor))
