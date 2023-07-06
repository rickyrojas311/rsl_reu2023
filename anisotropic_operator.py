"""
Testing enviroment for anisotropic_operator_subclass
"""
import math
import numpy as np
import anisotropic_operator_subclass as ani


def is_transpose(op, ishape, oshape):
    """
    Checks properties of the transpose of A to verify A.H is the transpose
    of A
    """
    x = np.random.rand(*ishape)
    y = np.random.rand(*oshape)
    A_x = op(x)
    A_T = op.H
    A_T_y = A_T(y)
    left = np.vdot(A_x, y)
    right = np.vdot(x, A_T_y)
    return left, right, math.isclose(left, right)

if __name__ == "__main__":
    # np.random.seed(100)
    i_shape = (240, 240)
    x = np.random.rand(*i_shape)
    w = np.random.rand(*i_shape)
    op = ani.AnisotropicOperator(i_shape, w)
    print(is_transpose(op, i_shape, (2,*i_shape)))
    