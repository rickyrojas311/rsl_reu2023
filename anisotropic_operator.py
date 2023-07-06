"""
Implements parts of an Anisotropic Regularizer
for image reconstruction
"""
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
    left = np.dot(A_x, y)
    right = np.dot(x, A_T_y)
    return left, right

if __name__ == "__main__":
    np.random.seed(100)
    i_shape = (2,2)
    x = np.array([[1,2],[3,4]])
    w = np.array([[5,6],[7,8]])
    op = ani.AnisotropicOperator(i_shape,w)
    print(is_transpose(op, i_shape, (2,2,2)))
    