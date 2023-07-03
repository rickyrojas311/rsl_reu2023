"""
Implements parts of an Anisotropic Regularizer
for image reconstruction
"""
import math
import numpy as np
import sigpy as sp


# def get_norm(matrix, num_iterations: int):
#     """
#     Uses power iteration formula to find the norm of a potentially nonsquare matrix
#     """
#     A = matrix.T @ matrix
#     b_k = np.random.rand(A.shape[1])

#     for _ in range(num_iterations):
#         b_k1 = A @ b_k
#         b_k1_norm = np.linalg.norm(b_k1)
#         b_k = b_k1 / b_k1_norm

#     return b_k1_norm

def get_xi(v: np.ndarray, eta=0.001):
    """
    Calculates xi which is a normalized version of the delta of matrix v
    """
    gradient_v = sp.linop.FiniteDifference(v.shape)(v)
    return gradient_v / math.sqrt(eta ** 2 + np.linalg.norm(gradient_v) ** 2)

def anisotropic_operator(x: np.ndarray, v: np.ndarray):
    """
    Linear Operator for anisotropic refularization
    """
    gradient_x = sp.linop.FiniteDifference(x.shape)(x)
    xi = get_xi(v)
    dot_scalar = xi @ gradient_x
    return gradient_x - dot_scalar * xi

if __name__ == "__main__":
    np.random.seed(100)
    q = np.random.rand(2,2)
    w = np.random.rand(2,2)
    print(anisotropic_operator(q,w))
