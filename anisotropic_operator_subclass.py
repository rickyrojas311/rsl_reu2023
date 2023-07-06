"""
Linop Subclass for sigpy that implements the behavior of an
anisotropic operator which encodes anatomical data in its operation
"""
import numpy as np
import sigpy as sp
from sigpy import backend
from downsampling_practice import is_transpose

class AnisotropicOperator(sp.linop.Linop):
    """
    Linear Operator for anisotropic regularizer
    Requires input data and anatomical_data
    """
    def __init__(self, ishape: tuple[int], anatomical_data: np.ndarray):
        oshape = ishape
        self.a_data = anatomical_data
        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = backend.get_device(input)
        with device:
            return anisotropic_operator(input, self.a_data, self.ishape)

    def _adjoint_linop(self):
        return AnisotropicAdjoint(self.ishape, self.a_data)
        
def get_xi(v: np.ndarray, eta=0.001):
    r"""
    Calculates xi which is a normalized version of the delta of matrix v
    ξ(x) = \frac{∇v(x)}
    {\sqrt{η2 + |∇v(x)|^2}}
    """
    gradient_v = sp.linop.FiniteDifference(v.shape)(v)
    return gradient_v / np.sqrt(eta ** 2 + np.linalg.norm(gradient_v, axis=0) ** 2)


def projection_operator(x: np.ndarray, v: np.ndarray):
    r"""
    Linear Operator for anisotropic regularizer of form
    D(x) = I - ξ(x)ξ^\top(x)
    """
    xi = get_xi(v)
    dot_scalar = xi @ x
    return x - dot_scalar * xi

def anisotropic_operator(x: np.ndarray, v: np.ndarray, ishape):
    """
    Combines D(G(x)) where G(x) is the sigpy gradient operator
    FiniteDifference
    And D(x) is a projection operator
    """
    gradient_x = sp.linop.FiniteDifference(ishape)(x)
    return projection_operator(gradient_x, v)

class AnisotropicAdjoint(sp.linop.Linop):
    """
    The adjoint operator class for AnisotropicOperator
    """
    def __init__(self, ishape: tuple[int], anatomical_data: np.ndarray):
        oshape = ishape
        self.a_data = anatomical_data
        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = backend.get_device(input)
        with device:
            return anisotropic_adjoint_operator(input, self.a_data, self.oshape)
        
    def _adjoint_linop(self):
        return AnisotropicOperator(self.ishape, self.a_data)

def anisotropic_adjoint_operator(iarray, v, oshape):
    """
    Adjoint Operator for anisotropic_operator class
    """
    projection_trans = projection_operator(iarray, v)
    gradient_op_trans = sp.linop.FiniteDifference(oshape).H
    return gradient_op_trans(projection_trans)
