"""
Linop Subclass for sigpy that implements the behavior of an
anisotropic operator which encodes anatomical data in its operation
"""
import numpy as np
import sigpy as sp
from sigpy import backend


class ProjectionOperator(sp.linop.Linop):
    r"""
    Projection Operator for anisotropic regularization
    Requires input data and anatomical_data

    D(x) = I - ξ(x)ξ^\top(x)
    """
    def __init__(self, ishape: tuple[int], anatomical_data: np.ndarray, eta: float = 1.):
        oshape = ishape
        self._anatomical_data = anatomical_data
        self._eta = eta
        self._xi = get_xi(anatomical_data, eta)
        super().__init__(oshape, ishape)

    @property
    def anatomical_data(self):
        """
        returns anatomical data built into operator
        """
        return self._anatomical_data

    @property
    def xi(self):
        """
        returns curent xi value
        """
        return self._xi

    @property
    def eta(self):
        """
        returns current eta value
        """
        return self._eta

    #Don't understand the need for this
    @xi.setter
    def xi(self, value: np.ndarray):
        """
        Allows for xi to be adjusted outside operator
        """
        self._xi = value

    @eta.setter
    def eta(self, value: float):
        """
        Allows for eta to be adjusted outside operator
        """
        self._eta = value
        self._xi = get_xi(self._anatomical_data, self._eta)

    @anatomical_data.setter
    def anatomical_data(self, value: np.ndarray):
        """
        Allows for anatomical_data associated with the operator to be adjusted
        """
        self._anatomical_data = value
        self._xi = get_xi(self._anatomical_data, self._eta)

    def _apply(self, input):
        device = backend.get_device(input)
        with device:
            dot_scalar = (self._xi * input).sum(axis=0)
            return input - dot_scalar * self._xi

    def _adjoint_linop(self):
        return ProjectionOperator(self.oshape, self._anatomical_data, self._eta)


def get_xi(v: np.ndarray, eta: float):
    r"""
    Calculates xi which is a normalized version of the delta of matrix v
    ξ(x) = \frac{∇v(x)}
    {\sqrt{η2 + |∇v(x)|^2}}
    """
    gradient_v = sp.linop.FiniteDifference(v.shape)(v)
    xi = gradient_v / np.sqrt(eta ** 2 + np.linalg.norm(gradient_v, axis=0) ** 2)
    # import ipdb; ipdb.set_trace()
    return xi
