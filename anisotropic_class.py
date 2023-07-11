"""
Implements subclass for Anatomically guided reconstruction using linear least
squares regressor from sigpy and other subclasses
Setters allow for dynamic setting
"""
from typing import Any
import numpy as np
import sigpy as sp
import nibabel as nib

import downsampling_subclass as spl
import projection_operator_subclass as proj

class AnatomicReconstructor():
    """
    class to facilate anatomic reconstruction, apply on low res data
    """
    def __init__(self, anatomical_data: np.ndarray, downsampling_factor: tuple[int], given_lamda: float, given_eta: float, max_iter: int, normalize: bool=True) -> None:
        """
        Pass in needed information to set up reconstruction
        """
        if normalize:
            self._anatomical_data = normalize_matrix(anatomical_data)
        else:
            self._anatomical_data = anatomical_data

        self._downsampling_factor = downsampling_factor
        self._given_lamda = given_lamda
        self._given_eta = given_eta
        self._max_iter = max_iter
        self._normalize = normalize

        self._downsampler = spl.AverageDownsampling(self.shape, *downsampling_factor)

    @property
    def anatomical_data(self):
        """
        returns anatomical data built into operator
        """
        return self._anatomical_data
    
    @property
    def downsampling_factor(self):
        """
        return set downsampling_factor
        """
        return self._downsampling_factor
    
    @property
    def given_lambda(self):
        """
        return set lambda
        """
        return self._given_lamda
    
    @property
    def given_eta(self):
        """
        returns current eta value
        """
        return self._given_eta

    @property
    def max_iter(self):
        """
        returns level of iterations set
        """
        return self._max_iter

    @property
    def normalize(self):
        """
        sets if the AnatomicReconstructor will normalize data by default
        """
        return self._normalize


    @anatomical_data.setter
    def anatomical_data(self, value: np.ndarray):
        """
        Allows for anatomical data to be adjusted 
        """
        if self._normalize:
            self._anatomical_data = normalize_matrix(value)
        else:
            self._anatomical_data = value

    @downsampling_factor.setter
    def downsampling_factor(self, value: tuple[int]):
        self._downsampling_factor = value
    
    @given_lambda.setter
    def given_lambda(self, value: float):
        self._given_lamda = value

    @given_eta.setter
    def given_eta(self, value: float):
        self._given_eta = value

    @max_iter.setter
    def max_iter(self, value: int):
        self._max_iter = value

    def __call__(self, input) -> Any:
        

def normalize_matrix(matrix):
    """
    normalizes inputed matrix so that all of it's values range from 0 - 1
    """
    m_max = matrix.max()
    return matrix/m_max
