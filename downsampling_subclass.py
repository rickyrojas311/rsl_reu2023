"""
Subclass for the downsampling object in scipy and associated functions
"""

import sigpy as sp
from sigpy import backend
import numpy as np

class AverageDownsampling(sp.linop.Downsample):
    """Downsampling linear operator.

    Args:
        ishape (tuple of ints): Input shape.
        factor (tuple of ints): Downsampling factor.
        shift (None of tuple of ints): Shifts before down-sampling.
    """

    def __init__(self, ishape, factors, shift=None):
        self.factors = factors
        super().__init__(ishape, factors, shift)
    # def __init__(self, ishape, factors, shift=None):
    #     self.factors = factors

    #     if shift is None:
    #         shift = [0] * len(ishape)

    #     self.shift = shift
    #     oshape = [
    #         ((i - s + f - 1) // f) for i, f, s in zip(ishape, factors, shift)
    #     ]

    #     super().__init__(oshape, ishape)


    def _apply(self, input):
        device = backend.get_device(input)
        with device:
            return downsample_average(input, self.factors)

    def _adjoint_linop(self):
        return sp.linop.Upsample(self.ishape, self.factors, shift=self.shift)
    


def downsample_average(target_array, factor):
    """
    Downsamples inputed target array by the inputed factor tuple by 
    averaging consecutive values together.
    Reads the downsampling factor in the factor tuple for each dimension
    then splits the array into buckets and averages all the values in the buckets
    together

    @parameter
    factor downsamples by given factor in direction given in the tuple
    """
    for i in range(target_array.ndim):
        arrays = []
        if i < len(factor):
            downsampling_rate = factor[i]
        else:
            #A rate of one doesn't alter the matrix in that direction
            downsampling_rate = 1
        bucket_size = np.ceil(target_array.shape[i]/downsampling_rate)
        split_array = np.array_split(target_array, bucket_size, axis=i)
        for array in split_array:
            arrays.append(np.mean(array, axis=i))
        target_array = np.stack(arrays, axis=i)
    return target_array

def upsample_linear(target_array, factor):
    """
    Given a m by n matrix 
    """


if __name__ == "__main__":
    A = np.array([[1,2],[3,4],[5,6]])
    # A = np.arange(180)
    print(downsample_average(A, ))